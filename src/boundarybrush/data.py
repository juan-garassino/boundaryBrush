"""Oxford-IIIT Pet loading, the 128px tensor cache, augmentation and copy-paste compositing.

The cache stores the *trimap* (1 pet, 2 background, 3 undefined border band) so targets and
loss weights are derived after augmentation: target = trimap != 2, weight = band_weight on 3.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as TF

from boundarybrush.ops import normalize, resize_labels, resize_rgb

logger = logging.getLogger(__name__)

BACKGROUND = 2
BORDER = 3


class PetSource(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray]: ...


class OxfordPetSource:
    """torchvision's OxfordIIITPet as (rgb uint8 HxWx3, trimap uint8 HxW) pairs."""

    def __init__(self, root: Path, split: str, download: bool = False):
        from torchvision.datasets import OxfordIIITPet

        self._ds = OxfordIIITPet(root, split=split, target_types="segmentation", download=download)

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        image, trimap = self._ds[i]
        return np.asarray(image.convert("RGB")), np.asarray(trimap, dtype=np.uint8)


def build_cache(source: PetSource, size: int) -> dict[str, torch.Tensor]:
    images, trimaps = [], []
    for i in range(len(source)):
        image, trimap = source[i]
        images.append(torch.from_numpy(resize_rgb(image, size)).permute(2, 0, 1))
        trimaps.append(torch.from_numpy(resize_labels(trimap, size)))
        if (i + 1) % 500 == 0:
            logger.info("cached %d/%d", i + 1, len(source))
    return {"images": torch.stack(images).contiguous(), "trimaps": torch.stack(trimaps).contiguous()}


def cache_path(data_dir: Path, split: str, size: int) -> Path:
    return data_dir / "cache" / f"pets_{split}_{size}.pt"


def save_cache(cache: dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(cache, tmp)
    tmp.replace(path)


def load_cache(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(path, weights_only=True)


def split_indices(n: int, val_fraction: float = 0.1, seed: int = 0) -> tuple[list[int], list[int]]:
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    n_val = round(n * val_fraction)
    return sorted(perm[n_val:]), sorted(perm[:n_val])


def targets_from_trimap(trimap: torch.Tensor, band_weight: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    target = (trimap != BACKGROUND).float()
    weight = torch.where(trimap == BORDER, band_weight, 1.0).float()
    return target, weight


def random_crop_params(size: int, gen: torch.Generator, scale=(0.7, 1.0), ratio=(0.8, 1.25)):
    """RandomResizedCrop parameters drawn from `gen` (torchvision's use the global RNG)."""
    area = size * size
    for _ in range(10):
        target_area = area * float(torch.empty(1).uniform_(*scale, generator=gen))
        log_r = (math.log(ratio[0]), math.log(ratio[1]))
        aspect = math.exp(float(torch.empty(1).uniform_(*log_r, generator=gen)))
        w = round(math.sqrt(target_area * aspect))
        h = round(math.sqrt(target_area / aspect))
        if 0 < w <= size and 0 < h <= size:
            top = int(torch.randint(0, size - h + 1, (1,), generator=gen))
            left = int(torch.randint(0, size - w + 1, (1,), generator=gen))
            return top, left, h, w
    return 0, 0, size, size


def augment_pair(image: torch.Tensor, trimap: torch.Tensor, gen: torch.Generator):
    """Joint hflip + random resized crop; bilinear for the image, nearest for the trimap."""
    size = image.shape[-1]
    if float(torch.rand(1, generator=gen)) < 0.5:
        image, trimap = TF.horizontal_flip(image), TF.horizontal_flip(trimap)
    top, left, h, w = random_crop_params(size, gen)
    image = TF.resized_crop(image, top, left, h, w, [size, size], antialias=True)
    trimap = TF.resized_crop(
        trimap[None], top, left, h, w, [size, size], interpolation=InterpolationMode.NEAREST
    )[0]
    return image, trimap


def paste(
    image_a: torch.Tensor,
    trimap_a: torch.Tensor,
    image_b: torch.Tensor,
    trimap_b: torch.Tensor,
    gen: torch.Generator,
    scale: tuple[float, float] = (0.4, 0.7),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Paste a scaled copy of pet B onto image A.

    Returns the composite image and one trimap per object, (2, S, S): A with the pasted
    region occluded (set to background), and B placed in the frame.
    """
    size = image_a.shape[-1]
    ps = max(8, round(size * float(torch.empty(1).uniform_(*scale, generator=gen))))
    small_img = TF.resize(image_b, [ps, ps], antialias=True)
    small_tri = TF.resize(trimap_b[None], [ps, ps], interpolation=InterpolationMode.NEAREST)[0]
    top = int(torch.randint(0, size - ps + 1, (1,), generator=gen))
    left = int(torch.randint(0, size - ps + 1, (1,), generator=gen))
    pet = small_tri != BACKGROUND
    region = (slice(top, top + ps), slice(left, left + ps))

    image = image_a.clone()
    image[:, region[0], region[1]] = torch.where(pet, small_img, image[:, region[0], region[1]])
    tri_a = trimap_a.clone()
    tri_a[region] = torch.where(pet, torch.full_like(small_tri, BACKGROUND), tri_a[region])
    tri_b = torch.full_like(trimap_a, BACKGROUND)
    tri_b[region] = torch.where(pet, small_tri, tri_b[region])
    return image, torch.stack([tri_a, tri_b])


class PetsDataset(Dataset):
    """Cached pets -> dict(image normalized 3xSxS, target SxS, weight SxS, trimap SxS)."""

    def __init__(
        self,
        cache: dict[str, torch.Tensor],
        indices: list[int],
        augment: bool = False,
        seed: int | None = None,
        band_weight: float = 0.5,
    ):
        self.images = cache["images"]
        self.trimaps = cache["trimaps"]
        self.indices = list(indices)
        self.augment = augment
        self.band_weight = band_weight
        self.gen = torch.Generator()
        self.gen.manual_seed(seed if seed is not None else int(torch.seed() % 2**31))

    def __len__(self) -> int:
        return len(self.indices)

    def raw(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        j = self.indices[i]
        image, trimap = self.images[j], self.trimaps[j]
        if self.augment:
            image, trimap = augment_pair(image, trimap, self.gen)
        return image, trimap

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        image, trimap = self.raw(i)
        target, weight = targets_from_trimap(trimap, self.band_weight)
        return {"image": normalize(image), "target": target, "weight": weight, "trimap": trimap}


class PromptedPetsDataset(PetsDataset):
    """Mini-SAM training items: one image, K prompt sets, each with its own target.

    With probability `paste_prob` a second (augmented) pet is pasted in, and each prompt set
    picks one of the two pets as its target, so the prompt has to decide what to segment.
    """

    def __init__(self, *args, prompts_per_image: int = 3, paste_prob: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = prompts_per_image
        self.paste_prob = paste_prob

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        from boundarybrush.prompts import sample_prompt

        image, trimap = self.raw(i)
        trimaps = trimap[None]
        if len(self.indices) > 1 and float(torch.rand(1, generator=self.gen)) < self.paste_prob:
            j = int(torch.randint(0, len(self.indices), (1,), generator=self.gen))
            other_image, other_trimap = self.raw(j)
            image, trimaps = paste(image, trimap, other_image, other_trimap, self.gen)
        objects = [t for t in trimaps if (t != BACKGROUND).any()] or [trimaps[0]]
        coords, types, targets, weights = [], [], [], []
        for _ in range(self.k):
            tri = objects[int(torch.randint(0, len(objects), (1,), generator=self.gen))]
            target, weight = targets_from_trimap(tri, self.band_weight)
            c, t = sample_prompt(target.bool(), self.gen)
            coords.append(c)
            types.append(t)
            targets.append(target)
            weights.append(weight)
        return {
            "image": normalize(image),
            "coords": torch.stack(coords),
            "types": torch.stack(types),
            "target": torch.stack(targets),
            "weight": torch.stack(weights),
        }
