from __future__ import annotations

import torch

from boundarybrush import data


def test_build_cache_shapes_and_dtypes(synthetic_source):
    cache = data.build_cache(synthetic_source, size=32)
    assert cache["images"].shape == (4, 3, 32, 32) and cache["images"].dtype == torch.uint8
    assert cache["trimaps"].shape == (4, 32, 32) and cache["trimaps"].dtype == torch.uint8
    assert set(cache["trimaps"].unique().tolist()) <= {1, 2, 3}


def test_cache_roundtrip(tmp_path, synthetic_source):
    cache = data.build_cache(synthetic_source, size=16)
    path = tmp_path / "c.pt"
    data.save_cache(cache, path)
    loaded = data.load_cache(path)
    assert torch.equal(loaded["images"], cache["images"])


def test_split_is_deterministic_and_disjoint():
    a_train, a_val = data.split_indices(100, val_fraction=0.1, seed=0)
    b_train, b_val = data.split_indices(100, val_fraction=0.1, seed=0)
    assert a_train == b_train and a_val == b_val
    assert len(a_val) == 10 and not set(a_train) & set(a_val)
    assert sorted(a_train + a_val) == list(range(100))


def test_targets_from_trimap():
    tri = torch.tensor([[1, 2, 3]], dtype=torch.uint8)
    target, weight = data.targets_from_trimap(tri, band_weight=0.5)
    assert target.tolist() == [[1.0, 0.0, 1.0]]
    assert weight.tolist() == [[1.0, 1.0, 0.5]]


def test_dataset_augmentation_keeps_image_and_mask_aligned(synthetic_source):
    cache = data.build_cache(synthetic_source, size=32)
    ds = data.PetsDataset(cache, indices=[0, 1, 2, 3], augment=True, seed=0)
    for i in range(len(ds)):
        item = ds[i]
        assert item["image"].shape == (3, 32, 32) and item["image"].dtype == torch.float32
        assert set(item["target"].unique().tolist()) <= {0.0, 1.0}
        red = item["image"][0] > 0.5  # normalized red is strongly positive only on the pet
        agree = (red == item["target"].bool()).float().mean()
        assert agree > 0.9


def test_dataset_without_augmentation_is_exact(synthetic_source):
    cache = data.build_cache(synthetic_source, size=32)
    ds = data.PetsDataset(cache, indices=[2], augment=False)
    target, _ = data.targets_from_trimap(cache["trimaps"][2])
    assert torch.equal(ds[0]["target"], target)


def test_paste_builds_two_disjoint_objects(synthetic_source):
    cache = data.build_cache(synthetic_source, size=32)
    gen = torch.Generator().manual_seed(3)
    image, trimaps = data.paste(
        cache["images"][0], cache["trimaps"][0], cache["images"][1], cache["trimaps"][1], gen
    )
    assert image.shape == (3, 32, 32) and trimaps.shape == (2, 32, 32)
    a, b = trimaps != 2
    assert b.any()
    assert not (a & b).any()  # the pasted pet occludes the base pet
    red_on_b = image[0][b].float().mean()
    assert red_on_b > 200
