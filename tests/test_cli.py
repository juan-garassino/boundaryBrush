from __future__ import annotations

import argparse
import json

import numpy as np
import pytest
from PIL import Image

from boundarybrush.cli import build_parser, main, parse_box, parse_click


def test_parse_click_and_box():
    assert parse_click("10,20") == ((10.0, 20.0), 1)
    assert parse_click("10.5,20:0") == ((10.5, 20.0), 0)
    assert parse_box("1,2,3,4") == (1.0, 2.0, 3.0, 4.0)
    for bad in ("10", "a,b", "1,2:5"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_click(bad)


def test_parser_roundtrip():
    args = build_parser().parse_args(
        ["predict", "img.png", "-b", "unet", "--click", "1,2", "--click", "3,4:0", "--box", "0,0,5,5"]
    )
    assert args.backend == "unet" and len(args.click) == 2 and args.box == (0, 0, 5, 5)
    args = build_parser().parse_args(["train", "minisam", "--epochs", "2", "--resume"])
    assert args.model == "minisam" and args.epochs == 2 and args.resume


def test_predict_with_mock_backend_writes_pngs(tmp_path, capsys):
    img = tmp_path / "img.png"
    Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8)).save(img)
    out = tmp_path / "out"
    assert main(["predict", str(img), "-b", "mock", "--click", "30,20", "-o", str(out)]) == 0
    assert (out / "mask.png").exists() and (out / "overlay.png").exists()
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["pixels"] > 0
