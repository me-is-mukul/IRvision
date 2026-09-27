import numpy as np

from irvision.preprocessing.patches import extract_patches, patch_windows, split_windows


def test_patch_windows_only_complete():
    assert patch_windows(512, 512, 256, 256) == [(0, 0), (0, 256), (256, 0), (256, 256)]
    assert len(patch_windows(600, 300, 256, 128)) == 3 * 1
    assert patch_windows(100, 100, 256, 256) == []


def test_split_windows_stay_inside_stripes():
    height, frac = 2048, (0.7, 0.15, 0.15)   # stripes: 0..1433.6, 1433.6..1740.8, 1740.8..2048
    windows = split_windows(height, 512, 256, 128, frac)
    bounds = {"train": (0, 1433.6), "val": (1433.6, 1740.8), "test": (1740.8, 2048)}
    for split, row, _ in windows:
        lo, hi = bounds[split]
        assert lo <= row and row + 256 <= hi, (split, row)
    # a stripe only ~307 px tall must still produce a row of patches
    assert {s for s, _, _ in windows} == {"train", "val", "test"}


def _arrays(h, w):
    rng = np.random.default_rng(0)
    ir = rng.random((h, w), dtype=np.float32)
    return ir, ir.copy(), rng.random((3, h, w), dtype=np.float32), np.ones((h, w), bool)


def test_extract_patches_shapes_and_filter():
    ir, ir_c, rgb, valid = _arrays(1024, 512)
    valid[:256, :256] = False                # first window fully invalid
    patches = list(extract_patches(ir, ir_c, rgb, valid, size=256, stride=256,
                                   max_invalid_fraction=0.05, fractions=(0.5, 0.25, 0.25)))
    assert (0, 0) not in {(p.row, p.col) for p in patches}
    assert {p.split for p in patches} == {"train", "val", "test"}
    p = patches[0]
    assert p.ir.shape == (1, 256, 256) and p.ir_clahe.shape == (1, 256, 256)
    assert p.rgb.shape == (3, 256, 256) and p.valid.shape == (256, 256)
    np.testing.assert_array_equal(p.rgb, rgb[:, p.row:p.row + 256, p.col:p.col + 256])


def test_force_split():
    ir, ir_c, rgb, valid = _arrays(512, 512)
    patches = list(extract_patches(ir, ir_c, rgb, valid, size=256, stride=256, force_split="test"))
    assert len(patches) == 4 and all(p.split == "test" for p in patches)
