import numpy as np
import pytest

from irvision.preprocessing.masks import qa_flag, qa_invalid_mask


def test_qa_bits_decoded():
    qa = np.array([0b0000_0001, 0b0000_1000, 0b0001_0000, 0b0100_0000, 0b1000_0000], dtype=np.uint16)
    assert qa_flag(qa, "fill").tolist() == [True, False, False, False, False]
    assert qa_flag(qa, "cloud").tolist() == [False, True, False, False, False]
    assert qa_flag(qa, "cloud_shadow").tolist() == [False, False, True, False, False]
    assert qa_flag(qa, "water").tolist() == [False, False, False, False, True]


def test_invalid_mask_combines_flags():
    qa = np.array([1, 8, 16, 64, 128], dtype=np.uint16)  # fill, cloud, shadow, clear, water
    invalid = qa_invalid_mask(qa, ["fill", "cloud", "cloud_shadow"])
    assert invalid.tolist() == [True, True, True, False, False]


def test_unknown_flag():
    with pytest.raises(ValueError):
        qa_flag(np.zeros(1, np.uint16), "haze")
