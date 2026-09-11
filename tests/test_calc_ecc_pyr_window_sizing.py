import cv2
import numpy as np
import pytest

import calcTemplateMatchPyr_calcEccPyr_claude as tracking


def test_ecc_uses_small_search_windows_after_coarsest_level(monkeypatch):
    requested_search_widths = []

    def fake_ecc_align(*args, **kwargs):
        requested_search_widths.append(args[8][0])
        return args[3], args[4], 1.0, True

    monkeypatch.setattr(tracking, "_ecc_align_one_level", fake_ecc_align)
    image = np.zeros((128, 128), dtype=np.uint8)
    tracking.calcEccPyr(
        image, image, np.array([[64.0, 64.0]]), win_size=31,
        search_range=120, num_levels=3, refine_radius=5.0,
    )

    # Coarsest level scales the global range (120 / 4); the two finer levels
    # and the final ECC polish use the local +/- 5 px refinement span (10 px).
    assert requested_search_widths == [30.0, 10.0, 10.0, 10.0]


def test_template_match_rejects_unnormalized_sqdiff_scores():
    image = np.zeros((64, 64), dtype=np.uint8)

    with pytest.raises(ValueError, match="non-normalized template-match scores"):
        tracking.calcTemplateMatchPyr(
            image, image, np.array([[32.0, 32.0]]), template_size=21,
            search_range=20, num_levels=1, match_method=cv2.TM_SQDIFF,
        )
