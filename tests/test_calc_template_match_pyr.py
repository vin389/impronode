import cv2
import numpy as np

from calcTemplateMatchPyr_codex import calcTemplateMatchPyr


def test_pyramid_template_match_recovers_large_translation():
    rng = np.random.default_rng(42)
    previous = cv2.GaussianBlur(
        rng.integers(0, 256, size=(384, 512), dtype=np.uint8), (0, 0), 1.2)
    displacement = np.array((31.0, -23.0))
    transform = np.array(((1.0, 0.0, 31.0), (0.0, 1.0, -23.0)), dtype=np.float32)
    following = cv2.warpAffine(
        previous, transform, (512, 384), borderMode=cv2.BORDER_REFLECT_101)
    previous_points = np.array(((150.0, 150.0), (330.0, 260.0)))

    next_points, status, confidence = calcTemplateMatchPyr(
        previous, following, previous_points, template_size=51,
        search_range=48, num_levels=4, min_correlation=0.6,
    )

    assert np.all(status == 1)
    assert np.allclose(next_points - previous_points, displacement, atol=0.75)
    assert np.all((0.0 <= confidence) & (confidence <= 1.0))
