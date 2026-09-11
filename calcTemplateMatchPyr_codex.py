"""Coarse-to-fine template matching for large-displacement DIC tracking.

The routine in this module deliberately favours readable, inspectable logic
over throughput.  It is useful when a 51 px subset can move much farther than
an optical-flow linearisation is comfortable with between two DIC images.
"""

from __future__ import annotations

import math
from typing import TypeAlias

import cv2
import numpy as np


IntPair: TypeAlias = int | tuple[int, int]
_SQDIFF_METHODS = {cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED}


def calcTemplateMatchPyr(
    prev_img: np.ndarray,
    next_img: np.ndarray,
    prev_pts: np.ndarray,
    template_size: IntPair,
    search_range: IntPair,
    num_levels: int,
    match_method: int = cv2.TM_CCOEFF_NORMED,
    subpixel: bool = True,
    *,
    min_correlation: float = 0.5,
    fine_search_radius: IntPair = 4,
    sidelobe_exclusion_radius: int = 2,
    next_pts_init: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Track points from ``prev_img`` to ``next_img`` by pyramid matching.

    Parameters
    ----------
    prev_img, next_img:
        Equal-sized grayscale or BGR/BGRA images.  They are converted to
        ``float32`` grayscale before matching, so uint8 and float images are
        both accepted.
    prev_pts:
        Source point coordinates of shape ``(N, 2)`` in ``(x, y)`` pixel
        order at full resolution.
    template_size:
        Width/height of the full-resolution source subset.  An ``int`` means
        a square subset.  At each pyramid level it is reduced proportionally
        and rounded to an odd size so its centre has an unambiguous pixel.
    search_range:
        Maximum full-resolution displacement to search at the coarsest level.
        An ``int`` means +/- the same amount in x and y.
    num_levels:
        Number of Gaussian-pyramid images, including the original image.
        ``num_levels=1`` therefore performs ordinary full-resolution template
        matching.
    match_method:
        Any OpenCV ``cv2.matchTemplate`` method.  Normalized methods are
        recommended because ``min_correlation`` then has a stable meaning.
    subpixel:
        When true, fit a two-dimensional paraboloid to the 3x3 neighbourhood
        around the selected response peak.  A response peak on the border has
        no complete 3x3 neighbourhood, so it is intentionally not refined and
        receives a confidence penalty.
    min_correlation:
        Success threshold for the normalized peak quality in [0, 1].  For
        CCOEFF_NORMED and CCORR_NORMED it is the correlation value (negative
        CCOEFF values are treated as zero); for SQDIFF_NORMED it is
        ``1 - sqdiff``.  Non-normalized OpenCV methods use their response
        contrast instead, so their threshold is less physically meaningful.
    fine_search_radius:
        Residual +/- search radius at every level after the coarsest one.
        The displacement found at a coarse level is multiplied by two before
        entering the next finer level, leaving only this small residual search.
    sidelobe_exclusion_radius:
        Radius in response-map pixels masked around the main peak before the
        strongest remaining local maximum is measured for the sharpness score.
    next_pts_init:
        Optional ``(N, 2)`` initial-guess locations in ``next_img``, full
        resolution -- the same role as ``cv2.calcOpticalFlowPyrLK``'s
        ``nextPts`` used with ``OPTFLOW_USE_INITIAL_FLOW``.  When given, a
        point's coarsest-level search is centred on its guess instead of on
        its ``prev_pts`` location (zero motion).  A non-finite guess for a
        point falls back to zero motion for that point only.  ``search_range``
        at the coarsest level must still be large enough to cover how wrong
        the guess might be.

    Returns
    -------
    next_pts:
        ``(N, 2)`` ``float64`` points.  Failed points are ``NaN`` rather than
        an invented displacement.
    status:
        ``uint8`` array where 1 means the peak met ``min_correlation`` and all
        required template/search windows were valid; 0 means failed.
    confidence:
        ``float64`` [0, 1] score combining normalized peak quality and a
        peak-to-sidelobe sharpness measure.  Border-clipped searches and
        response-border peaks are penalized but may still be successful.

    Notes
    -----
    Each point is processed independently.  That avoids cross-point coupling
    and makes border failures explicit; it is also appropriate for the low
    frame rates commonly acceptable for large DIC point sets.
    """
    template_wh = _as_pair(template_size, "template_size", minimum=3)
    search_xy = _as_pair(search_range, "search_range", minimum=0)
    fine_xy = _as_pair(fine_search_radius, "fine_search_radius", minimum=0)
    if num_levels < 1:
        raise ValueError("num_levels must be at least 1")
    if sidelobe_exclusion_radius < 0:
        raise ValueError("sidelobe_exclusion_radius must be non-negative")
    if not 0.0 <= min_correlation <= 1.0:
        raise ValueError("min_correlation must be in the range [0, 1]")

    prev_gray = _as_matchable_gray(prev_img, "prev_img")
    next_gray = _as_matchable_gray(next_img, "next_img")
    if prev_gray.shape != next_gray.shape:
        raise ValueError("prev_img and next_img must have the same image shape")

    points = np.asarray(prev_pts, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("prev_pts must have shape (N, 2) in (x, y) order")

    init_points = None
    if next_pts_init is not None:
        init_points = np.asarray(next_pts_init, dtype=np.float64)
        if init_points.shape != points.shape:
            raise ValueError("next_pts_init must have the same shape as prev_pts")

    prev_pyramid = _build_gaussian_pyramid(prev_gray, num_levels)
    next_pyramid = _build_gaussian_pyramid(next_gray, num_levels)
    next_pts = np.full(points.shape, np.nan, dtype=np.float64)
    status = np.zeros(len(points), dtype=np.uint8)
    confidence = np.zeros(len(points), dtype=np.float64)

    for point_index, full_point in enumerate(points):
        if not np.isfinite(full_point).all():
            continue

        # ``offset`` is expressed in pixels of the current pyramid level.
        # Going from level L to L-1 doubles it because pyrDown halves both
        # coordinate axes.  The next search is centred on this prediction,
        # not on the original point, which is the key to large displacement.
        offset = np.zeros(2, dtype=np.float64)
        if init_points is not None:
            init_point = init_points[point_index]
            if np.isfinite(init_point).all():
                coarsest_scale = 2 ** (num_levels - 1)
                offset = (init_point - full_point) / coarsest_scale
        point_failed = False
        final_quality = 0.0
        final_sharpness = 0.0
        final_border_limited = False

        for level in range(num_levels - 1, -1, -1):
            scale = 2 ** level
            source_center = full_point / scale
            if level == num_levels - 1:
                # A wide coarsest search covers the user-supplied maximum
                # displacement after it has been reduced by the pyramid scale.
                radius = (
                    int(math.ceil(search_xy[0] / scale)),
                    int(math.ceil(search_xy[1] / scale)),
                )
            else:
                offset *= 2.0
                radius = fine_xy

            template_wh_level = _scaled_odd_pair(template_wh, scale)
            match = _match_at_level(
                prev_pyramid[level],
                next_pyramid[level],
                source_center=source_center,
                predicted_center=source_center + offset,
                template_wh=template_wh_level,
                search_radius=radius,
                match_method=match_method,
                subpixel=subpixel,
                sidelobe_exclusion_radius=sidelobe_exclusion_radius,
            )
            if match is None:
                point_failed = True
                break

            matched_center, peak_quality, sharpness, border_limited, anchor_center = match
            offset = matched_center - anchor_center
            final_quality = peak_quality
            final_sharpness = sharpness
            final_border_limited = border_limited

        if point_failed:
            continue

        # Geometric averaging means an ambiguous peak cannot receive high
        # confidence merely because its absolute correlation happens to be high.
        combined = math.sqrt(max(0.0, final_quality) * max(0.0, final_sharpness))
        if final_border_limited:
            combined *= 0.5
        confidence[point_index] = float(np.clip(combined, 0.0, 1.0))
        if final_quality >= min_correlation:
            next_pts[point_index] = full_point + offset
            status[point_index] = 1

    return next_pts, status, confidence


def _as_pair(value: IntPair, name: str, minimum: int) -> tuple[int, int]:
    """Convert a scalar or (x, y) setting into validated integer dimensions."""
    if isinstance(value, (int, np.integer)):
        pair = (int(value), int(value))
    else:
        try:
            pair = (int(value[0]), int(value[1]))
        except (TypeError, IndexError, ValueError) as exc:
            raise ValueError(f"{name} must be an int or a two-item (x, y) pair") from exc
    if pair[0] < minimum or pair[1] < minimum:
        raise ValueError(f"{name} values must be at least {minimum}")
    return pair


def _as_matchable_gray(image: np.ndarray, name: str) -> np.ndarray:
    """Convert a gray/BGR/BGRA image to the float32 format matchTemplate accepts."""
    array = np.asarray(image)
    if array.ndim == 2:
        gray = array
    elif array.ndim == 3 and array.shape[2] == 3:
        gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    elif array.ndim == 3 and array.shape[2] == 4:
        gray = cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"{name} must be a 2D grayscale, BGR, or BGRA image")
    if gray.size == 0:
        raise ValueError(f"{name} must not be empty")
    return np.ascontiguousarray(gray, dtype=np.float32)


def _build_gaussian_pyramid(image: np.ndarray, num_levels: int) -> list[np.ndarray]:
    """Build exactly ``num_levels`` images, failing clearly if one becomes too small."""
    pyramid = [image]
    for _ in range(1, num_levels):
        if min(pyramid[-1].shape[:2]) < 2:
            raise ValueError("num_levels is too large for the input image dimensions")
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return pyramid


def _scaled_odd_pair(size: tuple[int, int], scale: int) -> tuple[int, int]:
    """Shrink a full-resolution template while retaining a central pixel."""
    scaled = []
    for component in size:
        value = max(3, int(round(component / scale)))
        scaled.append(value if value % 2 else value + 1)
    return scaled[0], scaled[1]


def _match_at_level(
    prev_level: np.ndarray,
    next_level: np.ndarray,
    *,
    source_center: np.ndarray,
    predicted_center: np.ndarray,
    template_wh: tuple[int, int],
    search_radius: tuple[int, int],
    match_method: int,
    subpixel: bool,
    sidelobe_exclusion_radius: int,
) -> tuple[np.ndarray, float, float, bool, np.ndarray] | None:
    """Match one source subset at one pyramid level, returning its centre."""
    template_info = _extract_template(prev_level, source_center, template_wh)
    if template_info is None:
        # A source subset cannot be clipped without changing its content and
        # correlation statistics, so source-border points are clean failures.
        return None
    template, half_wh, anchor_center = template_info

    search_info = _extract_search_region(
        next_level, predicted_center, half_wh, search_radius)
    if search_info is None:
        return None
    search_region, origin, search_was_clipped = search_info
    if (search_region.shape[0] < template.shape[0]
            or search_region.shape[1] < template.shape[1]):
        return None

    response = cv2.matchTemplate(search_region, template, match_method)
    if response.size == 0 or not np.isfinite(response).all():
        return None

    is_sqdiff = match_method in _SQDIFF_METHODS
    min_value, max_value, min_location, max_location = cv2.minMaxLoc(response)
    peak_location = min_location if is_sqdiff else max_location
    peak_value = min_value if is_sqdiff else max_value
    oriented_response = -response if is_sqdiff else response
    border_peak = (
        peak_location[0] == 0
        or peak_location[1] == 0
        or peak_location[0] == response.shape[1] - 1
        or peak_location[1] == response.shape[0] - 1
    )

    peak_delta = np.zeros(2, dtype=np.float64)
    if subpixel and not border_peak:
        peak_delta = _subpixel_paraboloid_offset(oriented_response, peak_location)

    # matchTemplate reports a template's top-left corner.  Converting that
    # location to its centre makes the offset directly comparable to the
    # source point and is independent of the cropped search-region origin.
    matched_center = np.array([
        origin[0] + peak_location[0] + half_wh[0] + peak_delta[0],
        origin[1] + peak_location[1] + half_wh[1] + peak_delta[1],
    ], dtype=np.float64)
    peak_quality = _normalised_peak_quality(
        float(peak_value), response, is_sqdiff, match_method)
    sharpness = _peak_sharpness(
        oriented_response, peak_location, sidelobe_exclusion_radius)
    return (matched_center, peak_quality, sharpness,
            (search_was_clipped or border_peak), anchor_center)


def _extract_template(
    image: np.ndarray, center: np.ndarray, size_wh: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int], np.ndarray] | None:
    """Extract a fixed-size centred source subset, rejecting source borders.

    ``center`` is a full-precision (sub-pixel) coordinate, but a template can
    only be cropped at an integer pixel, so the crop is anchored at
    ``round(center)``.  The returned anchor lets the caller measure
    displacement relative to the pixel actually sampled, instead of relative
    to ``center``'s fractional position -- otherwise a point's fractional
    part would silently be replaced by the rounding remainder every time.
    """
    half_x, half_y = size_wh[0] // 2, size_wh[1] // 2
    center_x, center_y = int(round(center[0])), int(round(center[1]))
    x0, x1 = center_x - half_x, center_x + half_x + 1
    y0, y1 = center_y - half_y, center_y + half_y + 1
    if x0 < 0 or y0 < 0 or x1 > image.shape[1] or y1 > image.shape[0]:
        return None
    anchor = np.array([center_x, center_y], dtype=np.float64)
    return image[y0:y1, x0:x1], (half_x, half_y), anchor


def _extract_search_region(
    image: np.ndarray,
    center: np.ndarray,
    half_wh: tuple[int, int],
    radius_xy: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int], bool] | None:
    """Crop the target search area, clipping it and recording target borders."""
    center_x, center_y = int(round(center[0])), int(round(center[1]))
    requested_x0 = center_x - half_wh[0] - radius_xy[0]
    requested_x1 = center_x + half_wh[0] + radius_xy[0] + 1
    requested_y0 = center_y - half_wh[1] - radius_xy[1]
    requested_y1 = center_y + half_wh[1] + radius_xy[1] + 1
    x0, x1 = max(0, requested_x0), min(image.shape[1], requested_x1)
    y0, y1 = max(0, requested_y0), min(image.shape[0], requested_y1)
    if x0 >= x1 or y0 >= y1:
        return None
    clipped = (x0 != requested_x0 or x1 != requested_x1
               or y0 != requested_y0 or y1 != requested_y1)
    return image[y0:y1, x0:x1], (x0, y0), clipped


def _subpixel_paraboloid_offset(response: np.ndarray,
                                peak_location: tuple[int, int]) -> np.ndarray:
    """Fit ``ax² + by² + cxy + dx + ey + f`` to a 3x3 response neighbourhood.

    The response has already been oriented so that the desired match is a
    maximum.  We only accept a finite stationary point inside the local 3x3
    square whose Hessian is negative definite; otherwise the integer peak is
    safer than extrapolating a saddle or a poorly conditioned fit.
    """
    x, y = peak_location
    neighbourhood = response[y - 1:y + 2, x - 1:x + 2].astype(np.float64)
    coordinates = np.array(
        [(dx, dy) for dy in (-1.0, 0.0, 1.0) for dx in (-1.0, 0.0, 1.0)],
        dtype=np.float64,
    )
    design = np.column_stack((
        coordinates[:, 0] ** 2,
        coordinates[:, 1] ** 2,
        coordinates[:, 0] * coordinates[:, 1],
        coordinates[:, 0],
        coordinates[:, 1],
        np.ones(9),
    ))
    try:
        a, b, c, d, e, _ = np.linalg.lstsq(design, neighbourhood.ravel(), rcond=None)[0]
        hessian = np.array(((2.0 * a, c), (c, 2.0 * b)), dtype=np.float64)
        if np.any(np.linalg.eigvalsh(hessian) >= -1e-10):
            return np.zeros(2, dtype=np.float64)
        delta = np.linalg.solve(hessian, -np.array((d, e), dtype=np.float64))
    except np.linalg.LinAlgError:
        return np.zeros(2, dtype=np.float64)
    if not np.isfinite(delta).all() or np.any(np.abs(delta) > 1.0):
        return np.zeros(2, dtype=np.float64)
    return delta


def _normalised_peak_quality(peak_value: float, response: np.ndarray,
                             is_sqdiff: bool, match_method: int) -> float:
    """Map the preferred raw response value to a confidence-compatible score."""
    if match_method == cv2.TM_SQDIFF_NORMED:
        return float(np.clip(1.0 - peak_value, 0.0, 1.0))
    if match_method in (cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED):
        return float(np.clip(peak_value, 0.0, 1.0))

    # Absolute scales of unnormalized methods depend on template intensity and
    # size.  Their portable quantity is the winning response's separation from
    # the rest of this response map, expressed as a saturated z-score.
    oriented = -response if is_sqdiff else response
    spread = float(np.std(oriented))
    if spread <= np.finfo(np.float32).eps:
        return 0.0
    z_score = max(0.0, (float(oriented.max()) - float(np.mean(oriented))) / spread)
    return float(1.0 - math.exp(-z_score / 3.0))


def _peak_sharpness(response: np.ndarray, peak_location: tuple[int, int],
                    exclusion_radius: int) -> float:
    """Return a [0, 1] peak-to-sidelobe sharpness score for a max-oriented map."""
    peak_x, peak_y = peak_location
    yy, xx = np.ogrid[:response.shape[0], :response.shape[1]]
    outside = (xx - peak_x) ** 2 + (yy - peak_y) ** 2 > exclusion_radius ** 2
    if not np.any(outside):
        return 1.0

    # Restrict the comparison to local maxima when present.  This prevents a
    # broad shoulder from being mistaken for a separate competing match.
    local_maxima = response == cv2.dilate(response, np.ones((3, 3), np.float32))
    sidelobe_candidates = response[outside & local_maxima]
    if sidelobe_candidates.size == 0:
        sidelobe_candidates = response[outside]
    sidelobe = float(np.max(sidelobe_candidates))
    peak = float(response[peak_y, peak_x])
    spread = float(np.std(response[outside]))
    if spread <= np.finfo(np.float32).eps:
        return 1.0 if peak > sidelobe else 0.0
    psr = max(0.0, (peak - sidelobe) / spread)
    return float(1.0 - math.exp(-psr))


if __name__ == "__main__":
    # Self-contained smoke test: a textured image is translated by a known
    # amount, then several interior points recover that displacement.
    rng = np.random.default_rng(7)
    previous = cv2.GaussianBlur(
        rng.integers(0, 256, size=(480, 640), dtype=np.uint8), (0, 0), 1.2)
    # Non-integer motion exercises the final 3x3 paraboloid subpixel fit.
    ground_truth = np.array((37.35, -26.60))
    transform = np.array(
        ((1.0, 0.0, ground_truth[0]), (0.0, 1.0, ground_truth[1])), dtype=np.float32)
    following = cv2.warpAffine(
        previous, transform, (previous.shape[1], previous.shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    seed_points = np.array(((180.0, 180.0), (360.0, 220.0), (460.0, 330.0)))

    tracked, ok, quality = calcTemplateMatchPyr(
        previous, following, seed_points, template_size=51, search_range=60,
        num_levels=4, min_correlation=0.6)
    recovered = tracked - seed_points
    print("ground truth:", ground_truth)
    print("recovered:\n", recovered)
    print("status:", ok, "confidence:", np.round(quality, 3))
    assert np.all(ok == 1)
    assert np.allclose(recovered, ground_truth, atol=0.35)
