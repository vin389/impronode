"""
dic_pyramid_tracking.py
========================
Pyramidal (coarse-to-fine) point tracking utilities for Digital Image Correlation (DIC)
style applications, where per-point displacements (tens to a couple hundred pixels) can be
large relative to the small correlation/template window (tens of pixels) used to track each
point, inside very large images (thousands of pixels per side).

Two independent trackers are provided, built the same way (Gaussian pyramid, coarse-to-fine
refinement, explicit hand-off of the previous level's estimate to the next level):

    * calcEccPyr             - sub-pixel accurate intensity-based alignment via OpenCV's
                                Enhanced Correlation Coefficient algorithm
                                (cv2.findTransformECC). Supports translation / Euclidean /
                                affine / homography per-point local motion models.

    * calcTemplateMatchPyr   - classic normalized cross-correlation template matching
                                (cv2.matchTemplate) with a 2D quadratic ("biquadratic")
                                sub-pixel peak fit and a peak-sharpness confidence metric.

Both are written as clear, per-point Python loops (no vectorization across points), because
the stated performance budget for this application is very low (0.01 - 0.1 FPS is
acceptable) -- clarity and robustness at the per-point level matters far more than raw
throughput here.

THE SINGLE MOST IMPORTANT / BUG-PRONE PIECE OF LOGIC IN BOTH FUNCTIONS
------------------------------------------------------------------------
When you go from a coarse pyramid level to the next finer level, every pixel *distance*
(a displacement, an ECC translation, a template-match offset) has to be DOUBLED, because
the finer image has twice the linear resolution of the coarser one. Anything that is a
*ratio* or an *angle* instead of a distance (a rotation angle, a uniform scale factor, a
shear coefficient -- i.e. everything in the top-left 2x2 "rotation/scale/shear" block of a
Euclidean/affine/homography warp matrix) must NOT be touched, because those quantities are
scale-invariant.

Getting this wrong (e.g. multiplying the *whole* warp matrix by 2, or forgetting to scale
the propagated offset at all) is one of the most common bugs in pyramidal tracking code, and
it fails silently: the tracker still "runs", it just converges to the wrong answer, often
only for the larger displacements where you can least afford to be wrong. Both functions
below scale ONLY the translation component at every level transition -- see
`_scale_warp_translation()` for calcEccPyr, and the `disp *= 2.0` line inside
`calcTemplateMatchPyr` for the equivalent rule applied to a plain displacement vector.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import cv2
import numpy as np


_NORMALIZED_TEMPLATE_MATCH_METHODS = {
    cv2.TM_SQDIFF_NORMED,
    cv2.TM_CCORR_NORMED,
    cv2.TM_CCOEFF_NORMED,
}

# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------

#: Minimum allowed side length (pixels) for a template or search window at any pyramid
#: level. Windows that would clip down smaller than this (e.g. a point too close to the
#: image border, or too many pyramid levels for the requested window size) are treated as
#: a failure for that point/level rather than being handed to OpenCV, which would either
#: raise an exception or produce a meaningless result on a degenerate crop.
MIN_WINDOW_PX = 6

IntOrPair = Union[int, Tuple[int, int]]


def _as_wh(value: IntOrPair) -> Tuple[float, float]:
    """Normalize a size given as a single int (square) or an (w, h) pair to a float pair."""
    if isinstance(value, (tuple, list, np.ndarray)):
        w, h = value
        return float(w), float(h)
    return float(value), float(value)


def _to_gray_f32(img: np.ndarray) -> np.ndarray:
    """Convert an image to single-channel float32, which is what both ECC and
    matchTemplate want for the most accurate, well-conditioned sub-pixel results."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.float32:
        img = img.astype(np.float32)
#    return img
    return img.copy()  # to avoid aliasing risk 

def _build_pyramid(img: np.ndarray, num_levels: int) -> List[np.ndarray]:
    """Build a Gaussian pyramid with `num_levels` images using cv2.pyrDown.

    pyramid[0]               -> full resolution (original) image
    pyramid[num_levels - 1]  -> coarsest level, downsampled by 2**(num_levels - 1)

    cv2.pyrDown applies a 5x5 Gaussian low-pass filter before decimating by 2, which
    matters here: naive strided sub-sampling (img[::2, ::2]) would alias the fine
    speckle/texture pattern typical of DIC images and corrupt the coarse-level gradients
    that ECC/matchTemplate rely on to find the coarse alignment in the first place.
    """
    pyramid = [_to_gray_f32(img)]
    for _ in range(num_levels - 1):
        pyramid.append(cv2.pyrDown(pyramid[-1]))
    return pyramid


def _clip_window(
    cx: float, cy: float, half_w: float, half_h: float, img_w: int, img_h: int
) -> Tuple[int, int, int, int]:
    """Return an integer (x0, y0, x1, y1) window centered at (cx, cy) with the requested
    half-extents, clipped to the image bounds [0, img_w) x [0, img_h). x1/y1 are exclusive,
    i.e. directly usable as `img[y0:y1, x0:x1]`. This is the single place border handling
    happens for both trackers: any window that would extend past the image edge is simply
    shrunk to fit, rather than raising -- callers then check `_window_ok` and mark the
    point as failed/low-confidence for that level instead of crashing."""
    x0 = int(np.floor(cx - half_w))
    y0 = int(np.floor(cy - half_h))
    x1 = int(np.ceil(cx + half_w))
    y1 = int(np.ceil(cy + half_h))
    x0 = max(0, min(x0, img_w - 1))
    y0 = max(0, min(y0, img_h - 1))
    x1 = max(0, min(x1, img_w))
    y1 = max(0, min(y1, img_h))
    return x0, y0, x1, y1


def _window_ok(x0: int, y0: int, x1: int, y1: int) -> bool:
    return (x1 - x0) >= MIN_WINDOW_PX and (y1 - y0) >= MIN_WINDOW_PX


# --------------------------------------------------------------------------------------
# calcEccPyr
# --------------------------------------------------------------------------------------

def _identity_warp(motion_type: int) -> np.ndarray:
    """Identity warp matrix in the shape OpenCV's ECC expects for a given motion model."""
    if motion_type == cv2.MOTION_HOMOGRAPHY:
        return np.eye(3, dtype=np.float32)
    return np.eye(2, 3, dtype=np.float32)


def _scale_warp_translation(warp: np.ndarray, factor: float) -> np.ndarray:
    """Scale ONLY the translation part of a warp matrix by `factor`.

    This is the pyramid level hand-off rule described at the top of this file:
      * Rows 0:2, column 2 (the translation column, first two rows) hold the (tx, ty)
        pixel displacement -> these get multiplied by `factor` (2.0 going coarse->fine).
      * Everything else -- the 2x2 rotation/scale/shear block for
        EUCLIDEAN/AFFINE/HOMOGRAPHY, and (for HOMOGRAPHY) the bottom perspective row -- is
        left untouched, because rotation angles, scale ratios and shear are dimensionless
        and identical at every pyramid level.

    Note on homography: the mathematically exact relationship between a homography H_c
    estimated at a coarse level and the equivalent H_f at a level with pixel coordinates
    `factor` times larger is the conjugation H_f = S @ H_c @ inv(S), S = diag(factor,
    factor, 1). Expanding that shows the translation-column-scaling rule used here is
    exact for the top-left 2x2 block and the translation column, and only approximate for
    the bottom perspective row (h20, h21), which strictly should be divided by `factor`.
    For the small-perspective-change-per-pyramid-octave regime typical of DIC tracking
    this approximation is negligible; it's called out explicitly here so it isn't a silent
    surprise if this code is reused for scenes with strong perspective distortion.
    """
    warp = warp.copy()
    warp[:2, 2] *= factor
    return warp


def _ecc_align_one_level(
    template_img: np.ndarray,
    search_img: np.ndarray,
    p_prev: np.ndarray,
    p_next_pred: np.ndarray,
    warp_submatrix: np.ndarray,
    motion_type: int,
    criteria: Tuple[int, int, float],
    win_wh: Tuple[float, float],
    search_wh: Tuple[float, float],
    retry_criteria: Optional[Tuple[int, int, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, float, bool]:
    """Run one ECC alignment for a single point at a single pyramid level.

    A small TEMPLATE window is cropped from `template_img` (prev image at this level)
    centered on the point's current-level position `p_prev`. A larger SEARCH window is
    cropped from `search_img` (next image at this level) centered on the *predicted*
    position `p_next_pred`.

    ECC (Gauss-Newton on the enhanced correlation coefficient) is a *local* optimizer: it
    only reliably converges when its initial warp is already close to the true answer --
    in practice within roughly a template-radius or less. Seeding it with the identity
    warp (i.e. assuming the template's window origin lines up with the search window's
    origin) is only "close" when the true residual displacement inside the search window
    happens to be small; for the large, previously-unknown displacements this pyramid is
    built to handle, that residual can easily be larger than the template itself,
    especially at the coarsest level where there is no prior estimate yet, and ECC will
    converge to the wrong local optimum (or fail outright) if started there.

    To fix this, every level first runs a cheap, brute-force cv2.matchTemplate pass of the
    template over the whole search window to get an integer-pixel best-translation seed
    (this is exactly the same operation `calcTemplateMatchPyr` performs, just reused here
    as a "coarse search" stage). ECC is then initialized with that seed (translation
    combined with the carried-over rotation/scale/shear submatrix) and used only for what
    it's good at: refining an already-close guess to sub-pixel accuracy and, for
    EUCLIDEAN/AFFINE/HOMOGRAPHY, recovering the non-translational part of the motion. This
    coarse-search + local-refine combination is what actually gives this tracker its
    robustness against the 50-200 px range of displacements relative to a ~50 px template.

    Returns
    -------
    p_next : the refined point location, in `search_img` (global, this-level) pixel coords.
    warp_out : the raw warp matrix returned by findTransformECC (maps template-local to
        search-local coordinates; NOT directly a global displacement -- see calcEccPyr).
    ecc_value : the correlation coefficient returned by findTransformECC (0 on failure).
    ok : whether ECC converged at this level (after an optional relaxed-criteria retry).
    """
    h_t, w_t = template_img.shape[:2]
    h_s, w_s = search_img.shape[:2]

    tw, th = win_wh
    sw, sh = search_wh
    # `search_wh` is the desired reachable displacement RANGE for the tracked POINT, not
    # the crop size. For a candidate match at the far edge of that range to actually fit
    # inside the search crop, the crop itself must be `search_wh` wider/taller than the
    # template on top of that (half the template on each side) -- otherwise the true
    # match position can require the template to hang off the edge of the search crop
    # and never be reachable by matchTemplate/ECC at all, even though it is well within
    # the nominal search_range. This is a common, easy-to-miss off-by-template-size bug.
    sw = max(sw + tw, tw + 2 * MIN_WINDOW_PX)
    sh = max(sh + th, th + 2 * MIN_WINDOW_PX)

    tx0, ty0, tx1, ty1 = _clip_window(p_prev[0], p_prev[1], tw / 2.0, th / 2.0, w_t, h_t)
    if not _window_ok(tx0, ty0, tx1, ty1):
        return p_next_pred, warp_submatrix, 0.0, False

    sx0, sy0, sx1, sy1 = _clip_window(p_next_pred[0], p_next_pred[1], sw / 2.0, sh / 2.0, w_s, h_s)
    if not _window_ok(sx0, sy0, sx1, sy1):
        return p_next_pred, warp_submatrix, 0.0, False

    template_patch = template_img[ty0:ty1, tx0:tx1]
    search_patch = search_img[sy0:sy1, sx0:sx1]

    # The search patch must be at least as large as the template patch in both dimensions
    # for findTransformECC's internal sampling to make sense.
    if search_patch.shape[0] < template_patch.shape[0] or search_patch.shape[1] < template_patch.shape[1]:
        return p_next_pred, warp_submatrix, 0.0, False

    # Coarse search stage: cv2.matchTemplate returns, in `search_patch`-local coordinates,
    # the top-left position where `template_patch` best matches -- i.e. exactly the pure
    # translation (in these same local coordinates) that ECC should start from. Using
    # local coordinates on both sides sidesteps having to reason about the window-origin
    # offset explicitly; matchTemplate already searches the *entire* search_patch, so this
    # is robust to displacements as large as (search window size - template size).
    coarse_result = cv2.matchTemplate(search_patch, template_patch, cv2.TM_CCOEFF_NORMED)
    _, _, _, coarse_max_loc = cv2.minMaxLoc(coarse_result)
    seed_tx, seed_ty = float(coarse_max_loc[0]), float(coarse_max_loc[1])

    init_warp = warp_submatrix.copy()
    init_warp[0, 2] = seed_tx
    init_warp[1, 2] = seed_ty

    def _run(crit):
        return cv2.findTransformECC(template_patch, search_patch, init_warp.copy(), motion_type, crit)

    ok = True
    try:
        ecc_value, warp_out = _run(criteria)
    except cv2.error:
        # findTransformECC raises cv2.error when it fails to converge (e.g. a
        # near-singular Hessian from a low-texture patch). Retry once with relaxed
        # criteria to still get a usable estimate to hand off to the next pyramid level,
        # rather than propagating pure zero motion from a hard failure.
        if retry_criteria is not None:
            try:
                ecc_value, warp_out = _run(retry_criteria)
            except cv2.error:
                warp_out, ecc_value, ok = init_warp, 0.0, False
        else:
            warp_out, ecc_value, ok = init_warp, 0.0, False

    # Map the tracked point from template-local coordinates, through the recovered warp,
    # into search-local coordinates, then back into this level's global pixel coordinates.
    pt_local = np.array([p_prev[0] - tx0, p_prev[1] - ty0, 1.0], dtype=np.float64)
    w = warp_out.astype(np.float64)
    if motion_type == cv2.MOTION_HOMOGRAPHY:
        proj = w @ pt_local
        proj = proj[:2] / proj[2]
    else:
        proj = w[:2, :3] @ pt_local
    p_next = np.array([proj[0] + sx0, proj[1] + sy0], dtype=np.float64)

    return p_next, warp_out.astype(np.float32), float(ecc_value), ok


def calcEccPyr(
    prev_img: np.ndarray,
    next_img: np.ndarray,
    prev_pts: np.ndarray,
    win_size: IntOrPair,
    search_range: IntOrPair,
    num_levels: int,
    motion_type: int = cv2.MOTION_TRANSLATION,
    criteria: Tuple[int, int, float] = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4,
    ),
    refine_radius: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Coarse-to-fine per-point ECC (Enhanced Correlation Coefficient) point tracking.

    Designed for DIC-style tracking where the per-point displacement between `prev_img`
    and `next_img` (tens to a couple hundred pixels) can be large relative to a small
    (tens of pixels) template window, inside very large (thousand+ pixel) images. Each
    point is tracked completely independently through its own Gaussian pyramid, coarsest
    level first, refining the estimate at each successively finer level.

    Parameters
    ----------
    prev_img, next_img : 2D (grayscale) or 3D (BGR) arrays, same size.
    prev_pts : (N, 2) array of (x, y) point coordinates in `prev_img`, full resolution.
    win_size : template window size (side length, or (w, h)) at full resolution.
    search_range : search window size (side length, or (w, h)) at full resolution, used
        only at the coarsest pyramid level to find the initial large displacement.
    num_levels : pyramid depth. Level 0 is full resolution; level `num_levels - 1` is
        downsampled by 2**(num_levels - 1). Choose this (together with `search_range` and
        `win_size`) so that, at the coarsest level, the search window is still large
        enough relative to the template to bracket the true displacement: e.g. a 150 px
        shift with a 4-level pyramid (coarsest downsample = 8x) is ~19 px at that level.
    motion_type : one of cv2.MOTION_TRANSLATION / MOTION_EUCLIDEAN / MOTION_AFFINE /
        MOTION_HOMOGRAPHY.
    criteria : OpenCV termination criteria passed to findTransformECC at every pyramid
        level; the final polish pass (see below) always uses a tighter version of this
        automatically.
    refine_radius : half-size in current-level pixels of the small residual search window
        used at every level after the coarsest, and for the final full-resolution polish.
        The prior level's displacement is doubled before each of these searches, so this
        window need only cover the remaining alignment error rather than the full motion.

    Returns
    -------
    next_pts : (N, 2) float64 array of tracked point locations in `next_img`.
    status : (N,) uint8 array. 1 = the point converged successfully at the finest level
        (including the mandatory final polish pass), 0 = it did not (next_pts still holds
        the best available estimate, e.g. propagated with no further motion applied).
    confidence : (N,) float64 array, the ECC correlation coefficient (in [-1, 1], higher
        is better) from the point's final alignment pass.
    warps : (N, 2, 3) or (N, 3, 3) float32 array (shape depends on motion_type) of each
        point's final warp matrix, with the translation entries (`[:2, 2]`) overwritten to
        hold the point's *actual global pixel displacement* (next_pt - prev_pt) rather
        than the window-local offset ECC uses internally, so it is directly usable as-is;
        the rotation/scale/shear part is exactly what ECC estimated and can be used
        downstream to estimate local strain/rotation per point (meaningful for
        MOTION_AFFINE/MOTION_HOMOGRAPHY; for MOTION_TRANSLATION it is always identity, for
        MOTION_EUCLIDEAN it is a pure rotation).
    """
    if num_levels < 1:
        raise ValueError("num_levels must be >= 1")
    if refine_radius < 0:
        raise ValueError("refine_radius must be non-negative")

    prev_pyr = _build_pyramid(prev_img, num_levels)
    next_pyr = _build_pyramid(next_img, num_levels)

    win_wh_full = _as_wh(win_size)
    search_wh_full = _as_wh(search_range)

    prev_pts = np.asarray(prev_pts, dtype=np.float64).reshape(-1, 2)
    n_pts = prev_pts.shape[0]

    next_pts = np.zeros((n_pts, 2), dtype=np.float64)
    status = np.zeros((n_pts,), dtype=np.uint8)
    confidence = np.zeros((n_pts,), dtype=np.float64)
    warps = np.tile(_identity_warp(motion_type), (n_pts, 1, 1))

    # A relaxed fallback criteria used for a single retry when ECC fails to converge at a
    # level: fewer required iterations, a much looser epsilon -- enough to often still get
    # a usable estimate to hand off to the next (finer) level, without being so loose that
    # it just fits noise.
    relaxed_criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        max(10, criteria[1] // 2),
        criteria[2] * 10.0,
    )
    # Tight criteria for the mandatory final polish pass at full resolution.
    tight_criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        max(criteria[1], 100),
        min(criteria[2], 1e-6),
    )

    for i in range(n_pts):
        p0 = prev_pts[i]
        # warp[:2, 2] does double duty as "current best displacement estimate" between
        # levels; its rotation/scale/shear part carries ECC's local-shape estimate.
        warp = _identity_warp(motion_type)
        ecc_val = 0.0
        ok_final = False
        p_next = p0.copy()

        for level in range(num_levels - 1, -1, -1):
            scale = float(2 ** level)

            if level != num_levels - 1:
                # Moving from a coarser level to this finer one: the pixel displacement
                # found so far doubles; rotation/scale/shear (already scale-invariant) is
                # left untouched. See _scale_warp_translation docstring.
                warp = _scale_warp_translation(warp, 2.0)

            p_prev_level = p0 / scale
            p_next_pred = p_prev_level + warp[:2, 2]

            tw = max(MIN_WINDOW_PX, win_wh_full[0] / scale)
            th = max(MIN_WINDOW_PX, win_wh_full[1] / scale)
            if level == num_levels - 1:
                # Only the coarsest level has no displacement prediction, so it
                # needs the full, scale-adjusted search range.
                sw = max(MIN_WINDOW_PX, search_wh_full[0] / scale)
                sh = max(MIN_WINDOW_PX, search_wh_full[1] / scale)
            else:
                # The 2x-propagated coarse estimate is already close at finer
                # levels.  Match calcTemplateMatchPyr by searching only a small
                # residual window here, rather than re-opening the wide search.
                sw = 2.0 * refine_radius
                sh = 2.0 * refine_radius

            p_next_level, warp_out, ecc_val, ok = _ecc_align_one_level(
                prev_pyr[level], next_pyr[level], p_prev_level, p_next_pred,
                warp, motion_type, criteria, (tw, th), (sw, sh),
                retry_criteria=relaxed_criteria,
            )

            # Re-derive the warp carried forward: keep ECC's rotation/scale/shear
            # estimate, but store the *actual* displacement (not the window-local ECC
            # offset) in the translation slot, so the next level's "multiply by 2" rule
            # operates on a physically meaningful quantity.
            warp = warp_out.copy()
            warp[:2, 2] = p_next_level - p_prev_level

            p_next = p_next_level
            ok_final = ok  # only the last (finest, level 0) iteration's value is kept

        # Mandatory final polish pass at full resolution with tight criteria for maximum
        # precision, seeded from the coarse-to-fine result above.
        p_prev_full = p0
        tw = max(MIN_WINDOW_PX, win_wh_full[0])
        th = max(MIN_WINDOW_PX, win_wh_full[1])
        # This is another local refinement around the level-0 estimate, not a
        # second global search.  Keeping it small is both faster and less prone
        # to jumping to a distant look-alike subset.
        sw = 2.0 * refine_radius
        sh = 2.0 * refine_radius
        p_next_polish, warp_out, ecc_val_polish, ok_polish = _ecc_align_one_level(
            prev_pyr[0], next_pyr[0], p_prev_full, p_next,
            warp, motion_type, tight_criteria, (tw, th), (sw, sh),
            retry_criteria=None,
        )
        if ok_polish:
            p_next = p_next_polish
            warp = warp_out.copy()
            warp[:2, 2] = p_next - p_prev_full
            ecc_val = ecc_val_polish
            ok_final = True
        # If the polish pass fails, we simply keep the coarse-to-fine (level 0) result.

        next_pts[i] = p_next
        status[i] = 1 if ok_final else 0
        confidence[i] = ecc_val
        warps[i] = warp

    return next_pts, status, confidence, warps


# --------------------------------------------------------------------------------------
# calcTemplateMatchPyr
# --------------------------------------------------------------------------------------

def _extremum_loc(result: np.ndarray, prefer_min: bool) -> Tuple[Tuple[int, int], float]:
    """Return ((row, col), value) of the extremum of `result` that a given match method
    is looking for: minimum for the SQDIFF family, maximum for everything else."""
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if prefer_min:
        return (min_loc[1], min_loc[0]), min_val
    return (max_loc[1], max_loc[0]), max_val


def _biquadratic_subpixel_offset(neighborhood: np.ndarray) -> Optional[Tuple[float, float]]:
    """Fit a 2D quadratic surface z = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f to a 3x3
    neighborhood of correlation-surface values centered on the integer peak (x, y in
    {-1, 0, 1}), and return the (dx, dy) location of the surface's stationary point -- the
    sub-pixel peak refinement.

    Returns None if the fit is degenerate (near-singular Hessian, i.e. the neighborhood is
    flat along some direction) or if the resulting offset falls outside the fitted +/-1 px
    cell, in which case the quadratic model is not trustworthy and the caller should fall
    back to the integer-pixel location.
    """
    assert neighborhood.shape == (3, 3)
    xs = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=np.float64)
    ys = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=np.float64)
    zs = neighborhood.astype(np.float64).flatten()

    A = np.column_stack([xs ** 2, ys ** 2, xs * ys, xs, ys, np.ones_like(xs)])
    coeffs, *_ = np.linalg.lstsq(A, zs, rcond=None)
    a, b, c, d, e, _f = coeffs

    # Stationary point of the quadratic form: grad = [2a*x + c*y + d, c*x + 2b*y + e] = 0
    M = np.array([[2 * a, c], [c, 2 * b]], dtype=np.float64)
    rhs = np.array([-d, -e], dtype=np.float64)
    det = np.linalg.det(M)
    if abs(det) < 1e-9:
        return None
    dx, dy = np.linalg.solve(M, rhs)
    if not (np.isfinite(dx) and np.isfinite(dy)) or abs(dx) > 1.0 or abs(dy) > 1.0:
        return None
    return float(dx), float(dy)


def _peak_sharpness(
    result: np.ndarray, peak_rc: Tuple[int, int], is_min: bool, exclusion_radius: int
) -> float:
    """Peak-to-sidelobe sharpness in [0, 1]: how much higher (in the "bigger is better"
    sense, after flipping SQDIFF-style surfaces so a bigger value is always better) the
    main peak is than the next-highest local maximum found outside a small circular
    exclusion zone around the main peak. A value near 1 means an isolated, well-defined
    peak (a trustworthy match); a value near 0 means a broad/ambiguous peak, or a
    near-as-good competing peak (e.g. from repetitive/periodic texture or a low-contrast
    region), and should be treated as an untrustworthy match."""
    surf = -result if is_min else result
    r0, c0 = peak_rc
    rr, cc = np.ogrid[: surf.shape[0], : surf.shape[1]]
    exclusion_mask = (rr - r0) ** 2 + (cc - c0) ** 2 <= exclusion_radius ** 2
    peak_val = float(surf[r0, c0])
    masked = surf.copy()
    masked[exclusion_mask] = -np.inf
    if not np.any(np.isfinite(masked)):
        # Search region too small to contain any sidelobe outside the exclusion zone --
        # sharpness can't be assessed. Assume it's fine rather than penalizing a point
        # just because its window happened to be small (e.g. near an image border).
        return 1.0
    second_val = float(np.max(masked))
    denom = abs(peak_val) + 1e-6
    return float(np.clip((peak_val - second_val) / denom, 0.0, 1.0))


def calcTemplateMatchPyr(
    prev_img: np.ndarray,
    next_img: np.ndarray,
    prev_pts: np.ndarray,
    template_size: IntOrPair,
    search_range: IntOrPair,
    num_levels: int,
    match_method: int = cv2.TM_CCOEFF_NORMED,
    subpixel: bool = True,
    min_correlation: float = 0.5,
    refine_radius: float = 4.0,
    sidelobe_exclusion_radius: int = 5,
    next_pts_init: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coarse-to-fine per-point normalized cross-correlation template matching.

    Same large-displacement / small-template / huge-image setting as calcEccPyr, but using
    cv2.matchTemplate instead of ECC. Often faster and more robust than ECC when the
    deformation between prev and next is close to pure translation (which, for a small
    enough template window, it usually is even under moderate strain/rotation).

    Parameters
    ----------
    prev_img, next_img : 2D (grayscale) or 3D (BGR) arrays, same size.
    prev_pts : (N, 2) array of (x, y) points in `prev_img`, full resolution.
    template_size : template patch size (side length, or (w, h)) at full resolution.
    search_range : search region size (side length, or (w, h)) at full resolution, used
        ONLY at the coarsest pyramid level -- this should comfortably bound the maximum
        expected displacement divided by 2**(num_levels - 1).
    num_levels : pyramid depth (see calcEccPyr for the same parameter).
    match_method : one of ``cv2.TM_SQDIFF_NORMED``, ``cv2.TM_CCORR_NORMED``, or
        ``cv2.TM_CCOEFF_NORMED``. Only normalized methods are accepted because
        ``min_correlation`` and the returned confidence must have a stable [0, 1]
        interpretation. SQDIFF_NORMED looks for a minimum; the other two look for a
        maximum.
    subpixel : if True, refine every level's integer-pixel peak with a 2D quadratic
        ("biquadratic") fit on the 3x3 neighborhood around it.
    min_correlation : minimum acceptable final-level match quality (the normalized "peak
        score" in [0, 1], see `confidence` below) for a point to be marked successful in
        `status`.
    refine_radius : half-size (pixels, at each level's own resolution) of the small search
        window used at every level EXCEPT the coarsest -- i.e. how far the true offset is
        allowed to move away from the coarser level's (2x-propagated) prediction at each
        refinement step. Larger is more robust to a poor coarse estimate but slower and
        more prone to locking onto a wrong, nearby peak.
    sidelobe_exclusion_radius : radius (pixels, in the correlation-surface/result array)
        excluded around the main peak when searching for the next-highest competing peak
        used in the sharpness/confidence metric.
    next_pts_init : optional (N, 2) array of (x, y) initial-guess locations in `next_img`,
        full resolution -- the same role as cv2.calcOpticalFlowPyrLK's `nextPts` used with
        OPTFLOW_USE_INITIAL_FLOW. When given, each point's coarsest-level search is
        centered on `next_pts_init[i]` instead of on `prev_pts[i]` (i.e. instead of
        assuming zero motion), which helps when a better estimate is already available
        (e.g. a constant-velocity prediction, or the previous frame's result). Defaults to
        None, which reproduces the original zero-initial-displacement behavior exactly.
        Note this only seeds the coarsest level's search center; `search_range` at that
        level must still be large enough to cover how wrong the guess might be.

    Returns
    -------
    next_pts : (N, 2) float64 array of tracked point locations in `next_img`.
    status : (N,) uint8 array. 1 = final match quality >= min_correlation, else 0.
    confidence : (N,) float64 array in [0, 1], combining the normalized peak match value
        and the peak-to-sidelobe sharpness (see `_peak_sharpness`).
    """
    if num_levels < 1:
        raise ValueError("num_levels must be >= 1")
    if match_method not in _NORMALIZED_TEMPLATE_MATCH_METHODS:
        raise ValueError(
            "match_method must be TM_SQDIFF_NORMED, TM_CCORR_NORMED, or "
            "TM_CCOEFF_NORMED; non-normalized template-match scores cannot be "
            "used with min_correlation."
        )

    prev_pyr = _build_pyramid(prev_img, num_levels)
    next_pyr = _build_pyramid(next_img, num_levels)

    tmpl_wh_full = _as_wh(template_size)
    search_wh_full = _as_wh(search_range)

    prev_pts = np.asarray(prev_pts, dtype=np.float64).reshape(-1, 2)
    n_pts = prev_pts.shape[0]

    if next_pts_init is not None:
        next_pts_init = np.asarray(next_pts_init, dtype=np.float64).reshape(-1, 2)
        if next_pts_init.shape[0] != n_pts:
            raise ValueError("next_pts_init must have the same number of points as prev_pts")

    is_min = match_method == cv2.TM_SQDIFF_NORMED

    next_pts = np.zeros((n_pts, 2), dtype=np.float64)
    status = np.zeros((n_pts,), dtype=np.uint8)
    confidence = np.zeros((n_pts,), dtype=np.float64)

    for i in range(n_pts):
        p0 = prev_pts[i]
        # Running displacement estimate, current-level px. Seeded from next_pts_init (if
        # given) scaled down to the coarsest level, else zero -- same role as
        # calcOpticalFlowPyrLK's nextPts initial-flow seed.
        if next_pts_init is not None:
            coarsest_scale = float(2 ** (num_levels - 1))
            disp = (next_pts_init[i] - p0) / coarsest_scale
        else:
            disp = np.zeros(2, dtype=np.float64)
        peak_score = 0.0
        sharpness = 0.0
        p_next = p0.copy()
        point_valid = True

        for level in range(num_levels - 1, -1, -1):
            scale = float(2 ** level)
            img_prev_lvl = prev_pyr[level]
            img_next_lvl = next_pyr[level]
            h_p, w_p = img_prev_lvl.shape[:2]
            h_n, w_n = img_next_lvl.shape[:2]

            if level != num_levels - 1:
                # Same rule as calcEccPyr: a pixel offset doubles when moving to a level
                # with twice the linear resolution.
                disp *= 2.0

            p_prev_level = p0 / scale
            p_next_pred = p_prev_level + disp

            tw = max(MIN_WINDOW_PX, tmpl_wh_full[0] / scale)
            th = max(MIN_WINDOW_PX, tmpl_wh_full[1] / scale)
            tx0, ty0, tx1, ty1 = _clip_window(p_prev_level[0], p_prev_level[1], tw / 2.0, th / 2.0, w_p, h_p)
            if not _window_ok(tx0, ty0, tx1, ty1):
                point_valid = False
                break
            template_patch = img_prev_lvl[ty0:ty1, tx0:tx1]

            if level == num_levels - 1:
                # Coarsest level: search the full (scaled) search_range, since there is no
                # prior estimate yet to narrow the search around.
                search_span_w = max(MIN_WINDOW_PX, search_wh_full[0] / scale)
                search_span_h = max(MIN_WINDOW_PX, search_wh_full[1] / scale)
            else:
                # Finer levels: a small, fixed-radius refinement window around the
                # 2x-propagated prediction from the coarser level, per `refine_radius`.
                search_span_w = 2.0 * refine_radius
                search_span_h = 2.0 * refine_radius
            # `search_span_*` is the reachable displacement RANGE for the point itself.
            # The actual crop must be that much larger than the template on top of it, or
            # a true match near the edge of the nominal search range would require the
            # template to hang off the edge of the crop and never be found -- the same
            # off-by-template-size issue documented in `_ecc_align_one_level`.
            sw = max(search_span_w + tw, tw + 2 * MIN_WINDOW_PX)
            sh = max(search_span_h + th, th + 2 * MIN_WINDOW_PX)

            sx0, sy0, sx1, sy1 = _clip_window(p_next_pred[0], p_next_pred[1], sw / 2.0, sh / 2.0, w_n, h_n)
            if not _window_ok(sx0, sy0, sx1, sy1):
                point_valid = False
                break
            search_patch = img_next_lvl[sy0:sy1, sx0:sx1]
            if search_patch.shape[0] < template_patch.shape[0] or search_patch.shape[1] < template_patch.shape[1]:
                point_valid = False
                break

            result = cv2.matchTemplate(search_patch, template_patch, match_method)
            (pr, pc), peak_val = _extremum_loc(result, is_min)

            sharpness = _peak_sharpness(result, (pr, pc), is_min, sidelobe_exclusion_radius)

            sub_dx, sub_dy = 0.0, 0.0
            on_border = pr == 0 or pc == 0 or pr == result.shape[0] - 1 or pc == result.shape[1] - 1
            if subpixel and not on_border:
                refined = _biquadratic_subpixel_offset(result[pr - 1: pr + 2, pc - 1: pc + 2])
                if refined is not None:
                    sub_dx, sub_dy = refined
            # else: peak sits on the border of the search region -- a 3x3 neighborhood
            # isn't available, so sub-pixel refinement is skipped for this level and the
            # (unrefined-peak) sharpness computed above stands as the confidence signal.

            # matchTemplate's result[row, col] is the match score with the template's
            # TOP-LEFT corner placed at (col, row) in the search patch. The matched
            # CENTER of the template (i.e. the tracked point's new location) is therefore
            # the peak location plus half the template size, in search-patch local
            # coordinates, converted to this level's global coordinates by adding the
            # search patch's own top-left corner.
            match_center_local = np.array([
                pc + sub_dx + template_patch.shape[1] / 2.0,
                pr + sub_dy + template_patch.shape[0] / 2.0,
            ])
            p_next_level = np.array([sx0, sy0], dtype=np.float64) + match_center_local

            # The public API accepts normalized methods only, so this quality score has a
            # stable [0, 1] meaning.  SQDIFF_NORMED is inverted because lower is better.
            if match_method == cv2.TM_SQDIFF_NORMED:
                peak_score = float(np.clip(1.0 - peak_val, 0.0, 1.0))
            else:
                peak_score = float(np.clip(peak_val, 0.0, 1.0))

            disp = p_next_level - p_prev_level
            p_next = p_next_level

        if not point_valid:
            next_pts[i] = p_next
            status[i] = 0
            confidence[i] = 0.0
            continue

        next_pts[i] = p_next
        confidence[i] = float(np.clip(0.5 * peak_score + 0.5 * sharpness, 0.0, 1.0))
        status[i] = 1 if peak_score >= min_correlation else 0

    return next_pts, status, confidence


# --------------------------------------------------------------------------------------
# Self-contained example / smoke test
# --------------------------------------------------------------------------------------

def _make_speckle_image(size: Tuple[int, int], seed: int = 0) -> np.ndarray:
    """Synthesize a random-textured ("speckle-like") 8-bit grayscale image typical of a
    DIC target: correlation-based tracking needs local texture/contrast to lock onto, so
    a blank or smoothly-varying image would not exercise either tracker meaningfully."""
    rng = np.random.default_rng(seed)
    h, w = size
    noise = rng.random((h, w)).astype(np.float32)
    speckle = cv2.GaussianBlur(noise, (0, 0), sigmaX=1.5)
    speckle = cv2.normalize(speckle, None, 0, 255, cv2.NORM_MINMAX)
    return speckle.astype(np.uint8)


def _shift_image(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Shift `img` by a known, possibly sub-pixel, (dx, dy) translation using warpAffine.
    With the default (non-inverse) mapping convention, a feature located at (x, y) in
    `img` ends up at (x + dx, y + dy) in the output -- i.e. the output IS `img` displaced
    by exactly (dx, dy), which is the ground truth we then ask the trackers to recover."""
    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(
        img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


if __name__ == "__main__":
    print("Building synthetic speckle image pair with a known translation...")
    img_size = (600, 600)  # (h, w) -- kept small for a fast demo; real use case is ~5000x5000
    prev = _make_speckle_image(img_size, seed=42)
    true_dx, true_dy = 68.3, -41.7  # sub-pixel, well within the stated 50-200 px large-displacement regime
    nxt = _shift_image(prev, true_dx, true_dy)
    # A touch of sensor-like noise so the test isn't unrealistically noise-free.
    rng = np.random.default_rng(1)
    nxt = np.clip(nxt.astype(np.float32) + rng.normal(0, 2.0, nxt.shape), 0, 255).astype(np.uint8)

    # A handful of interior points, kept far enough from the border that the largest
    # (coarsest-level) search window never needs to clip -- see the separate near-border
    # demonstration further below for the graceful-failure path.
    margin = 140
    pts = np.array(
        [
            [margin, margin],
            [img_size[1] - margin, margin],
            [margin, img_size[0] - margin],
            [img_size[1] - margin, img_size[0] - margin],
            [img_size[1] // 2, img_size[0] // 2],
        ],
        dtype=np.float64,
    )

    win_size = 51
    search_range = 220
    num_levels = 3
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-4)

    print("\n--- calcEccPyr (MOTION_TRANSLATION) ---")
    next_pts_ecc, status_ecc, conf_ecc, warps_ecc = calcEccPyr(
        prev, nxt, pts, win_size, search_range, num_levels,
        motion_type=cv2.MOTION_TRANSLATION, criteria=criteria,
    )
    disp_ecc = next_pts_ecc - pts
    for i in range(len(pts)):
        print(
            f"  pt {i}: prev={pts[i]}  recovered disp=({disp_ecc[i, 0]:+.2f}, {disp_ecc[i, 1]:+.2f})  "
            f"status={status_ecc[i]}  ecc={conf_ecc[i]:.4f}"
        )

    print("\n--- calcTemplateMatchPyr (TM_CCOEFF_NORMED, subpixel) ---")
    next_pts_tm, status_tm, conf_tm = calcTemplateMatchPyr(
        prev, nxt, pts, template_size=win_size, search_range=search_range, num_levels=num_levels,
        match_method=cv2.TM_CCOEFF_NORMED, subpixel=True,
    )
    disp_tm = next_pts_tm - pts
    for i in range(len(pts)):
        print(
            f"  pt {i}: prev={pts[i]}  recovered disp=({disp_tm[i, 0]:+.2f}, {disp_tm[i, 1]:+.2f})  "
            f"status={status_tm[i]}  confidence={conf_tm[i]:.3f}"
        )

    print(f"\nGround truth displacement: ({true_dx:+.2f}, {true_dy:+.2f})")

    tol = 0.5  # pixels -- clean synthetic data should recover sub-pixel accuracy
    assert np.all(status_ecc == 1), "calcEccPyr: not all points converged"
    assert np.allclose(disp_ecc[:, 0], true_dx, atol=tol), "calcEccPyr: dx too far from ground truth"
    assert np.allclose(disp_ecc[:, 1], true_dy, atol=tol), "calcEccPyr: dy too far from ground truth"

    assert np.all(status_tm == 1), "calcTemplateMatchPyr: not all points converged"
    assert np.allclose(disp_tm[:, 0], true_dx, atol=tol), "calcTemplateMatchPyr: dx too far from ground truth"
    assert np.allclose(disp_tm[:, 1], true_dy, atol=tol), "calcTemplateMatchPyr: dy too far from ground truth"

    print("\nAll points recovered within tolerance for both trackers. OK.")

    # --- Border-handling demonstration (not asserted -- just illustrating graceful failure) ---
    # A point whose window has no overlap with the image at all (e.g. a mistakenly
    # out-of-frame coordinate, or a point that has drifted off-frame between frames).
    # `_clip_window` shrinks the requested window to fit the image, and once that shrunk
    # window is smaller than MIN_WINDOW_PX in either dimension, the point is marked failed
    # instead of being handed to OpenCV (which would otherwise raise on an empty crop).
    print("\n--- Border handling demo: a point with no valid window inside the image ---")
    border_pt = np.array([[-30.0, -30.0]])  # entirely outside the top-left corner
    next_pts_border, status_border, conf_border, _ = calcEccPyr(
        prev, nxt, border_pt, win_size, search_range, num_levels,
        motion_type=cv2.MOTION_TRANSLATION, criteria=criteria,
    )
    print(
        f"  border point status={status_border[0]} (0 = gracefully marked failed, no exception raised); "
        f"confidence={conf_border[0]:.3f}"
    )
    assert status_border[0] == 0, "expected the out-of-frame point to be marked failed"

    # A point close enough to the border that its window is legitimately shrunk but still
    # usable -- this should still succeed (clip, don't fail, whenever there's still a
    # usable window).
    print("\n--- Border handling demo: a point near (but not off) the edge, window shrunk but usable ---")
    near_edge_pt = np.array([[15.0, 15.0]])
    next_pts_edge, status_edge, conf_edge, _ = calcEccPyr(
        prev, nxt, near_edge_pt, win_size, search_range, num_levels,
        motion_type=cv2.MOTION_TRANSLATION, criteria=criteria,
    )
    print(f"  near-edge point status={status_edge[0]}, confidence={conf_edge[0]:.3f} (window was clipped, not rejected)")
