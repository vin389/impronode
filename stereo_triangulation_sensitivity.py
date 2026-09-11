"""
stereo_triangulation_sensitivity.py
====================================

Utilities for quantifying how sensitive a stereo-triangulated 3D point is to
small (sub-pixel to few-pixel) errors in the four measured image coordinates
that feed it: (x_left, y_left, x_right, y_right).

PHYSICAL / MATHEMATICAL BACKGROUND
-----------------------------------
Given a 3D point P = (X, Y, Z), each camera's forward projection is:

    p_L = pi_L(P)   # 2-vector (x_left,  y_left ), via cv2.projectPoints
    p_R = pi_R(P)   # 2-vector (x_right, y_right), via cv2.projectPoints

Stacking both cameras gives an over-determined map from 3 unknowns to 4
observations:

    F(P) = [ pi_L(P) ]        dF/dP = J = [ dpi_L/dP ]     J has shape (4, 3)
           [ pi_R(P) ]                    [ dpi_R/dP ]

Real stereo triangulation (cv2.triangulatePoints, or any bundle-adjustment
style triangulation) solves for P via a (locally linearized) least-squares
fit against all 4 measurements simultaneously. The first-order sensitivity
of that least-squares-optimal P to a small perturbation in the 4 image
measurements is the Moore-Penrose pseudo-inverse of J:

    delta_P = pinv(J) @ delta_(x_left, y_left, x_right, y_right)
    pinv(J) has shape (3, 4)

Splitting pinv(J) by columns gives exactly the two Jacobians this module
returns:

    jacobian_left  = pinv(J)[:, 0:2]   shape (3, 2)   d(X,Y,Z)/d(x_left,  y_left )
    jacobian_right = pinv(J)[:, 2:4]   shape (3, 2)   d(X,Y,Z)/d(x_right, y_right)

IMPORTANT: because matrix inversion mixes information across ALL rows,
jacobian_left is NOT computable from the left camera's parameters alone --
it inherently depends on BOTH cameras' full parameter sets. Physically: how
much a triangulated point moves in response to a 1-pixel error in ONE
camera's measurement depends on how strongly the OTHER camera's independent
measurement constrains that same point (baseline / vergence geometry) -- a
single camera alone cannot even pose a triangulation-sensitivity question,
since one 2D measurement only constrains a ray, not a point.

Jacobians are estimated by CENTRAL FINITE DIFFERENCES on the 3D point
(perturbing X, Y, Z independently and re-projecting with cv2.projectPoints),
rather than analytically differentiating the distortion model. This means
the approach works unmodified for any distortion model cv2.projectPoints
supports, at the cost of a small, controllable numerical-differentiation
error (see finite_diff_step).

WORKING-PLANE GRID (Q4 BILINEAR INTERPOLATION)
------------------------------------------------
The grid of 3D points sensitivity is evaluated at is defined the same way a
4-node (Q4) finite element's geometry is defined: 4 corner points in 3D plus
bilinear shape-function interpolation between them. The 4 corners need not
be exactly coplanar -- bilinear interpolation still produces a well-defined
doubly-ruled surface through them, which is useful in practice since a real
specimen surface located by 4 corner points is rarely perfectly planar.

Corner ordering (REQUIRED -- standard Q4 element convention, counter-
clockwise, starting at the corner that maps to grid index (0,0)):

        corner3 -----------------  corner2
           |     eta=+1               |
           |                          |
           |          o P(xi,eta)     |
           |                          |
           |     eta=-1               |
        corner0 -----------------  corner1
              xi=-1            xi=+1

    corner0 -> grid index (ix=0,      iy=0     )
    corner1 -> grid index (ix=nx-1,   iy=0     )
    corner2 -> grid index (ix=nx-1,   iy=ny-1  )
    corner3 -> grid index (ix=0,      iy=ny-1  )

Supplying corners in a different order (e.g. swapping corner1 and corner3)
silently produces a TWISTED / self-intersecting grid rather than raising an
error -- there is no way to detect "wrong" order from the corners alone, so
getting this right is the caller's responsibility.

DESIGN DECISIONS MADE WHERE THE REQUEST WAS AMBIGUOUS (documented here and
again at the point they matter, below):

  1. grid_x_left / grid_y_left / grid_x_right / grid_y_right are each
     returned with shape (ny, nx) -- NOT (ny, nx, 2). They were requested as
     four SEPARATE return values (x and y split apart), so each one is a
     plain scalar-per-grid-cell field; the "2" only makes sense if x and y
     were packed into one array, which they are not here.

  2. working_plane_nx_ny is interpreted as (nx, ny), matching the parameter
     name literally. All returned grid ARRAYS nonetheless use shape
     (ny, nx, ...) -- row count = ny, column count = nx -- matching the
     universal image/matrix convention (row = y, column = x). Do not
     confuse the (nx, ny) INPUT tuple order with the (ny, nx, ...) OUTPUT
     array shape order; a comment is placed at the reshape calls below as
     a reminder.

  3. No image-bounds / behind-camera validity mask is computed or returned,
     since none was requested in this interface. If you need to know which
     grid cells actually fall inside each camera's real image extent,
     compare grid_x_left/grid_y_left (and the _right pair) against your
     known image width/height after calling this function -- it is a cheap
     check the caller can do directly on the returned arrays.

  4. Points that project behind a camera, or where the two cameras' rays
     are (near-)parallel (degenerate stereo geometry), are NOT specially
     flagged. In those cases np.linalg.pinv still returns a (least-squares
     minimum-norm) result, but the corresponding worst_case_sensitivity_
     per_px value will simply be very large (or, in exact degeneracy,
     numerically huge) -- which is in fact the CORRECT sensitivity
     signal ("triangulation is extremely poorly conditioned here"), so no
     extra handling is needed for that case specifically. It is worth
     visually checking worst_case_sensitivity_per_px for outlier-large
     values as a matter of course when reviewing results from this module.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Input coercion helpers (self-contained utility module -- no dependency on
# the node-editor framework, so these are re-declared locally rather than
# imported from a node file).
# ---------------------------------------------------------------------------

def _coerce_cmat(arr) -> np.ndarray:
    """Coerce a camera matrix to (3,3) float64."""
    return np.asarray(arr, dtype=np.float64).reshape(3, 3)


def _coerce_dvec(arr) -> np.ndarray:
    """Coerce distortion coefficients to (1,K) float64."""
    return np.asarray(arr, dtype=np.float64).ravel().reshape(1, -1)


def _coerce_vec3(arr) -> np.ndarray:
    """Coerce a rotation or translation vector to (3,1) float64."""
    return np.asarray(arr, dtype=np.float64).ravel()[:3].reshape(3, 1)


def _coerce_object_points(arr: np.ndarray) -> np.ndarray:
    """Reshape an (N,3)-ish array to the (N,1,3) float64 shape
    cv2.projectPoints expects. Raises if the element count is not a
    multiple of 3 (mirrors the coercion rule used elsewhere in this
    project for object-point inputs)."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size % 3 != 0:
        raise ValueError(
            f"object points array has {arr.size} elements, "
            f"which is not a multiple of 3"
        )
    return arr.reshape(-1, 1, 3)


# ---------------------------------------------------------------------------
# Q4 bilinear grid generation
# ---------------------------------------------------------------------------

def _q4_shape_functions(
    xi: np.ndarray, eta: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Standard 4-node (Q4) bilinear finite-element shape functions, evaluated
    at parametric coordinates (xi, eta) in [-1, 1] x [-1, 1].

    N0 corresponds to corner0 (xi=-1,eta=-1), N1 to corner1 (xi=+1,eta=-1),
    N2 to corner2 (xi=+1,eta=+1), N3 to corner3 (xi=-1,eta=+1) -- matching
    the corner-ordering convention documented in the module docstring.

    xi, eta may be arrays of any matching shape; N0..N3 share that shape,
    ready to use directly as interpolation weights.
    """
    n0 = 0.25 * (1.0 - xi) * (1.0 - eta)
    n1 = 0.25 * (1.0 + xi) * (1.0 - eta)
    n2 = 0.25 * (1.0 + xi) * (1.0 + eta)
    n3 = 0.25 * (1.0 - xi) * (1.0 + eta)
    return n0, n1, n2, n3


def _build_q4_grid(corners: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """
    Bilinearly interpolate a (ny, nx, 3) grid of 3D points across the 4
    corners of a Q4-style patch. See module docstring for corner ordering.

    Grid index [iy, ix] corresponds to parametric coordinates:
        xi  = -1 + 2 * ix / (nx - 1)
        eta = -1 + 2 * iy / (ny - 1)
    so grid[0, 0] == corners[0] and grid[ny-1, nx-1] == corners[2] exactly
    (a Q4 interpolation always reproduces its 4 corners exactly).
    """
    corners = np.asarray(corners, dtype=np.float64).reshape(4, 3)

    xi_1d = np.linspace(-1.0, 1.0, nx)
    eta_1d = np.linspace(-1.0, 1.0, ny)
    # indexing="xy" makes xi vary along axis 1 (columns) and eta along
    # axis 0 (rows), giving each array shape (ny, nx) directly.
    xi, eta = np.meshgrid(xi_1d, eta_1d, indexing="xy")

    n0, n1, n2, n3 = _q4_shape_functions(xi, eta)   # each (ny, nx)

    grid = (
        n0[..., np.newaxis] * corners[0]
        + n1[..., np.newaxis] * corners[1]
        + n2[..., np.newaxis] * corners[2]
        + n3[..., np.newaxis] * corners[3]
    )
    return grid   # (ny, nx, 3)


# ---------------------------------------------------------------------------
# Core per-point sensitivity computation
# ---------------------------------------------------------------------------

def calc_stereo_triangulation_sensitivity(
    object_points: np.ndarray,
    camera_matrix_left: np.ndarray,
    dist_coeffs_left: np.ndarray,
    rvec_left: np.ndarray,
    tvec_left: np.ndarray,
    camera_matrix_right: np.ndarray,
    dist_coeffs_right: np.ndarray,
    rvec_right: np.ndarray,
    tvec_right: np.ndarray,
    finite_diff_step: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute first-order stereo-triangulation sensitivity Jacobians at a
    batch of N arbitrary 3D world points. See module docstring for the
    full derivation.

    Parameters
    ----------
    object_points : (N,3) or (3,) array of 3D world points.
    camera_matrix_*, dist_coeffs_*, rvec_*, tvec_* : standard OpenCV
        per-camera calibration parameters. Shapes/dtype are coerced
        automatically so callers do not need to pre-reshape calibration
        results (e.g. output straight from a calibration or
        ProjectPointsNode-style pipeline works as-is).
    finite_diff_step : central-difference step size in WORLD units (the
        same units as object_points / tvec, e.g. mm). Default 1e-3 (i.e.
        1 micron if world units are mm) is small enough to stay in the
        locally-linear regime of the projection + distortion model, and
        large enough to avoid float64 cancellation error. Scale this if
        your working volume's units are very different from mm.

    Returns
    -------
    jacobian_left     (N,3,2)  d(X,Y,Z)/d(x_left,  y_left ), holding the
                                right-camera measurement fixed.
    jacobian_right    (N,3,2)  d(X,Y,Z)/d(x_right, y_right), holding the
                                left-camera measurement fixed.
    singular_values   (N,3)    singular values (descending) of the combined
                                (3,4) pseudo-inverse Jacobian
                                (= concatenate([jacobian_left,
                                jacobian_right], axis=-1)). These are
                                singular values of the PSEUDO-INVERSE, not
                                of the forward Jacobian -- i.e. already in
                                "world-units-moved per pixel-of-error" form.
    worst_case_sensitivity_per_px  (N,)  the largest of the 3 singular
                                values above: the largest possible 3D
                                displacement resulting from a combined
                                1-pixel-magnitude error across all 4
                                measurements, in the worst-case error
                                direction. Large values flag geometrically
                                weak triangulation (e.g. narrow baseline,
                                point far outside the calibrated volume,
                                near-parallel viewing rays).
    image_point_left  (N,2)   projected left-image coordinates of
                                object_points (returned for convenience,
                                e.g. sanity-checking against image bounds).
    image_point_right (N,2)   projected right-image coordinates.
    """
    cmat_l = _coerce_cmat(camera_matrix_left)
    dvec_l = _coerce_dvec(dist_coeffs_left)
    rvec_l = _coerce_vec3(rvec_left)
    tvec_l = _coerce_vec3(tvec_left)
    cmat_r = _coerce_cmat(camera_matrix_right)
    dvec_r = _coerce_dvec(dist_coeffs_right)
    rvec_r = _coerce_vec3(rvec_right)
    tvec_r = _coerce_vec3(tvec_right)

    if finite_diff_step <= 0:
        raise ValueError("finite_diff_step must be positive")

    pts = _coerce_object_points(object_points)   # (N,1,3)
    n = pts.shape[0]
    eps = float(finite_diff_step)

    # --- baseline projections (also returned to the caller) ----------------
    img_l, _ = cv2.projectPoints(pts, rvec_l, tvec_l, cmat_l, dvec_l)
    img_r, _ = cv2.projectPoints(pts, rvec_r, tvec_r, cmat_r, dvec_r)
    img_l = img_l.reshape(n, 2)
    img_r = img_r.reshape(n, 2)

    # --- build all 6*N perturbed points in ONE batch (3 axes x +/-eps) ------
    # This is what makes the whole grid function cheap: rather than looping
    # per point, every perturbed point for every axis/sign/grid-cell is
    # projected in exactly 2 cv2.projectPoints calls total (one per camera).
    base = pts.reshape(n, 3)
    perturbed_plus = np.empty((3, n, 3), dtype=np.float64)
    perturbed_minus = np.empty((3, n, 3), dtype=np.float64)
    for axis in range(3):
        offset = np.zeros(3, dtype=np.float64)
        offset[axis] = eps
        perturbed_plus[axis] = base + offset
        perturbed_minus[axis] = base - offset

    # Stack into one (6N,1,3) batch, ordered [+X,+Y,+Z,-X,-Y,-Z], N points each.
    all_plus = perturbed_plus.reshape(3 * n, 1, 3)
    all_minus = perturbed_minus.reshape(3 * n, 1, 3)
    all_pts = np.concatenate([all_plus, all_minus], axis=0)   # (6N,1,3)

    proj_l_all, _ = cv2.projectPoints(all_pts, rvec_l, tvec_l, cmat_l, dvec_l)
    proj_r_all, _ = cv2.projectPoints(all_pts, rvec_r, tvec_r, cmat_r, dvec_r)
    proj_l_all = proj_l_all.reshape(6, n, 2)   # [+X,+Y,+Z,-X,-Y,-Z] x N x (u,v)
    proj_r_all = proj_r_all.reshape(6, n, 2)

    # central difference: d/d(axis) ~= (f(+eps) - f(-eps)) / (2*eps)
    jac_l_full = np.empty((n, 2, 3), dtype=np.float64)   # d(u_l,v_l)/d(X,Y,Z)
    jac_r_full = np.empty((n, 2, 3), dtype=np.float64)   # d(u_r,v_r)/d(X,Y,Z)
    for axis in range(3):
        jac_l_full[:, :, axis] = (proj_l_all[axis] - proj_l_all[axis + 3]) / (2.0 * eps)
        jac_r_full[:, :, axis] = (proj_r_all[axis] - proj_r_all[axis + 3]) / (2.0 * eps)

    # Combined FORWARD Jacobian J: (N,4,3). Rows 0:2 = left camera
    # measurements, rows 2:4 = right camera measurements.
    j_forward = np.concatenate([jac_l_full, jac_r_full], axis=1)   # (N,4,3)

    # Pseudo-inverse, batched over the leading N axis -> (N,3,4).
    # np.linalg.pinv supports stacked/batched matrices natively (no Python
    # loop needed), which is what keeps this whole module fast even for
    # tens of thousands of grid points.
    j_pinv = np.linalg.pinv(j_forward)

    jacobian_left = j_pinv[:, :, 0:2]    # (N,3,2)
    jacobian_right = j_pinv[:, :, 2:4]   # (N,3,2)

    # Singular values of the pseudo-inverse itself -- already in "world
    # units moved per pixel of combined measurement error" form.
    singular_values = np.linalg.svd(j_pinv, compute_uv=False)   # (N,3), descending
    worst_case_sensitivity_per_px = singular_values[:, 0]         # (N,)

    return (
        jacobian_left,
        jacobian_right,
        singular_values,
        worst_case_sensitivity_per_px,
        img_l,
        img_r,
    )


# ---------------------------------------------------------------------------
# Working-plane grid sensitivity
# ---------------------------------------------------------------------------

def calc_stereo_triangulation_sensitivity_grid(
    camera_matrix_left: np.ndarray,
    dist_coeffs_left: np.ndarray,
    rvec_left: np.ndarray,
    tvec_left: np.ndarray,
    camera_matrix_right: np.ndarray,
    dist_coeffs_right: np.ndarray,
    rvec_right: np.ndarray,
    tvec_right: np.ndarray,
    working_plane_corners: np.ndarray,
    working_plane_nx_ny: Tuple[int, int],
    finite_diff_step: float = 1e-3,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    """
    Evaluate stereo-triangulation sensitivity over a Q4-bilinear-interpolated
    grid of 3D points spanning a 4-corner "working plane" patch.

    Parameters
    ----------
    camera_matrix_*, dist_coeffs_*, rvec_*, tvec_* : per-camera calibration
        parameters, as in calc_stereo_triangulation_sensitivity().
    working_plane_corners : (4,3) float array, the 3D world coordinates of
        the 4 corners of the patch. MUST be given in the counter-clockwise
        Q4 order documented in the module docstring -- corner0 maps to grid
        index (0,0), corner1 to (nx-1,0), corner2 to (nx-1,ny-1), corner3
        to (0,ny-1). The 4 corners need not be exactly coplanar.
    working_plane_nx_ny : (nx, ny) -- number of grid points along the
        corner0->corner1 direction (nx) and the corner0->corner3 direction
        (ny) respectively. Both must be >= 2 (a Q4 patch needs at least its
        4 corners to be well-defined as a grid). NOTE: the input tuple is
        (nx, ny), but all returned ARRAYS use shape (ny, nx, ...) --
        row-major, matching image/matrix convention (row=y, column=x). Do
        not confuse these two orderings.
    finite_diff_step : see calc_stereo_triangulation_sensitivity().

    Returns
    -------
    jacobian_left, jacobian_right    (ny,nx,3,2)   see
        calc_stereo_triangulation_sensitivity() for exact meaning; here
        evaluated independently at every grid cell.
    singular_values                    (ny,nx,3)     per-cell singular
        values of the combined pseudo-inverse Jacobian, descending order.
    object_points                       (ny,nx,3)     the Q4-interpolated 3D
        point at each grid cell (corner-corner bilinear interpolation).
    grid_x_left, grid_y_left            (ny,nx) each  the left-image pixel
        coordinates each 3D grid point projects to. Returned as TWO
        separate (ny,nx) scalar fields rather than one combined (ny,nx,2)
        array, since they were requested as separate named outputs -- see
        module docstring decision #1.
    grid_x_right, grid_y_right          (ny,nx) each  same, for the right
        camera's projection of the same 3D grid points.
    worst_case_sensitivity_per_px       (ny,nx)       per-cell worst-case
        sensitivity (largest singular value); this is the single scalar
        field most useful for a heatmap / ArrayViewer-style visualization
        of where the stereo rig's triangulation is geometrically weak.

    Notes
    -----
    No image-bounds or behind-camera validity mask is computed here (see
    module docstring decision #3) -- compare grid_x_left/grid_y_left (and
    the _right pair) against your actual image width/height afterward if
    you need to know which grid cells are realizable in both cameras' real
    fields of view.
    """
    nx, ny = working_plane_nx_ny
    if nx < 2 or ny < 2:
        raise ValueError(
            f"working_plane_nx_ny must each be >= 2 to define a grid; "
            f"got nx={nx}, ny={ny}"
        )

    corners = np.asarray(working_plane_corners, dtype=np.float64).reshape(4, 3)

    # (ny, nx, 3) -- see _build_q4_grid docstring for the index convention.
    grid_pts = _build_q4_grid(corners, nx, ny)

    # Flatten to (ny*nx, 3) to reuse the fully-vectorized per-point function
    # above with a single call, rather than looping over grid cells.
    flat_pts = grid_pts.reshape(-1, 3)

    (
        jac_l_flat,
        jac_r_flat,
        sv_flat,
        worst_flat,
        img_l_flat,
        img_r_flat,
    ) = calc_stereo_triangulation_sensitivity(
        flat_pts,
        camera_matrix_left, dist_coeffs_left, rvec_left, tvec_left,
        camera_matrix_right, dist_coeffs_right, rvec_right, tvec_right,
        finite_diff_step=finite_diff_step,
    )

    # Reshape every flat (ny*nx, ...) result back to the (ny, nx, ...) grid
    # shape. NOTE the (nx, ny) INPUT order vs (ny, nx, ...) OUTPUT order --
    # see the working_plane_nx_ny parameter docstring above.
    jacobian_left = jac_l_flat.reshape(ny, nx, 3, 2)
    jacobian_right = jac_r_flat.reshape(ny, nx, 3, 2)
    singular_values = sv_flat.reshape(ny, nx, 3)
    worst_case_sensitivity_per_px = worst_flat.reshape(ny, nx)
    object_points = grid_pts   # already (ny, nx, 3)

    # img_l_flat / img_r_flat are (ny*nx, 2); split into separate x/y fields
    # per decision #1 in the module docstring, each reshaped to (ny, nx).
    grid_x_left = img_l_flat[:, 0].reshape(ny, nx)
    grid_y_left = img_l_flat[:, 1].reshape(ny, nx)
    grid_x_right = img_r_flat[:, 0].reshape(ny, nx)
    grid_y_right = img_r_flat[:, 1].reshape(ny, nx)

    return (
        jacobian_left,
        jacobian_right,
        singular_values,
        object_points,
        grid_x_left,
        grid_y_left,
        grid_x_right,
        grid_y_right,
        worst_case_sensitivity_per_px,
    )


# ---------------------------------------------------------------------------
# Self-contained example / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Setting up a synthetic converging stereo rig...")

    # Simple synthetic rig: left camera at the world origin looking down +Z;
    # right camera translated 200mm along +X, both cameras verged slightly
    # inward (a few degrees of yaw) toward a working volume centered near
    # (0, 0, 1000) mm -- a plausible DIC-style rig geometry.
    focal = 5000.0
    cmat = np.array([
        [focal, 0.0, 960.0],
        [0.0, focal, 540.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dvec = np.zeros((1, 5), dtype=np.float64)

    baseline_mm = 200.0
    verge_deg = 3.0
    verge_rad = np.deg2rad(verge_deg)

    rvec_left = np.zeros((3, 1), dtype=np.float64)
    tvec_left = np.zeros((3, 1), dtype=np.float64)

    # Right camera: translated along X, yawed inward by -verge_deg (toward -X)
    # so both cameras' optical axes converge somewhere in front of the rig.
    rvec_right = np.array([[0.0], [-verge_rad], [0.0]], dtype=np.float64)
    tvec_right = np.array([[baseline_mm], [0.0], [0.0]], dtype=np.float64)

    print("\n--- Single-point sensitivity check ---")
    test_point = np.array([[0.0, 0.0, 1000.0]])   # 1m in front of the rig
    (jac_l, jac_r, sv, worst, img_l, img_r) = calc_stereo_triangulation_sensitivity(
        test_point,
        cmat, dvec, rvec_left, tvec_left,
        cmat, dvec, rvec_right, tvec_right,
        finite_diff_step=1e-3,
    )
    print(f"  left  image point:  {img_l[0]}")
    print(f"  right image point:  {img_r[0]}")
    print(f"  jacobian_left shape:  {jac_l.shape}  (expect (1,3,2))")
    print(f"  jacobian_right shape: {jac_r.shape}  (expect (1,3,2))")
    print(f"  singular values:      {sv[0]}")
    print(f"  worst-case sensitivity: {worst[0]:.4f} mm per px")

    assert jac_l.shape == (1, 3, 2)
    assert jac_r.shape == (1, 3, 2)
    assert np.all(np.isfinite(jac_l)) and np.all(np.isfinite(jac_r))
    assert worst[0] > 0.0

    print("\n--- Grid sensitivity check (Q4 working plane) ---")
    # A 400x300mm patch centered at (0,0,1000), corners in the required
    # counter-clockwise order (corner0 -> (0,0) grid index).
    corners = np.array([
        [-200.0, -150.0, 1000.0],   # corner0: (ix=0,    iy=0)
        [200.0, -150.0, 1000.0],    # corner1: (ix=nx-1, iy=0)
        [200.0, 150.0, 1000.0],     # corner2: (ix=nx-1, iy=ny-1)
        [-200.0, 150.0, 1000.0],    # corner3: (ix=0,    iy=ny-1)
    ], dtype=np.float64)

    nx, ny = 21, 15
    (
        g_jac_l, g_jac_r, g_sv, g_obj,
        g_x_l, g_y_l, g_x_r, g_y_r, g_worst,
    ) = calc_stereo_triangulation_sensitivity_grid(
        cmat, dvec, rvec_left, tvec_left,
        cmat, dvec, rvec_right, tvec_right,
        corners, (nx, ny),
        finite_diff_step=1e-3,
    )

    print(f"  object_points shape:   {g_obj.shape}  (expect ({ny},{nx},3))")
    print(f"  jacobian_left shape:   {g_jac_l.shape}  (expect ({ny},{nx},3,2))")
    print(f"  grid_x_left shape:     {g_x_l.shape}  (expect ({ny},{nx}))")
    print(f"  worst_case sensitivity: min={g_worst.min():.4f}  "
          f"max={g_worst.max():.4f}  mean={g_worst.mean():.4f}  mm/px")

    # Corners must be reproduced exactly by the Q4 interpolation.
    assert np.allclose(g_obj[0, 0], corners[0]), "corner0 not reproduced exactly"
    assert np.allclose(g_obj[0, nx - 1], corners[1]), "corner1 not reproduced exactly"
    assert np.allclose(g_obj[ny - 1, nx - 1], corners[2]), "corner2 not reproduced exactly"
    assert np.allclose(g_obj[ny - 1, 0], corners[3]), "corner3 not reproduced exactly"

    assert g_jac_l.shape == (ny, nx, 3, 2)
    assert g_jac_r.shape == (ny, nx, 3, 2)
    assert g_sv.shape == (ny, nx, 3)
    assert g_worst.shape == (ny, nx)
    assert np.all(np.isfinite(g_worst)) and np.all(g_worst > 0.0)

    print("\nAll shape and corner-reproduction checks passed. OK.")