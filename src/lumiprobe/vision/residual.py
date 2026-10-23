"""Surface normal ratios and reference-plane residual computation.

Cancels surface albedo via Francken-style gradient illumination ratios
and eliminates near-field screen curvature using robust polynomial baseline fitting.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class NormalFieldResult:
    """Estimated gradient ratios, fitted baseline, and residual normal field."""

    r_x: np.ndarray  # Raw x-gradient ratio
    r_y: np.ndarray  # Raw y-gradient ratio
    baseline_x: np.ndarray  # Fitted low-order background polynomial
    baseline_y: np.ndarray  # Fitted low-order background polynomial
    residual_x: np.ndarray  # Delta r_x (surface relief gradient x)
    residual_y: np.ndarray  # Delta r_y (surface relief gradient y)
    normals: np.ndarray  # (H, W, 3) unit normal vector field
    residual_energy: float  # High-frequency energy metric
    valid_mask: np.ndarray  # Saturated/underexposed exclusion mask


class SurfaceResidualEstimator:
    """Estimates normal fields and detects micro-relief through polynomial residual analysis."""

    def __init__(self, poly_degree: int = 2, eps: float = 1e-4):
        self.poly_degree = poly_degree
        self.eps = eps

    def fit_polynomial_surface(self, field: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Fit a low-order 2D polynomial surface P(x, y) to a field using least-squares."""
        h, w = field.shape
        y_coords, x_coords = np.indices((h, w), dtype=np.float32)

        # Normalize coordinates to [-1, 1] for numerical stability
        xn = (2.0 * x_coords / (w - 1.0)) - 1.0
        yn = (2.0 * y_coords / (h - 1.0)) - 1.0

        terms = []
        for i in range(self.poly_degree + 1):
            for j in range(self.poly_degree + 1 - i):
                terms.append((xn**i) * (yn**j))

        # Vandermonde matrix: (H*W, num_terms)
        A = np.column_stack([t.ravel() for t in terms])
        b = field.ravel()

        if mask is not None:
            valid = mask.ravel() > 0
            if np.sum(valid) > A.shape[1]:
                coeffs, _, _, _ = np.linalg.lstsq(A[valid], b[valid], rcond=None)
            else:
                coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        else:
            coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        fitted = (A @ coeffs).reshape((h, w))
        return fitted.astype(np.float32)

    def estimate_from_gradients(
        self,
        j_lr: np.ndarray,
        j_rl: np.ndarray,
        j_tb: np.ndarray,
        j_bt: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> NormalFieldResult:
        """Compute normal ratios, fit reference plane, and compute residual normals."""
        # Albedo-cancelled ratios:
        # r_x = (J_LR - J_RL) / (J_LR + J_RL)
        denom_x = j_lr + j_rl + self.eps
        r_x = (j_lr - j_rl) / denom_x

        # r_y = (J_TB - J_BT) / (J_TB + J_BT)
        denom_y = j_tb + j_bt + self.eps
        r_y = (j_tb - j_bt) / denom_y

        # Build mask for valid linear irradiance (not saturated, adequate SNR)
        if valid_mask is None:
            valid_mask = (denom_x > 0.02) & (denom_y > 0.02)

        # Fit near-field screen geometric baseline polynomial
        base_x = self.fit_polynomial_surface(r_x, valid_mask)
        base_y = self.fit_polynomial_surface(r_y, valid_mask)

        # Subtract baseline: leaves pure noise on flat planes, relief signal on real 3D objects
        res_x = np.where(valid_mask, r_x - base_x, 0.0).astype(np.float32)
        res_y = np.where(valid_mask, r_y - base_y, 0.0).astype(np.float32)

        # High-frequency residual variance/energy
        if np.any(valid_mask):
            energy = float(np.var(res_x[valid_mask]) + np.var(res_y[valid_mask]))
        else:
            energy = 0.0

        # Construct unit normal field n = normalize(res_x, res_y, 1.0)
        norm_mag = np.sqrt(res_x**2 + res_y**2 + 1.0)
        nx = res_x / norm_mag
        ny = res_y / norm_mag
        nz = 1.0 / norm_mag
        normals = np.stack([nx, ny, nz], axis=-1).astype(np.float32)

        return NormalFieldResult(
            r_x=r_x.astype(np.float32),
            r_y=r_y.astype(np.float32),
            baseline_x=base_x,
            baseline_y=base_y,
            residual_x=res_x,
            residual_y=res_y,
            normals=normals,
            residual_energy=energy,
            valid_mask=valid_mask,
        )
