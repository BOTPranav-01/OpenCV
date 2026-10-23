"""Frankot-Chellappa surface height integration from normal gradients.

Solves the Poisson equation in the 2D Fourier domain to reconstruct
integrable 3D depth/height maps from noisy surface gradients p = dz/dx, q = dz/dy.
Reference: Frankot & Chellappa, IEEE TPAMI 1988.
"""

import numpy as np


class FrankotChellappaIntegrator:
    """Fourier-domain Poisson solver for 3D surface reconstruction."""

    @staticmethod
    def integrate(
        dz_dx: np.ndarray,
        dz_dy: np.ndarray,
        scale_factor: float = 1.0,
        pad_width: int = 16,
    ) -> np.ndarray:
        """Reconstruct 3D height map z(x, y) from partial derivatives dz/dx and dz/dy.

        Args:
            dz_dx: (H, W) partial derivatives along horizontal axis
            dz_dy: (H, W) partial derivatives along vertical axis
            scale_factor: scaling factor for relative height
            pad_width: boundary padding to eliminate spectral wraparound leakage

        Returns:
            (H, W) float32 reconstructed surface height map
        """
        h, w = dz_dx.shape

        # Symmetric padding to reduce FFT boundary discontinuities
        if pad_width > 0:
            p = np.pad(dz_dx, pad_width, mode="reflect")
            q = np.pad(dz_dy, pad_width, mode="reflect")
        else:
            p = dz_dx
            q = dz_dy

        ph, pw = p.shape

        # Discrete Fourier frequencies
        # Standard FFT coordinates [-pi, pi]
        u = np.fft.fftfreq(pw).reshape(1, -1) * 2.0 * np.pi
        v = np.fft.fftfreq(ph).reshape(-1, 1) * 2.0 * np.pi

        # 2D FFT of gradient fields
        p_hat = np.fft.fft2(p)
        q_hat = np.fft.fft2(q)

        # Denominator: u^2 + v^2
        denom = u**2 + v**2
        denom[0, 0] = 1.0  # Avoid division by zero at DC

        # Frankot-Chellappa formula:
        # Z_hat = (-1j * u * P_hat - 1j * v * Q_hat) / (u^2 + v^2)
        z_hat = (-1j * u * p_hat - 1j * v * q_hat) / denom
        z_hat[0, 0] = 0.0  # Set zero mean height (DC component)

        # Inverse 2D FFT
        z_padded = np.real(np.fft.ifft2(z_hat))

        # Crop back to original dimensions
        if pad_width > 0:
            z = z_padded[pad_width : pad_width + h, pad_width : pad_width + w]
        else:
            z = z_padded

        # Normalize height: zero mean and scaled
        z = (z - np.mean(z)) * scale_factor
        return z.astype(np.float32)

    @staticmethod
    def height_to_obj_mesh(height_map: np.ndarray, step: int = 2) -> str:
        """Convert height map to Wavefront OBJ format for 3D viewing."""
        h, w = height_map.shape
        sub_h = height_map[::step, ::step]
        sh, sw = sub_h.shape

        lines = ["# LumiProbe Reconstructed 3D Surface Mesh\n"]

        # Normalize coordinates to [-1, 1] in XY
        xs = np.linspace(-1.0, 1.0, sw)
        ys = np.linspace(1.0, -1.0, sh)

        # Emit vertices
        for r in range(sh):
            y = ys[r]
            for c in range(sw):
                x = xs[c]
                z = sub_h[r, c]
                lines.append(f"v {x:.4f} {y:.4f} {z:.4f}\n")

        # Emit quad/tri faces (1-indexed)
        def v_idx(r, c):
            return r * sw + c + 1

        for r in range(sh - 1):
            for c in range(sw - 1):
                v1 = v_idx(r, c)
                v2 = v_idx(r, c + 1)
                v3 = v_idx(r + 1, c + 1)
                v4 = v_idx(r + 1, c)
                # Two triangles per grid cell
                lines.append(f"f {v1} {v2} {v3}\n")
                lines.append(f"f {v1} {v3} {v4}\n")

        return "".join(lines)
