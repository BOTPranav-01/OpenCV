"""Radiometric linearization using sRGB inverse transfer function.

Physical camera sensors record photon counts linearly, but standard camera pipelines
apply non-linear tone curves (sRGB). Photometric stereo requires true linear irradiance.
"""

import numpy as np


class RadiometricLinearizer:
    """Fast 8-bit lookup table (LUT) and float inverse sRGB conversion."""

    def __init__(self):
        # Build 256-entry float32 LUT
        idx = np.arange(256, dtype=np.float32) / 255.0
        # Official IEC 61966-2-1 inverse sRGB definition
        linear = np.where(idx <= 0.04045, idx / 12.92, np.power((idx + 0.055) / 1.055, 2.4))
        self.lut = linear.astype(np.float32)

    def to_linear(self, img: np.ndarray) -> np.ndarray:
        """Convert uint8 [0, 255] or float [0, 1] sRGB image to linear irradiance [0, 1]."""
        if img.dtype == np.uint8:
            return self.lut[img]

        # Float input in [0, 1]
        clamped = np.clip(img, 0.0, 1.0)
        return np.where(clamped <= 0.04045, clamped / 12.92, np.power((clamped + 0.055) / 1.055, 2.4)).astype(np.float32)

    def to_luminance(self, linear_rgb: np.ndarray) -> np.ndarray:
        """Standard Rec.709 photometric luminance calculation from linear RGB."""
        if linear_rgb.ndim == 2:
            return linear_rgb
        # ITU-R BT.709 coefficients: Y = 0.2126 R + 0.7152 G + 0.0722 B
        return (0.2126 * linear_rgb[..., 0] + 0.7152 * linear_rgb[..., 1] + 0.0722 * linear_rgb[..., 2]).astype(np.float32)
