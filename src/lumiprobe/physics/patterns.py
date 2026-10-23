"""Screen-based lighting pattern generators for active photometric stereo.

Generates calibrated physical light patterns on the display screen:
- Dark frame (ambient estimation)
- Uniform white (albedo & baseline)
- Linear spatial gradients (Francken-style normal estimation)
- Half-planes (directional shadow contrast)
- Grazing bars (directional high-frequency surface feature excitation)
- Directional spots (localized oblique lighting)
"""

import numpy as np


class PatternBank:
    """Factory and registry for display illumination patterns."""

    @staticmethod
    def gamma_correct(linear: np.ndarray, gamma: float = 2.2) -> np.ndarray:
        """Apply display gamma encoding from linear light [0, 1] to sRGB."""
        clipped = np.clip(linear, 0.0, 1.0)
        return np.power(clipped, 1.0 / gamma)

    @staticmethod
    def make_dark(width: int = 1920, height: int = 1080) -> np.ndarray:
        """Absolute dark frame (screen black) for ambient baseline subtraction."""
        return np.zeros((height, width, 3), dtype=np.float32)

    @staticmethod
    def make_white(width: int = 1920, height: int = 1080) -> np.ndarray:
        """Uniform white frame for overall albedo scaling."""
        return np.ones((height, width, 3), dtype=np.float32)

    @staticmethod
    def make_gradient_lr(width: int = 1920, height: int = 1080) -> np.ndarray:
        """Left-to-Right linear intensity ramp: 0.0 -> 1.0."""
        x = np.linspace(0.0, 1.0, width, dtype=np.float32)
        grad = np.tile(x, (height, 1))
        return np.repeat(grad[:, :, np.newaxis], 3, axis=2)

    @staticmethod
    def make_gradient_rl(width: int = 1920, height: int = 1080) -> np.ndarray:
        """Right-to-Left linear intensity ramp: 1.0 -> 0.0."""
        x = np.linspace(1.0, 0.0, width, dtype=np.float32)
        grad = np.tile(x, (height, 1))
        return np.repeat(grad[:, :, np.newaxis], 3, axis=2)

    @staticmethod
    def make_gradient_tb(width: int = 1920, height: int = 1080) -> np.ndarray:
        """Top-to-Bottom linear intensity ramp: 0.0 -> 1.0."""
        y = np.linspace(0.0, 1.0, height, dtype=np.float32)
        grad = np.tile(y[:, np.newaxis], (1, width))
        return np.repeat(grad[:, :, np.newaxis], 3, axis=2)

    @staticmethod
    def make_gradient_bt(width: int = 1920, height: int = 1080) -> np.ndarray:
        """Bottom-to-Top linear intensity ramp: 1.0 -> 0.0."""
        y = np.linspace(1.0, 0.0, height, dtype=np.float32)
        grad = np.tile(y[:, np.newaxis], (1, width))
        return np.repeat(grad[:, :, np.newaxis], 3, axis=2)

    @staticmethod
    def make_half_plane(side: str, width: int = 1920, height: int = 1080) -> np.ndarray:
        """Half-plane illumination (side in {'left', 'right', 'top', 'bottom'})."""
        img = np.zeros((height, width, 3), dtype=np.float32)
        if side == "left":
            img[:, : width // 2, :] = 1.0
        elif side == "right":
            img[:, width // 2 :, :] = 1.0
        elif side == "top":
            img[: height // 2, :, :] = 1.0
        elif side == "bottom":
            img[height // 2 :, :, :] = 1.0
        return img

    @staticmethod
    def make_bar(angle_deg: float, width_px: int = 240, width: int = 1920, height: int = 1080) -> np.ndarray:
        """Bright grazing bar oriented at specified angle across screen center."""
        img = np.zeros((height, width), dtype=np.float32)
        cx, cy = width / 2.0, height / 2.0
        theta = np.deg2rad(angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        y_indices, x_indices = np.indices((height, width), dtype=np.float32)
        # Distance to line passing through center
        dist = np.abs((x_indices - cx) * (-sin_t) + (y_indices - cy) * cos_t)
        mask = dist <= (width_px / 2.0)
        img[mask] = 1.0
        return np.repeat(img[:, :, np.newaxis], 3, axis=2)

    @staticmethod
    def make_spot(cx_ratio: float, cy_ratio: float, radius_ratio: float = 0.2, width: int = 1920, height: int = 1080) -> np.ndarray:
        """Movable circular spot with smooth cosine falloff for directional oblique lighting."""
        cx, cy = cx_ratio * width, cy_ratio * height
        radius = radius_ratio * min(width, height)

        y_indices, x_indices = np.indices((height, width), dtype=np.float32)
        r = np.sqrt((x_indices - cx) ** 2 + (y_indices - cy) ** 2)

        spot = np.clip((1.0 - (r / radius)), 0.0, 1.0)
        # Smooth falloff
        spot = 0.5 * (1.0 - np.cos(np.pi * spot))
        return np.repeat(spot[:, :, np.newaxis], 3, axis=2)

    @classmethod
    def get_pattern(cls, pattern_id: str, width: int = 1920, height: int = 1080) -> np.ndarray:
        """Lookup and render pattern array."""
        pid = pattern_id.upper()
        if pid == "DARK":
            return cls.make_dark(width, height)
        if pid == "WHITE":
            return cls.make_white(width, height)
        if pid == "GRAD_LR":
            return cls.make_gradient_lr(width, height)
        if pid == "GRAD_RL":
            return cls.make_gradient_rl(width, height)
        if pid == "GRAD_TB":
            return cls.make_gradient_tb(width, height)
        if pid == "GRAD_BT":
            return cls.make_gradient_bt(width, height)
        if pid == "HALF_L":
            return cls.make_half_plane("left", width, height)
        if pid == "HALF_R":
            return cls.make_half_plane("right", width, height)
        if pid == "HALF_T":
            return cls.make_half_plane("top", width, height)
        if pid == "HALF_B":
            return cls.make_half_plane("bottom", width, height)
        if pid.startswith("BAR_"):
            try:
                angle = float(pid.split("_")[1])
                return cls.make_bar(angle, width=width, height=height)
            except Exception:
                return cls.make_bar(0.0, width=width, height=height)
        if pid == "SPOT_TL":
            return cls.make_spot(0.2, 0.2, 0.35, width, height)
        if pid == "SPOT_TR":
            return cls.make_spot(0.8, 0.2, 0.35, width, height)
        if pid == "SPOT_BL":
            return cls.make_spot(0.2, 0.8, 0.35, width, height)
        if pid == "SPOT_BR":
            return cls.make_spot(0.8, 0.8, 0.35, width, height)
        if pid == "SPOT_C":
            return cls.make_spot(0.5, 0.5, 0.4, width, height)

        # Fallback to white
        return cls.make_white(width, height)

    @classmethod
    def all_pattern_ids(cls) -> list[str]:
        """Return canonical list of pattern IDs."""
        return [
            "DARK",
            "WHITE",
            "GRAD_LR",
            "GRAD_RL",
            "GRAD_TB",
            "GRAD_BT",
            "HALF_L",
            "HALF_R",
            "HALF_T",
            "HALF_B",
            "BAR_0",
            "BAR_45",
            "BAR_90",
            "BAR_135",
            "SPOT_TL",
            "SPOT_TR",
            "SPOT_BL",
            "SPOT_BR",
        ]
