"""Spatial geometry and extended light ray integration for laptop screen photometric stereo."""

from dataclasses import dataclass
import numpy as np


@dataclass
class SetupGeometry:
    """Physical configuration of screen, webcam, and object plane in meters."""

    screen_width: float = 0.310  # 14-inch screen approx 310mm width
    screen_height: float = 0.175  # approx 175mm height
    screen_tilt_deg: float = 105.0  # Angle between desk and screen normal/plane
    camera_offset_y: float = 0.010  # Webcam slightly above top of screen bezel
    object_distance_z: float = 0.220  # Distance along desk from screen base to card center
    object_roi_size: float = 0.060  # Inspection window size (60mm x 60mm)

    @property
    def tilt_rad(self) -> float:
        return np.deg2rad(self.screen_tilt_deg)


class ScreenLightIntegrator:
    """Computes integrated light direction vectors L_p(x) for surface positions."""

    def __init__(self, geom: SetupGeometry = SetupGeometry(), grid_res: tuple[int, int] = (16, 16)):
        self.geom = geom
        self.grid_res = grid_res
        self._init_screen_emitter_grid()

    def _init_screen_emitter_grid(self):
        """Discretize the screen surface into a grid of directional differential emitters."""
        ny, nx = self.grid_res
        # Screen coordinates: base of screen is at y=0, z=0.
        # Screen tilts backward by (geom.screen_tilt_deg - 90) relative to vertical.
        phi = self.geom.tilt_rad - (np.pi / 2.0)  # Tilt relative to z-axis

        xs = np.linspace(-self.geom.screen_width / 2.0, self.geom.screen_width / 2.0, nx)
        ys = np.linspace(0.0, self.geom.screen_height, ny)

        grid_x, grid_y = np.meshgrid(xs, ys)

        # 3D coordinates of screen points in world frame:
        # World origin = base center of screen on desk.
        # X: left-to-right along screen base
        # Y: vertical (up from desk)
        # Z: forward along desk (towards user/camera view)
        p_x = grid_x
        p_y = grid_y * np.cos(phi)
        p_z = -grid_y * np.sin(phi)

        self.emitters = np.stack([p_x, p_y, p_z], axis=-1).reshape(-1, 3)

        # Screen surface normal pointing towards the desk
        normal = np.array([0.0, -np.sin(phi), np.cos(phi)], dtype=np.float32)
        normal /= np.linalg.norm(normal)
        self.screen_normal = normal

    def compute_lighting_vector(self, pattern_weights: np.ndarray, surface_pt: np.ndarray) -> np.ndarray:
        """Compute net directional light vector L_p for a surface point x.

        Args:
            pattern_weights: 2D array (ny, nx) of brightness in [0, 1] across screen emitters
            surface_pt: (3,) [x, y, z] position on desk (y=0)

        Returns:
            (3,) normalized lighting direction vector L_p
        """
        w = pattern_weights.reshape(-1)
        # Vectors from surface point to each screen emitter
        vecs = self.emitters - surface_pt  # (N, 3)
        dist_sq = np.sum(vecs**2, axis=-1, keepdims=True)  # (N, 1)
        dist = np.sqrt(dist_sq)

        # Cosine of emission angle: -vecs . screen_normal / dist
        cos_emit = np.clip(np.sum(-vecs * self.screen_normal, axis=-1, keepdims=True) / (dist + 1e-6), 0.0, 1.0)

        # Radiance contribution: w_i * cos_emit / dist^2 * (vecs / dist)
        contributions = w[:, np.newaxis] * cos_emit * (vecs / (dist**3 + 1e-6))
        net_l = np.sum(contributions, axis=0)

        norm_l = np.linalg.norm(net_l)
        if norm_l < 1e-6:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return (net_l / norm_l).astype(np.float32)

    def sample_pattern_to_grid(self, pattern_img: np.ndarray) -> np.ndarray:
        """Downsample full screen pattern image (H, W, 3) to emitter grid resolution."""
        import cv2

        ny, nx = self.grid_res
        if pattern_img.ndim == 3:
            gray = cv2.cvtColor(pattern_img, cv2.COLOR_BGR2GRAY if pattern_img.dtype == np.uint8 else cv2.COLOR_RGB2GRAY)
        else:
            gray = pattern_img

        resized = cv2.resize(gray, (nx, ny), interpolation=cv2.INTER_AREA)
        if resized.dtype == np.uint8:
            return (resized / 255.0).astype(np.float32)
        return resized.astype(np.float32)
