"""Procedural 3D surface height generator and forward photometric renderer.

Synthesizes physically accurate surface geometries:
- 3D embossed coins with edge lettering
- Tactile Braille dot arrays
- Industrial scratched / cracked parts
- Flat 2D printed photographs (counterfeit / spoof) with zero 3D height
Simulates radiometric camera capture under arbitrary screen lighting patterns.
"""

from dataclasses import dataclass
import cv2
import numpy as np
from lumiprobe.physics.geometry import ScreenLightIntegrator, SetupGeometry


@dataclass
class SyntheticSurface:
    """Ground truth geometric and radiometric surface properties."""

    height_map: np.ndarray  # Ground truth 3D depth/height z(x, y)
    albedo_map: np.ndarray  # Ground truth diffuse albedo rho(x, y)
    normals: np.ndarray  # (H, W, 3) surface unit normals
    is_relief: bool  # True if true 3D surface, False if 2D flat photo/replay


class ProceduralSurfaceGenerator:
    """Generates synthetic test targets for benchmark validation."""

    @staticmethod
    def generate_coin(size: int = 512) -> SyntheticSurface:
        """Create embossed coin with rim, portrait relief, and textured albedo."""
        y, x = np.meshgrid(np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32), indexing='ij')
        cx, cy = size / 2.0, size / 2.0
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        radius = size * 0.42

        # 3D Height geometry
        z = np.zeros((size, size), dtype=np.float32)
        coin_mask = r <= radius

        # Raised outer rim
        rim_mask = (r >= radius * 0.88) & coin_mask
        z[rim_mask] = 0.8

        # Inner embossed central head / profile relief
        inner_mask = r < radius * 0.88
        # Central smooth dome + micro features
        z[inner_mask] = 0.5 * np.cos((np.pi / 2.0) * (r[inner_mask] / (radius * 0.88)))

        # Embossed portrait contour
        profile_x = np.sin((x[inner_mask] - cx) * 0.08) * np.cos((y[inner_mask] - cy) * 0.08)
        z[inner_mask] += 0.3 * np.maximum(profile_x, 0.0)

        # Albedo map (metallic bronze/gold texture)
        albedo = np.full((size, size, 3), 0.35, dtype=np.float32)
        albedo[coin_mask] = [0.80, 0.65, 0.30]  # Bronze/gold
        # Add slight texture
        noise_tex = np.random.normal(1.0, 0.05, (size, size, 1)).astype(np.float32)
        albedo = np.clip(albedo * noise_tex, 0.0, 1.0)

        # Compute ground truth surface normals pointing UP from desk [Nx, Ny, Nz]
        # X: left-right, Y: up from desk, Z: along desk towards viewer
        dz_dx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3) * 0.25
        dz_dy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3) * 0.25
        norm_mag = np.sqrt(dz_dx**2 + dz_dy**2 + 1.0)
        normals = np.stack([-dz_dx / norm_mag, 1.0 / norm_mag, -dz_dy / norm_mag], axis=-1).astype(np.float32)

        return SyntheticSurface(height_map=z, albedo_map=albedo, normals=normals, is_relief=True)

    @staticmethod
    def generate_flat_print(size: int = 512) -> SyntheticSurface:
        """Create a 2D flat printed photograph of a coin (spoof copy).

        Has identical visual albedo/texture as the coin, but absolute zero 3D height!
        """
        coin = ProceduralSurfaceGenerator.generate_coin(size)
        # Flatten height to zero: perfectly planar
        flat_z = np.zeros((size, size), dtype=np.float32)
        # Normal is purely [0, 1, 0] (pointing straight up) everywhere
        flat_normals = np.zeros((size, size, 3), dtype=np.float32)
        flat_normals[:, :, 1] = 1.0

        return SyntheticSurface(
            height_map=flat_z,
            albedo_map=coin.albedo_map,
            normals=flat_normals,
            is_relief=False,
        )

    @staticmethod
    def generate_braille(size: int = 512) -> SyntheticSurface:
        """Create tactile Braille dot array on paper."""
        z = np.zeros((size, size), dtype=np.float32)
        albedo = np.full((size, size, 3), 0.90, dtype=np.float32)  # White card

        # Grid of Braille hemispherical dots
        spacing = size // 6
        dot_r = 16.0
        y_grid, x_grid = np.meshgrid(np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32), indexing='ij')

        for r_idx in range(1, 6):
            for c_idx in range(1, 6):
                if (r_idx + c_idx) % 2 == 0:  # Patterned dots
                    dot_cx = c_idx * spacing
                    dot_cy = r_idx * spacing
                    dist = np.sqrt((x_grid - dot_cx) ** 2 + (y_grid - dot_cy) ** 2)
                    dot_mask = dist <= dot_r
                    # Hemispherical dome
                    z[dot_mask] += np.sqrt(np.maximum(dot_r**2 - dist[dot_mask] ** 2, 0.0)) / dot_r

        dz_dx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3) * 0.35
        dz_dy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3) * 0.35
        norm_mag = np.sqrt(dz_dx**2 + dz_dy**2 + 1.0)
        normals = np.stack([-dz_dx / norm_mag, 1.0 / norm_mag, -dz_dy / norm_mag], axis=-1).astype(np.float32)

        return SyntheticSurface(height_map=z, albedo_map=albedo, normals=normals, is_relief=True)


class PhotometricForwardRenderer:
    """Raytraces surface irradiance under screen pattern lighting."""

    def __init__(self, geom: SetupGeometry = SetupGeometry()):
        self.integrator = ScreenLightIntegrator(geom)

    def render_frame(
        self,
        surface: SyntheticSurface,
        pattern_img: np.ndarray,
        ambient_level: float = 0.05,
        noise_std: float = 0.008,
    ) -> np.ndarray:
        """Render realistic camera capture for given surface and screen pattern."""
        h, w = surface.normals.shape[:2]

        # Sample pattern to screen emitter grid
        pattern_weights = self.integrator.sample_pattern_to_grid(pattern_img)

        # Center surface point on desk
        surface_pt = np.array([0.0, 0.0, self.integrator.geom.object_distance_z], dtype=np.float32)

        total_power = float(np.sum(pattern_weights))
        if total_power < 1e-5:
            # Screen is off (DARK frame): zero diffuse reflection, only ambient
            diffuse = np.zeros_like(surface.albedo_map)
        else:
            l_vec = self.integrator.compute_lighting_vector(pattern_weights, surface_pt)
            # Lambertian cosine shading: max(N . L, 0)
            n_dot_l = np.clip(np.sum(surface.normals * l_vec, axis=-1, keepdims=True), 0.0, 1.0)
            diffuse = surface.albedo_map * n_dot_l

        # Add ambient background illumination
        irradiance = diffuse + ambient_level

        # Add camera sensor Gaussian shot + read noise
        noise = np.random.normal(0.0, noise_std, irradiance.shape).astype(np.float32)
        noisy_linear = np.clip(irradiance + noise, 0.0, 1.0)

        # Convert back to sRGB display representation for webcam emulation
        srgb = np.power(noisy_linear, 1.0 / 2.2)
        return (np.clip(srgb * 255.0, 0, 255)).astype(np.uint8)
