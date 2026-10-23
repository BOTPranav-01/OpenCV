"""Surface defect, scratch, and anomaly detection from 3D relief residuals."""

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class DefectSummary:
    """Quantitative defect metrics on inspected surface."""

    defect_count: int
    total_defect_area_px: int
    max_defect_length_px: float
    defect_mask: np.ndarray  # Binary mask (H, W) uint8
    heatmap_colored: np.ndarray  # RGB colored visualization


class DefectDetector:
    """Detects surface anomalies, scratches, cracks, and dents using spatial z-score filtering."""

    def __init__(self, z_threshold: float = 3.0, min_area_px: int = 15):
        self.z_threshold = z_threshold
        self.min_area_px = min_area_px

    def detect_anomalies(self, height_map: np.ndarray, valid_mask: np.ndarray | None = None) -> DefectSummary:
        """Segment high-gradient anomalies and measure defect statistics."""
        h, w = height_map.shape

        # Median filter to eliminate single-pixel noise
        med = cv2.medianBlur(height_map.astype(np.float32), 3)

        # High-pass filter (difference of Gaussians)
        low_freq = cv2.GaussianBlur(med, (15, 15), 3.0)
        high_freq = np.abs(med - low_freq)

        if valid_mask is not None:
            region = high_freq[valid_mask > 0]
        else:
            region = high_freq.ravel()

        if len(region) == 0:
            return DefectSummary(0, 0, 0.0, np.zeros((h, w), dtype=np.uint8), np.zeros((h, w, 3), dtype=np.uint8))

        med_val = np.median(region)
        # Robust estimate of standard deviation (MAD * 1.4826)
        mad = np.median(np.abs(region - med_val)) + 1e-6
        robust_std = mad * 1.4826

        z_scores = (high_freq - med_val) / robust_std

        # Threshold to binary defect mask
        raw_mask = (z_scores > self.z_threshold).astype(np.uint8) * 255
        if valid_mask is not None:
            raw_mask = cv2.bitwise_and(raw_mask, raw_mask, mask=valid_mask.astype(np.uint8))

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

        # Connected component analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)

        filtered_mask = np.zeros_like(clean_mask)
        count = 0
        total_area = 0
        max_length = 0.0

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_area_px:
                filtered_mask[labels == i] = 255
                count += 1
                total_area += area
                diag = np.sqrt(stats[i, cv2.CC_STAT_WIDTH] ** 2 + stats[i, cv2.CC_STAT_HEIGHT] ** 2)
                if diag > max_length:
                    max_length = float(diag)

        # Generate false-color visualization heatmap
        norm_z = np.clip(z_scores / (self.z_threshold * 2.0), 0.0, 1.0)
        heatmap_u8 = (norm_z * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)

        return DefectSummary(
            defect_count=count,
            total_defect_area_px=int(total_area),
            max_defect_length_px=max_length,
            defect_mask=filtered_mask,
            heatmap_colored=heatmap_color,
        )
