"""Safety pre-emption, motion validation, and environmental health checks."""

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class SafetyCheckResult:
    """Diagnostic state of image acquisition quality."""

    safe_to_proceed: bool
    saturation_fraction: float
    motion_shift_px: float
    focus_laplacian_var: float
    flicker_detected: bool
    action_override: str | None  # Safety action to execute if unsafe
    reason: str


class SafetyGuard:
    """Pre-empts active inspection planner when physical conditions degrade."""

    def __init__(
        self,
        max_saturation: float = 0.05,  # Max 5% saturated pixels
        max_motion_shift: float = 3.0,  # Shift threshold in pixels
        min_focus_var: float = 20.0,  # Minimum Laplacian variance for sharp focus
    ):
        self.max_saturation = max_saturation
        self.max_motion_shift = max_motion_shift
        self.min_focus_var = min_focus_var
        self.last_frame_gray: np.ndarray | None = None

    def check(self, frame: np.ndarray) -> SafetyCheckResult:
        """Analyze current frame for saturation, camera jitter, and optical defocus."""
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        h, w = gray.shape

        # 1. Saturation check
        sat_mask = gray >= 250
        sat_fraction = float(np.mean(sat_mask))
        if sat_fraction > self.max_saturation:
            return SafetyCheckResult(
                safe_to_proceed=False,
                saturation_fraction=sat_fraction,
                motion_shift_px=0.0,
                focus_laplacian_var=100.0,
                flicker_detected=False,
                action_override="REDUCE_BRIGHTNESS",
                reason=f"Surface specular saturation ({sat_fraction*100:.1f}% > 3%). Decreasing illumination intensity.",
            )

        # 2. Focus / Blur check
        lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        if lap_var < self.min_focus_var:
            return SafetyCheckResult(
                safe_to_proceed=False,
                saturation_fraction=sat_fraction,
                motion_shift_px=0.0,
                focus_laplacian_var=lap_var,
                flicker_detected=False,
                action_override="REQUEST_FOCUS",
                reason=f"Defocus / camera blur detected (Laplacian variance {lap_var:.1f} < {self.min_focus_var}). Hold steady.",
            )

        # 3. Motion shift check via cv2.phaseCorrelate
        motion_shift = 0.0
        if self.last_frame_gray is not None and self.last_frame_gray.shape == gray.shape:
            # Phase correlation on windowed float32 images
            f1 = self.last_frame_gray.astype(np.float32)
            f2 = gray.astype(np.float32)
            shift, _ = cv2.phaseCorrelate(f1, f2)
            motion_shift = float(np.sqrt(shift[0] ** 2 + shift[1] ** 2))

            if motion_shift > self.max_motion_shift:
                self.last_frame_gray = gray.copy()
                return SafetyCheckResult(
                    safe_to_proceed=False,
                    saturation_fraction=sat_fraction,
                    motion_shift_px=motion_shift,
                    focus_laplacian_var=lap_var,
                    flicker_detected=False,
                    action_override="DISCARD_AND_WAIT",
                    reason=f"Motion shift {motion_shift:.2f}px detected. Discarding jittered frame.",
                )

        self.last_frame_gray = gray.copy()

        # 4. Mains flicker check (FFT along row profile)
        row_profile = np.mean(gray, axis=1)
        row_fft = np.abs(np.fft.rfft(row_profile - np.mean(row_profile)))
        flicker_peak = float(np.max(row_fft[3:]) / (np.sum(row_fft[3:]) + 1e-6)) if len(row_fft) > 4 else 0.0
        flicker = flicker_peak > 0.45

        return SafetyCheckResult(
            safe_to_proceed=True,
            saturation_fraction=sat_fraction,
            motion_shift_px=motion_shift,
            focus_laplacian_var=lap_var,
            flicker_detected=flicker,
            action_override=None,
            reason="All physical quality indicators within nominal bounds.",
        )
