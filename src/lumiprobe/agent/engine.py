"""Central LumiProbe-X Inspection Engine orchestrating active vision loop."""

from dataclasses import dataclass
import time
import cv2
import numpy as np
from lumiprobe.agent.planner import ActiveIlluminationPlanner
from lumiprobe.agent.sprt import SPRTState, Verdict, WaldSPRT
from lumiprobe.agent.state import InspectionSessionState
from lumiprobe.capture.camera import CameraDevice
from lumiprobe.physics.patterns import PatternBank
from lumiprobe.vision.aruco_window import ArucoWindowTracker
from lumiprobe.vision.defects import DefectDetector, DefectSummary
from lumiprobe.vision.face_roi import FaceRoiExtractor
from lumiprobe.vision.integrate import FrankotChellappaIntegrator
from lumiprobe.vision.linearize import RadiometricLinearizer
from lumiprobe.vision.residual import NormalFieldResult, SurfaceResidualEstimator


@dataclass
class InspectionEngineResult:
    """Consolidated metrics and imagery for current inspection frame."""

    verdict: str
    confidence: float
    cumulative_llr: float
    uncertainty: float
    pattern_id: str
    reason: str
    step_count: int
    is_simulated: bool
    target_type: str
    raw_frame: np.ndarray
    rectified_roi: np.ndarray | None
    height_map: np.ndarray | None
    defect_summary: DefectSummary | None
    uncertainty_map: list[list[float]]


class LumiProbeEngine:
    """Autonomous active photometric stereo inspection engine."""

    def __init__(self, force_sim: bool = False, sim_target: str = "coin", mode: str = "surface"):
        self.mode = mode
        self.camera = CameraDevice(force_sim=force_sim, sim_target=sim_target)
        self.window_tracker = ArucoWindowTracker()
        self.face_extractor = FaceRoiExtractor()
        self.linearizer = RadiometricLinearizer()
        self.residual_estimator = SurfaceResidualEstimator()
        self.integrator = FrankotChellappaIntegrator()
        self.defect_detector = DefectDetector()
        self.planner = ActiveIlluminationPlanner()
        self.session = InspectionSessionState(mode=mode)

        self.current_pattern = "DARK"
        self.last_planned_reason = "System initialized. Beginning dark-frame baseline calibration."
        self.latest_height_map: np.ndarray | None = None
        self.latest_normals: np.ndarray | None = None
        self.latest_defects: DefectSummary | None = None

        # Apply initial pattern
        self.camera.set_pattern(self.current_pattern)

    def reset_session(self, target_type: str | None = None, mode: str | None = None):
        """Reset inspection state for a new target."""
        if mode is not None:
            self.mode = mode
            self.session.mode = mode
        if target_type is not None:
            self.camera.set_sim_target(target_type)

        self.session.reset()
        self.latest_height_map = None
        self.latest_normals = None
        self.latest_defects = None
        self.current_pattern = "DARK"
        self.last_planned_reason = "Session reset. Capturing dark ambient baseline."
        self.camera.set_pattern(self.current_pattern)

    def step(self) -> InspectionEngineResult:
        """Execute one complete active perception cycle."""
        # 1. Grab camera frame under current lighting pattern
        ret, frame = self.camera.read()
        if not ret or frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # 2. Extract and rectify inspection ROI
        roi: np.ndarray | None = None
        if self.mode == "face":
            face_res = self.face_extractor.detect(frame)
            roi = face_res.face_roi
        else:
            win_res = self.window_tracker.detect(frame)
            if win_res.detected and win_res.homography is not None:
                roi = self.window_tracker.warp_roi(frame, win_res.homography)
            else:
                # If window markers not yet locked, use center frame
                h, w = frame.shape[:2]
                s = min(h, w) // 2
                cy, cx = h // 2, w // 2
                crop = frame[cy - s // 2 : cy + s // 2, cx - s // 2 : cx + s // 2]
                roi = cv2.resize(crop, (512, 512))

        # 3. Store frame in session buffer
        self.session.pattern_frames[self.current_pattern] = roi

        # 4. Radiometric processing
        linear_rgb = self.linearizer.to_linear(roi)
        luminance = self.linearizer.to_luminance(linear_rgb)

        if self.current_pattern == "DARK":
            self.session.dark_frame = luminance.copy()

        # Ambient subtraction: J = max(I - D, 0)
        dark = self.session.dark_frame if self.session.dark_frame is not None else np.zeros_like(luminance)
        ambient_sub = np.maximum(luminance - dark, 0.0)

        # 5. Check if we have the baseline gradient quartet
        has_grads = all(k in self.session.pattern_frames for k in ["GRAD_LR", "GRAD_RL", "GRAD_TB", "GRAD_BT"])
        if has_grads and self.latest_normals is None:
            d = self.session.dark_frame if self.session.dark_frame is not None else 0.0
            j_lr = np.maximum(self.linearizer.to_luminance(self.linearizer.to_linear(self.session.pattern_frames["GRAD_LR"])) - d, 0.0)
            j_rl = np.maximum(self.linearizer.to_luminance(self.linearizer.to_linear(self.session.pattern_frames["GRAD_RL"])) - d, 0.0)
            j_tb = np.maximum(self.linearizer.to_luminance(self.linearizer.to_linear(self.session.pattern_frames["GRAD_TB"])) - d, 0.0)
            j_bt = np.maximum(self.linearizer.to_luminance(self.linearizer.to_linear(self.session.pattern_frames["GRAD_BT"])) - d, 0.0)

            # Estimate normal field and residual
            norm_res = self.residual_estimator.estimate_from_gradients(j_lr, j_rl, j_tb, j_bt)
            self.latest_normals = norm_res.normals

            # Frankot-Chellappa 3D surface height integration
            self.latest_height_map = self.integrator.integrate(norm_res.residual_x, norm_res.residual_y, scale_factor=3.5)

            # Detect micro defects
            self.latest_defects = self.defect_detector.detect_anomalies(self.latest_height_map)

        # 6. Compute patch residual energy for SPRT
        # Calibration & basis buildup patterns record step without LLR evidence
        if self.current_pattern in ("DARK", "WHITE", "GRAD_LR", "GRAD_RL", "GRAD_TB"):
            sprt_state = self.session.record_calibration_step(
                pattern_id=self.current_pattern,
                reason=self.last_planned_reason,
                timestamp=time.time(),
            )
        else:
            ny, nx = self.session.grid_dim
            ph, pw = ambient_sub.shape[0] // ny, ambient_sub.shape[1] // nx
            patch_energies = np.zeros((ny, nx), dtype=np.float32)
            patch_noises = np.full((ny, nx), 0.00120, dtype=np.float32)

            if has_grads and self.latest_normals is not None:
                d = self.session.dark_frame if self.session.dark_frame is not None else 0.0
                j_lr = np.maximum(self.linearizer.to_luminance(self.linearizer.to_linear(self.session.pattern_frames["GRAD_LR"])) - d, 0.0)
                j_rl = np.maximum(self.linearizer.to_luminance(self.linearizer.to_linear(self.session.pattern_frames["GRAD_RL"])) - d, 0.0)
                j_tb = np.maximum(self.linearizer.to_luminance(self.linearizer.to_linear(self.session.pattern_frames["GRAD_TB"])) - d, 0.0)
                j_bt = np.maximum(self.linearizer.to_luminance(self.linearizer.to_linear(self.session.pattern_frames["GRAD_BT"])) - d, 0.0)
                norm_res = self.residual_estimator.estimate_from_gradients(j_lr, j_rl, j_tb, j_bt)
                res_sq = norm_res.residual_x**2 + norm_res.residual_y**2

                for r in range(ny):
                    for c in range(nx):
                        patch = res_sq[r * ph : (r + 1) * ph, c * pw : (c + 1) * pw]
                        patch_energies[r, c] = float(np.mean(patch))
            else:
                patch_energies.fill(0.00100)

            # Update SPRT belief state
            sprt_state = self.session.update_with_measurement(
                pattern_id=self.current_pattern,
                reason=self.last_planned_reason,
                patch_energies=patch_energies,
                patch_noises=patch_noises,
                timestamp=time.time(),
            )

        # 8. Plan the Next Action
        next_action = self.planner.plan_next_step(self.session, roi)
        self.last_planned_reason = next_action.reason

        if next_action.action_type == "CHANGE_PATTERN":
            self.current_pattern = next_action.pattern_id
            self.camera.set_pattern(self.current_pattern)

        verdict_str = sprt_state.verdict.value if sprt_state else "ANALYZING"
        conf = sprt_state.confidence if sprt_state else 0.0
        llr = sprt_state.log_likelihood_ratio if sprt_state else 0.0
        unc = sprt_state.uncertainty_score if sprt_state else 1.0

        u_map_list = self.session.uncertainty_map.tolist()

        return InspectionEngineResult(
            verdict=verdict_str,
            confidence=conf,
            cumulative_llr=llr,
            uncertainty=unc,
            pattern_id=self.current_pattern,
            reason=self.last_planned_reason,
            step_count=len(self.session.steps),
            is_simulated=self.camera.is_simulated,
            target_type=self.camera.sim_target,
            raw_frame=frame,
            rectified_roi=roi,
            height_map=self.latest_height_map,
            defect_summary=self.latest_defects,
            uncertainty_map=u_map_list,
        )
