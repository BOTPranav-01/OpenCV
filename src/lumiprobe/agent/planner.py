"""Active perception planner: Next-Best-Light selector via information gain and uncertainty cooling."""

from dataclasses import dataclass
import numpy as np
from lumiprobe.agent.safety import SafetyCheckResult, SafetyGuard
from lumiprobe.agent.state import InspectionSessionState
from lumiprobe.physics.patterns import PatternBank


@dataclass
class PlannedAction:
    """Action decision produced by the active vision agent."""

    action_type: str  # 'CHANGE_PATTERN', 'STOP', 'REDUCE_BRIGHTNESS', 'HOLD_STEADY'
    pattern_id: str
    expected_gain: float
    reason: str


class ActiveIlluminationPlanner:
    """Autonomous runtime agent selecting dynamic light patterns based on visual evidence."""

    def __init__(self):
        self.safety = SafetyGuard()
        # Initial canonical gradient basis for albedo cancellation
        self.initial_sequence = ["DARK", "WHITE", "GRAD_LR", "GRAD_RL", "GRAD_TB", "GRAD_BT"]

    def plan_next_step(self, session: InspectionSessionState, current_frame: np.ndarray | None = None) -> PlannedAction:
        """Decide what light pattern to project next based on current evidence state."""
        # 1. Safety verification
        if current_frame is not None:
            safety_res = self.safety.check(current_frame)
            if not safety_res.safe_to_proceed:
                if safety_res.action_override == "REDUCE_BRIGHTNESS":
                    return PlannedAction(
                        action_type="ADJUST_BRIGHTNESS",
                        pattern_id=session.patterns_used[-1] if session.patterns_used else "WHITE",
                        expected_gain=0.0,
                        reason=safety_res.reason,
                    )
                if safety_res.action_override == "DISCARD_AND_WAIT":
                    return PlannedAction(
                        action_type="HOLD_STEADY",
                        pattern_id=session.patterns_used[-1] if session.patterns_used else "WHITE",
                        expected_gain=0.0,
                        reason=safety_res.reason,
                    )

        # 2. Check if SPRT has already reached stopping threshold
        if session.latest_sprt is not None and session.latest_sprt.verdict != "CONTINUE":
            return PlannedAction(
                action_type="STOP",
                pattern_id="DARK",
                expected_gain=0.0,
                reason=f"SPRT statistical threshold reached: {session.latest_sprt.verdict.value} (Confidence {session.latest_sprt.confidence*100:.1f}%).",
            )

        # 3. Baseline acquisition sequence
        for base_pat in self.initial_sequence:
            if base_pat not in session.patterns_used:
                reason = "Acquiring baseline gradient pair for radiometric albedo cancellation."
                if base_pat == "DARK":
                    reason = "Capturing dark ambient baseline to eliminate room light contamination."
                elif base_pat == "WHITE":
                    reason = "Capturing uniform white irradiance for surface albedo normalization."
                elif "GRAD" in base_pat:
                    reason = f"Projecting spatial linear gradient {base_pat} for differential normal estimation."

                return PlannedAction(
                    action_type="CHANGE_PATTERN",
                    pattern_id=base_pat,
                    expected_gain=1.0,
                    reason=reason,
                )

        # 4. Information-theoretic active pattern selection
        # Evaluate spatial uncertainty quadrants
        u_map = session.uncertainty_map  # (16, 16)
        h, w = u_map.shape
        top_u = float(np.mean(u_map[: h // 2, :]))
        bot_u = float(np.mean(u_map[h // 2 :, :]))
        left_u = float(np.mean(u_map[:, : w // 2]))
        right_u = float(np.mean(u_map[:, w // 2 :]))

        # Candidate active probes
        candidates = [
            ("BAR_0", "horizontal grazing bar to resolve vertical surface slope relief"),
            ("BAR_90", "vertical grazing bar to resolve horizontal micro-relief ridges"),
            ("BAR_45", "diagonal grazing bar to resolve 45-deg intaglio cross-hatching"),
            ("BAR_135", "oblique grazing bar for counter-diagonal surface relief"),
            ("SPOT_TL", "focused top-left oblique light for high-shadow relief excitation"),
            ("SPOT_TR", "focused top-right oblique light for micro-groove contrast"),
            ("SPOT_BL", "focused bottom-left directional illumination"),
            ("SPOT_BR", "focused bottom-right directional illumination"),
            ("HALF_L", "left half-plane step illumination for lateral shadow boundary"),
            ("HALF_R", "right half-plane step illumination for rightward shadow boundary"),
        ]

        best_pattern = None
        best_gain = -1.0
        best_reason = ""

        # Weight candidate patterns based on spatial uncertainty alignment
        for pid, desc in candidates:
            # Skip if already used recently
            if pid in session.patterns_used:
                continue

            gain = 0.5
            reason_detail = ""

            if "LEFT" in pid or "TL" in pid or "BL" in pid or pid == "HALF_L":
                if left_u > right_u:
                    gain += 0.4 * (left_u - right_u + 1.0)
                    reason_detail = f"Left quadrant uncertainty ({left_u:.2f} > {right_u:.2f}); directing light to maximize shadow contrast."
            elif "RIGHT" in pid or "TR" in pid or "BR" in pid or pid == "HALF_R":
                if right_u > left_u:
                    gain += 0.4 * (right_u - left_u + 1.0)
                    reason_detail = f"Right quadrant uncertainty ({right_u:.2f} > {left_u:.2f}); projecting directional beam across boundary."
            elif pid in ("BAR_0", "BAR_90", "BAR_45", "BAR_135"):
                gain += 0.3 * np.mean(u_map)
                reason_detail = f"High residual uncertainty ({np.mean(u_map):.2f}); applying {desc}."

            if gain > best_gain:
                best_gain = gain
                best_pattern = pid
                best_reason = reason_detail or f"Targeting uncertain spatial patches via {desc}."

        if best_pattern is None:
            # All candidates exhausted or max budget reached
            return PlannedAction(
                action_type="STOP",
                pattern_id="DARK",
                expected_gain=0.0,
                reason="Inspection budget complete. Sufficient evidence collected.",
            )

        return PlannedAction(
            action_type="CHANGE_PATTERN",
            pattern_id=best_pattern,
            expected_gain=float(best_gain),
            reason=best_reason,
        )
