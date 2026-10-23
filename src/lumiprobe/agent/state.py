"""Inspection state, spatial patch grid, and uncertainty map tracking."""

from dataclasses import dataclass, field
import numpy as np
from lumiprobe.agent.sprt import SPRTState, Verdict, WaldSPRT


@dataclass
class InspectionStep:
    """Record of a single active lighting step."""

    step_number: int
    pattern_id: str
    reason: str
    llr_delta: float
    cumulative_llr: float
    uncertainty: float
    timestamp: float


@dataclass
class InspectionSessionState:
    """Full operational state of an active inspection run."""

    mode: str = "surface"  # 'surface' or 'face'
    grid_dim: tuple[int, int] = (16, 16)  # 16x16 patch resolution
    sprt: WaldSPRT = field(default_factory=WaldSPRT)

    # Historical timeline
    steps: list[InspectionStep] = field(default_factory=list)
    patterns_used: list[str] = field(default_factory=list)

    # Cumulative evidence
    cumulative_llr: float = 0.0
    latest_sprt: SPRTState | None = None
    uncertainty_map: np.ndarray = field(default_factory=lambda: np.ones((16, 16), dtype=np.float32))
    patch_llr_map: np.ndarray = field(default_factory=lambda: np.zeros((16, 16), dtype=np.float32))

    # Stored image buffers for multi-illumination reconstruction
    pattern_frames: dict[str, np.ndarray] = field(default_factory=dict)
    dark_frame: np.ndarray | None = None

    def reset(self):
        """Reset session for a new inspection."""
        self.steps.clear()
        self.patterns_used.clear()
        self.cumulative_llr = 0.0
        self.latest_sprt = None
        self.uncertainty_map = np.ones(self.grid_dim, dtype=np.float32)
        self.patch_llr_map = np.zeros(self.grid_dim, dtype=np.float32)
        self.pattern_frames.clear()
        self.dark_frame = None

    def record_calibration_step(self, pattern_id: str, reason: str, timestamp: float) -> SPRTState:
        """Record calibration step (e.g. DARK or WHITE) without accumulating LLR evidence."""
        self.patterns_used.append(pattern_id)
        step_idx = len(self.steps) + 1
        sprt_state = self.sprt.evaluate(self.cumulative_llr, step_idx)
        self.latest_sprt = sprt_state
        self.steps.append(
            InspectionStep(
                step_number=step_idx,
                pattern_id=pattern_id,
                reason=reason,
                llr_delta=0.0,
                cumulative_llr=float(self.cumulative_llr),
                uncertainty=float(sprt_state.uncertainty_score),
                timestamp=timestamp,
            )
        )
        return sprt_state

    def update_with_measurement(
        self,
        pattern_id: str,
        reason: str,
        patch_energies: np.ndarray,
        patch_noises: np.ndarray,
        timestamp: float,
    ) -> SPRTState:
        """Incorporate new frame evidence across spatial grid patches."""
        ny, nx = self.grid_dim
        step_llr = 0.0

        for r in range(ny):
            for c in range(nx):
                e = float(patch_energies[r, c])
                n = float(patch_noises[r, c])
                llr_patch = self.sprt.compute_patch_llr(e, n)
                self.patch_llr_map[r, c] += llr_patch
                step_llr += llr_patch

                # Uncertainty in patch decreases as absolute LLR grows
                self.uncertainty_map[r, c] = float(np.exp(-0.25 * abs(self.patch_llr_map[r, c])))

        # Accumulate SPRT evidence over salient surface patches (top 32)
        flat_energies = patch_energies.ravel()
        flat_noises = patch_noises.ravel()
        sort_idx = np.argsort(flat_energies)[::-1]
        active_idx = sort_idx[:32]

        for idx in active_idx:
            e = float(flat_energies[idx])
            n = float(flat_noises[idx])
            llr_patch = self.sprt.compute_patch_llr(e, n)
            step_llr += llr_patch

        avg_step_llr = (step_llr / 32.0) * 4.0
        self.cumulative_llr += avg_step_llr

        step_idx = len(self.steps) + 1
        sprt_state = self.sprt.evaluate(self.cumulative_llr, step_idx)
        self.latest_sprt = sprt_state

        self.patterns_used.append(pattern_id)
        self.steps.append(
            InspectionStep(
                step_number=step_idx,
                pattern_id=pattern_id,
                reason=reason,
                llr_delta=float(avg_step_llr),
                cumulative_llr=float(self.cumulative_llr),
                uncertainty=float(sprt_state.uncertainty_score),
                timestamp=timestamp,
            )
        )

        return sprt_state
