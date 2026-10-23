"""Wald's Sequential Probability Ratio Test (SPRT) for active hypothesis testing.

Decides between:
  H0 (FLAT): Surface is a planar 2D photo, photocopy, or screen replay (residual ~ noise).
  H1 (RELIEF): Surface has true physical 3D depth, intaglio ink, or flesh relief.
Terminates inspection as soon as statistical evidence reaches desired error rates (alpha, beta).
"""

from dataclasses import dataclass
from enum import Enum
import numpy as np


class Verdict(str, Enum):
    RELIEF = "RELIEF"
    FLAT = "FLAT"
    CONTINUE = "CONTINUE"
    INCONCLUSIVE = "INCONCLUSIVE"
    UNRELIABLE = "UNRELIABLE"


@dataclass
class SPRTState:
    """Current cumulative SPRT decision state."""

    log_likelihood_ratio: float  # Cumulative Lambda
    upper_bound: float  # B1 = ln((1 - beta) / alpha)
    lower_bound: float  # B0 = ln(beta / (1 - alpha))
    verdict: Verdict
    confidence: float
    samples_count: int
    uncertainty_score: float  # Metric in [0, 1] indicating distance to threshold


class WaldSPRT:
    """Sequential Probability Ratio Test evaluator with calibrated error thresholds."""

    def __init__(self, alpha: float = 0.01, beta: float = 0.01, sigma_relief: float = 0.015):
        self.alpha = alpha
        self.beta = beta
        self.sigma_relief_sq = sigma_relief**2

        # Wald stopping thresholds
        self.b1 = np.log((1.0 - beta) / alpha)  # RELIEF boundary (> 0)
        self.b0 = np.log(beta / (1.0 - alpha))  # FLAT boundary (< 0)

    def compute_patch_llr(self, residual_energy: float, noise_var: float) -> float:
        """Compute log-likelihood ratio for a single spatial patch.

        H0: sigma_0^2 = noise_var
        H1: sigma_1^2 = noise_var + sigma_relief_sq
        """
        s0_sq = max(noise_var, 1e-6)
        s1_sq = s0_sq + self.sigma_relief_sq

        # Gaussian log-likelihood ratio:
        # 0.5 * [ ln(s0^2 / s1^2) + e * (1/s0^2 - 1/s1^2) ]
        llr = 0.5 * (np.log(s0_sq / s1_sq) + residual_energy * (1.0 / s0_sq - 1.0 / s1_sq))
        return float(llr)

    def evaluate(self, cumulative_llr: float, step: int, max_steps: int = 12) -> SPRTState:
        """Evaluate current cumulative LLR against Wald stopping boundaries."""
        if cumulative_llr >= self.b1:
            # Reached upper bound: 3D Relief confirmed
            conf = 1.0 - self.alpha
            verdict = Verdict.RELIEF
            unc = 0.0
        elif cumulative_llr <= self.b0:
            # Reached lower bound: Flat 2D print confirmed
            conf = 1.0 - self.beta
            verdict = Verdict.FLAT
            unc = 0.0
        else:
            # In continuation region
            if step >= max_steps:
                verdict = Verdict.INCONCLUSIVE
                conf = 0.5
                unc = 1.0
            else:
                verdict = Verdict.CONTINUE
                # Calculate normalized distance to nearest boundary
                dist = min(abs(self.b1 - cumulative_llr), abs(self.b0 - cumulative_llr))
                unc = float(np.clip(dist / max(self.b1, -self.b0), 0.0, 1.0))
                conf = 1.0 - unc

        return SPRTState(
            log_likelihood_ratio=float(cumulative_llr),
            upper_bound=float(self.b1),
            lower_bound=float(self.b0),
            verdict=verdict,
            confidence=float(conf),
            samples_count=step,
            uncertainty_score=float(unc),
        )
