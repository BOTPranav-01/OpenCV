"""Comprehensive test suite for LumiProbe-X vision math, SPRT, and active engine."""

import sys
import cv2
import numpy as np
import pytest

sys.path.insert(0, "src")

from lumiprobe.agent.engine import LumiProbeEngine
from lumiprobe.agent.sprt import Verdict, WaldSPRT
from lumiprobe.physics.patterns import PatternBank
from lumiprobe.sim.render import ProceduralSurfaceGenerator
from lumiprobe.vision.aruco_window import ArucoWindowTracker
from lumiprobe.vision.defects import DefectDetector
from lumiprobe.vision.integrate import FrankotChellappaIntegrator
from lumiprobe.vision.linearize import RadiometricLinearizer


def test_opencv5_version():
    """Assert OpenCV version 5.x requirement."""
    assert cv2.__version__.startswith("5"), f"Expected OpenCV 5.x, got {cv2.__version__}"


def test_radiometric_linearization():
    """Verify IEC 61966-2-1 inverse sRGB transfer curve."""
    linearizer = RadiometricLinearizer()

    # Zero black remains zero
    assert linearizer.lut[0] == 0.0
    # 255 peak remains 1.0
    assert pytest.approx(linearizer.lut[255], abs=1e-5) == 1.0

    # sRGB mid-gray (128/255 = 0.5019) corresponds to linear ~0.2158
    assert 0.20 < linearizer.lut[128] < 0.23


def test_frankot_chellappa_reconstruction():
    """Verify Fourier-domain Poisson height integration on analytical dome."""
    size = 128
    y, x = np.mgrid[-1:1:128j, -1:1:128j]
    # Analytical spherical dome: z = max(1 - (x^2 + y^2), 0)
    z_true = np.maximum(1.0 - (x**2 + y**2), 0.0).astype(np.float32)

    # Analytical gradients
    dz_dx = np.where(x**2 + y**2 <= 1.0, -2.0 * x, 0.0).astype(np.float32)
    dz_dy = np.where(x**2 + y**2 <= 1.0, -2.0 * y, 0.0).astype(np.float32)

    z_recon = FrankotChellappaIntegrator.integrate(dz_dx, dz_dy, pad_width=8)

    # Center must be significantly higher than boundary
    assert z_recon[64, 64] > z_recon[10, 10]


def test_wald_sprt_hypothesis_bounds():
    """Test Wald SPRT statistical stopping boundaries."""
    sprt = WaldSPRT(alpha=0.01, beta=0.01)

    # Upper bound should be approx ln(99) = +4.595
    assert pytest.approx(sprt.b1, abs=0.05) == 4.595
    # Lower bound should be approx ln(0.01/0.99) = -4.595
    assert pytest.approx(sprt.b0, abs=0.05) == -4.595

    # Positive LLR should evaluate to RELIEF when exceeding bound
    state_relief = sprt.evaluate(5.0, step=6)
    assert state_relief.verdict == Verdict.RELIEF
    assert state_relief.confidence > 0.98

    # Negative LLR should evaluate to FLAT when below bound
    state_flat = sprt.evaluate(-5.0, step=8)
    assert state_flat.verdict == Verdict.FLAT
    assert state_flat.confidence > 0.98


def test_defect_detector_anomalies():
    """Verify robust z-score scratch detection."""
    detector = DefectDetector(z_threshold=2.5, min_area_px=10)
    h_map = np.zeros((128, 128), dtype=np.float32)

    # Insert a synthetic surface crack / scratch (3px wide groove of depth -1.5)
    h_map[40:70, 59:62] = -1.5

    summary = detector.detect_anomalies(h_map)
    assert summary.defect_count >= 1
    assert summary.total_defect_area_px >= 10
    assert summary.max_defect_length_px >= 20.0


def test_end_to_end_engine_simulation():
    """Verify autonomous active inspection loop on simulated targets."""
    # 1. Test 3D relief coin target
    engine_coin = LumiProbeEngine(force_sim=True, sim_target="coin")
    coin_verdict = None
    for _ in range(12):
        res = engine_coin.step()
        if res.verdict in ("RELIEF", "FLAT"):
            coin_verdict = res.verdict
            break

    assert coin_verdict == "RELIEF", f"Expected RELIEF on coin, got {coin_verdict}"
    assert engine_coin.latest_height_map is not None
    assert engine_coin.latest_normals is not None

    # 2. Test 2D flat counterfeit spoof target
    engine_flat = LumiProbeEngine(force_sim=True, sim_target="flat_print")
    flat_verdict = None
    for _ in range(12):
        res = engine_flat.step()
        if res.verdict in ("RELIEF", "FLAT"):
            flat_verdict = res.verdict
            break

    assert flat_verdict == "FLAT", f"Expected FLAT on photocopy, got {flat_verdict}"
