"""Extended unit tests for LumiProbe-X: MCP server, cloud client, pattern bank, and 3D export."""

import sys
import numpy as np
import pytest

sys.path.insert(0, "src")

from lumiprobe.agent.engine import LumiProbeEngine
from lumiprobe.cloud.client import CloudInspectionClient
from lumiprobe.mcp.server import LumiProbeMCPServer
from lumiprobe.physics.patterns import PatternBank
from lumiprobe.vision.integrate import FrankotChellappaIntegrator


def test_pattern_bank_dimensions_and_channels():
    """Verify all PatternBank illumination generators create valid normalized tensors."""
    w, h = 640, 360

    dark = PatternBank.make_dark(w, h)
    assert dark.shape == (h, w, 3)
    assert np.all(dark == 0.0)

    white = PatternBank.make_white(w, h)
    assert white.shape == (h, w, 3)
    assert np.all(white == 1.0)

    grad_lr = PatternBank.make_gradient_lr(w, h)
    assert grad_lr.shape == (h, w, 3)
    assert grad_lr[0, 0, 0] == 0.0
    assert grad_lr[0, -1, 0] == 1.0

    bar_45 = PatternBank.make_bar(45.0, width_px=100, width=w, height=h)
    assert bar_45.shape == (h, w, 3)
    assert np.max(bar_45) == 1.0

    spot = PatternBank.make_spot(0.5, 0.5, radius_ratio=0.2, width=w, height=h)
    assert spot.shape == (h, w, 3)
    assert spot[h // 2, w // 2, 0] == 1.0


def test_gamma_correction():
    """Verify display gamma encoding converts linear light to sRGB."""
    linear = np.array([0.0, 0.218, 0.5, 1.0], dtype=np.float32)
    gamma_encoded = PatternBank.gamma_correct(linear, gamma=2.2)

    assert gamma_encoded[0] == 0.0
    assert pytest.approx(gamma_encoded[3], abs=1e-5) == 1.0
    assert gamma_encoded[1] > linear[1]  # Gamma expands low values for display output


def test_mcp_server_interface():
    """Verify Model Context Protocol (MCP) tool server functions."""
    engine = LumiProbeEngine(force_sim=True, sim_target="coin")
    mcp = LumiProbeMCPServer(engine=engine)

    # 1. Reset
    reset_res = mcp.reset_inspection(target_type="coin", mode="surface")
    assert reset_res["status"] == "success"
    assert reset_res["target"] == "coin"

    # 2. Step
    step_res = mcp.execute_step()
    assert "verdict" in step_res
    assert "current_pattern" in step_res
    assert step_res["step_count"] == 1

    # 3. Verdict query
    verdict_res = mcp.get_verdict()
    assert "verdict" in verdict_res


def test_cloud_inspection_client_offline_fallback():
    """Verify that cloud inspection client gracefully uses local fallback without network."""
    client = CloudInspectionClient()
    # Force cloud_enabled to False for deterministic test
    client.cloud_enabled = False

    record = client.sync_inspection(
        target_type="coin",
        verdict="RELIEF",
        confidence=0.992,
        llr=5.12,
        patterns_used=["DARK", "WHITE", "GRAD_LR", "GRAD_RL", "GRAD_TB", "GRAD_BT"],
        height_map=np.zeros((64, 64), dtype=np.float32),
        defect_count=0,
    )

    assert record["verdict"] == "RELIEF"
    assert "OFFLINE" in record["cloud_status"]
    assert record["inspection_id"] is not None


def test_wavefront_obj_export():
    """Verify 3D Wavefront OBJ mesh serialization."""
    integrator = FrankotChellappaIntegrator()
    # Create small synthetic dome heightmap (32x32)
    y, x = np.mgrid[-1:1:32j, -1:1:32j]
    h_map = np.maximum(1.0 - (x**2 + y**2), 0.0).astype(np.float32)

    obj_str = integrator.height_to_obj_mesh(h_map, step=2)

    assert obj_str.startswith("# LumiProbe")
    assert "v " in obj_str  # Has vertices
    assert "f " in obj_str  # Has triangular faces
