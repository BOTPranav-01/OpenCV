"""Model Context Protocol (MCP) tool server for Agentic Vision integration."""

from dataclasses import asdict
from lumiprobe.agent.engine import LumiProbeEngine


class LumiProbeMCPServer:
    """Exposes LumiProbe-X active photometric stereo tools to external LLM agents."""

    def __init__(self, engine: LumiProbeEngine | None = None):
        self.engine = engine or LumiProbeEngine()

    def reset_inspection(self, target_type: str = "coin", mode: str = "surface") -> dict:
        """Reset inspection state and initialize target."""
        self.engine.reset_session(target_type=target_type, mode=mode)
        return {"status": "success", "mode": mode, "target": target_type}

    def execute_step(self) -> dict:
        """Run one active perception cycle (capture, radiometric solve, SPRT, and next-light planning)."""
        res = self.engine.step()
        return {
            "verdict": res.verdict,
            "confidence": res.confidence,
            "cumulative_llr": res.cumulative_llr,
            "uncertainty": res.uncertainty,
            "current_pattern": res.pattern_id,
            "reason": res.reason,
            "step_count": res.step_count,
            "has_3d_mesh": res.height_map is not None,
            "defects_found": res.defect_summary.defect_count if res.defect_summary else 0,
        }

    def get_verdict(self) -> dict:
        """Retrieve current SPRT decision state."""
        sprt = self.engine.session.latest_sprt
        if sprt is None:
            return {"verdict": "NOT_STARTED", "confidence": 0.0, "llr": 0.0}
        return asdict(sprt)

    def export_obj_mesh(self) -> str:
        """Export latest reconstructed 3D surface height map as Wavefront OBJ string."""
        if self.engine.latest_height_map is None:
            return "# No 3D reconstruction available"
        return self.engine.integrator.height_to_obj_mesh(self.engine.latest_height_map)
