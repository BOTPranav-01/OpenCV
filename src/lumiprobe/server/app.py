"""FastAPI server providing real-time WebSocket telemetry, REST API, and static UI."""

import asyncio
import base64
import json
import os
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from lumiprobe.agent.engine import LumiProbeEngine
from lumiprobe.physics.patterns import PatternBank

app = FastAPI(title="LumiProbe-X: Active Photometric Reality Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate central engine
engine = LumiProbeEngine(force_sim=False, sim_target="coin", mode="surface")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)


class ActionRequest(BaseModel):
    action: str  # 'step', 'reset', 'run', 'set_target', 'set_mode'
    param: str | None = None


def encode_image_base64(img: np.ndarray, quality: int = 80) -> str:
    """Encode OpenCV BGR or float image to JPEG base64 string."""
    if img is None:
        return ""
    if img.dtype != np.uint8:
        # Normalize float to uint8
        norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img_u8 = norm.astype(np.uint8)
    else:
        img_u8 = img

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, buffer = cv2.imencode(".jpg", img_u8, encode_param)
    if not success:
        return ""
    return base64.b64encode(buffer).decode("utf-8")


@app.get("/")
async def get_dashboard():
    """Serve mission control dashboard."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>LumiProbe-X Dashboard Loading...</h1>")


@app.get("/lamp")
async def get_lamp():
    """Serve fullscreen programmable illumination lamp."""
    lamp_path = os.path.join(STATIC_DIR, "lamp.html")
    if os.path.exists(lamp_path):
        return FileResponse(lamp_path)
    return HTMLResponse("<h1>LumiProbe Lamp</h1>")


@app.get("/api/pattern/{pattern_id}")
async def get_pattern_image(pattern_id: str):
    """Render and return pattern as JPEG."""
    pat = PatternBank.get_pattern(pattern_id, 1280, 720)
    pat_u8 = (PatternBank.gamma_correct(pat) * 255.0).astype(np.uint8)
    _, buffer = cv2.imencode(".jpg", pat_u8)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@app.get("/api/export/obj")
async def export_obj():
    """Export reconstructed 3D surface mesh as Wavefront OBJ."""
    if engine.latest_height_map is not None:
        obj_data = engine.integrator.height_to_obj_mesh(engine.latest_height_map, step=2)
        return Response(
            content=obj_data,
            media_type="text/plain",
            headers={"Content-Disposition": "attachment; filename=lumiprobe_3d_relief.obj"},
        )
    return Response(content="# No 3D reconstruction yet available\n", media_type="text/plain")


@app.post("/api/action")
async def execute_action(req: ActionRequest):
    """Handle UI command actions."""
    if req.action == "reset":
        engine.reset_session()
        return {"status": "ok", "message": "Session reset"}
    if req.action == "set_target" and req.param:
        engine.reset_session(target_type=req.param)
        return {"status": "ok", "target": req.param}
    if req.action == "set_mode" and req.param:
        engine.reset_session(mode=req.param)
        return {"status": "ok", "mode": req.param}
    if req.action == "step":
        res = engine.step()
        return {
            "status": "ok",
            "verdict": res.verdict,
            "llr": res.cumulative_llr,
            "pattern": res.pattern_id,
            "reason": res.reason,
        }
    return {"status": "ignored"}


@app.websocket("/ws/stream")
async def websocket_telemetry(websocket: WebSocket):
    """High-speed telemetry stream pushing metrics, 3D heightmaps, and video."""
    await websocket.accept()
    try:
        while True:
            # Process one active perception step
            res = engine.step()

            # Encode imagery
            raw_b64 = encode_image_base64(res.raw_frame, quality=75)
            roi_b64 = encode_image_base64(res.rectified_roi, quality=80) if res.rectified_roi is not None else ""

            height_b64 = ""
            if res.height_map is not None:
                # Colorize height map for visualization
                h_norm = cv2.normalize(res.height_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                h_color = cv2.applyColorMap(h_norm, cv2.COLORMAP_VIRIDIS)
                height_b64 = encode_image_base64(h_color, quality=85)

            defect_b64 = ""
            defect_cnt = 0
            if res.defect_summary is not None:
                defect_b64 = encode_image_base64(res.defect_summary.heatmap_colored, quality=80)
                defect_cnt = res.defect_summary.defect_count

            # Downsample heightmap for Three.js 3D mesh rendering (e.g. 64x64 float array)
            mesh_grid = []
            if res.height_map is not None:
                small_h = cv2.resize(res.height_map, (64, 64), interpolation=cv2.INTER_AREA)
                mesh_grid = small_h.tolist()

            payload = {
                "verdict": res.verdict,
                "confidence": res.confidence,
                "cumulative_llr": res.cumulative_llr,
                "upper_bound": float(engine.session.sprt.b1),
                "lower_bound": float(engine.session.sprt.b0),
                "uncertainty": res.uncertainty,
                "pattern_id": res.pattern_id,
                "reason": res.reason,
                "step_count": res.step_count,
                "is_simulated": res.is_simulated,
                "target_type": res.target_type,
                "mode": engine.mode,
                "defect_count": defect_cnt,
                "raw_frame_b64": raw_b64,
                "roi_frame_b64": roi_b64,
                "height_frame_b64": height_b64,
                "defect_frame_b64": defect_b64,
                "uncertainty_map": res.uncertainty_map,
                "mesh_grid": mesh_grid,
            }

            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.12)  # ~8-10 Hz active probe cycle

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
