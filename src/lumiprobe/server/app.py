"""FastAPI server providing real-time WebSocket telemetry, REST API, and static UI."""

import asyncio
import base64
import json
import logging
import os
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel

from lumiprobe.agent.engine import LumiProbeEngine
from lumiprobe.physics.patterns import PatternBank

logger = logging.getLogger("lumiprobe.server")

app = FastAPI(title="LumiProbe-X: Active Photometric Reality Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default to Physics Simulator on launch so it works instantly without external hardware
engine = LumiProbeEngine(force_sim=True, sim_target="coin", mode="surface")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# Active WebSocket subscribers
dashboard_sockets: set[WebSocket] = set()
lamp_sockets: set[WebSocket] = set()
step_lock = asyncio.Lock()


class ActionRequest(BaseModel):
    action: str  # 'step', 'reset', 'start', 'pause', 'set_target', 'set_source', 'set_mode'
    param: str | None = None


def encode_image_base64(img: np.ndarray, quality: int = 75) -> str:
    """Encode OpenCV BGR or float image to JPEG base64 string."""
    if img is None:
        return ""
    if img.dtype != np.uint8:
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


async def broadcast_lamp_pattern(pattern_id: str):
    """Notify all connected lamp windows of the new illumination pattern."""
    if not lamp_sockets:
        return
    msg = json.dumps({"pattern_id": pattern_id})
    dead = set()
    for ws in lamp_sockets:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    lamp_sockets.difference_update(dead)


@app.post("/api/action")
async def execute_action(req: ActionRequest):
    """Handle UI command actions."""
    async with step_lock:
        if req.action == "reset":
            engine.reset_session()
            await broadcast_lamp_pattern(engine.current_pattern)
            return {"status": "ok", "message": "Session reset", "is_sim": engine.camera.is_simulated}

        if req.action == "set_target" and req.param:
            engine.set_input_source("sim", req.param)
            await broadcast_lamp_pattern(engine.current_pattern)
            return {"status": "ok", "target": req.param, "source": "sim"}

        if req.action == "set_source" and req.param:
            engine.set_input_source(req.param, engine.camera.sim_target)
            await broadcast_lamp_pattern(engine.current_pattern)
            return {"status": "ok", "source": req.param, "is_sim": engine.camera.is_simulated}

        if req.action == "start":
            engine.is_active_scanning = True
            return {"status": "ok", "scanning": True}

        if req.action == "pause":
            engine.is_active_scanning = False
            return {"status": "ok", "scanning": False}

        if req.action == "step":
            engine.is_active_scanning = False
            res = engine.step()
            await broadcast_lamp_pattern(res.pattern_id)
            return {
                "status": "ok",
                "verdict": res.verdict,
                "llr": res.cumulative_llr,
                "pattern": res.pattern_id,
                "reason": res.reason,
            }

    return {"status": "ignored"}


@app.websocket("/ws/lamp")
async def websocket_lamp(websocket: WebSocket):
    """Ultra-low latency WebSocket stream exclusively for the fullscreen Lamp projector."""
    await websocket.accept()
    lamp_sockets.add(websocket)
    try:
        # Immediately sync current pattern
        await websocket.send_text(json.dumps({"pattern_id": engine.current_pattern}))
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        lamp_sockets.discard(websocket)


@app.websocket("/ws/stream")
async def websocket_telemetry(websocket: WebSocket):
    """High-speed telemetry stream pushing metrics, 3D heightmaps, and video."""
    await websocket.accept()
    dashboard_sockets.add(websocket)
    try:
        while True:
            async with step_lock:
                if engine.is_active_scanning:
                    res = engine.step()
                else:
                    res = engine.get_preview_result()

                # Sync lamp if pattern changed
                await broadcast_lamp_pattern(res.pattern_id)

                raw_b64 = encode_image_base64(res.raw_frame, quality=70)
                roi_b64 = encode_image_base64(res.rectified_roi, quality=75) if res.rectified_roi is not None else ""

                height_b64 = ""
                if res.height_map is not None:
                    h_norm = cv2.normalize(res.height_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    h_color = cv2.applyColorMap(h_norm, cv2.COLORMAP_VIRIDIS)
                    height_b64 = encode_image_base64(h_color, quality=80)

                defect_b64 = ""
                defect_cnt = 0
                if res.defect_summary is not None:
                    defect_b64 = encode_image_base64(res.defect_summary.heatmap_colored, quality=75)
                    defect_cnt = res.defect_summary.defect_count

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
            await asyncio.sleep(0.14)  # ~7 Hz active perception probe cycle

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        dashboard_sockets.discard(websocket)
