"""Single-command runner for LumiProbe-X: Active Photometric Reality Engine."""

import os
import sys
import uvicorn

# Ensure src is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

if __name__ == "__main__":
    print("=" * 70)
    print("  LumiProbe-X | Active Photometric Reality Engine")
    print("  OpenCV 5.0 Core | Agentic Vision Track | AWS Graviton Ready")
    print("=" * 70)
    print("  Dashboard UI : http://localhost:8000")
    print("  Active Lamp  : http://localhost:8000/lamp")
    print("  API Docs     : http://localhost:8000/docs")
    print("=" * 70)
    uvicorn.run("lumiprobe.server.app:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
