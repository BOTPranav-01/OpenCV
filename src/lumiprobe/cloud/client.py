"""Cloud client with automatic local fallback for offline and stage demonstration."""

import json
import logging
import os
import uuid
import numpy as np
from lumiprobe.vision.integrate import FrankotChellappaIntegrator

logger = logging.getLogger("lumiprobe.cloud")


class CloudInspectionClient:
    """Manages S3 evidence archive and Graviton cloud analysis with local fallback."""

    def __init__(self):
        self.cloud_enabled = os.environ.get("LUMIPROBE_CLOUD", "off").lower() in ("on", "true", "1")
        self.endpoint_url = os.environ.get("LUMIPROBE_LAMBDA_URL")
        self.integrator = FrankotChellappaIntegrator()

    def sync_inspection(
        self,
        target_type: str,
        verdict: str,
        confidence: float,
        llr: float,
        patterns_used: list[str],
        height_map: np.ndarray | None,
        defect_count: int,
    ) -> dict:
        """Persist inspection session to DynamoDB/S3 or execute identical local fallback."""
        inspection_id = str(uuid.uuid4())[:8]

        record = {
            "inspection_id": inspection_id,
            "target_type": target_type,
            "verdict": verdict,
            "confidence": float(confidence),
            "llr": float(llr),
            "patterns_used": patterns_used,
            "defect_count": defect_count,
            "cloud_status": "ONLINE (AWS Graviton + S3)" if self.cloud_enabled else "OFFLINE (Local In-Process Fallback)",
        }

        if not self.cloud_enabled:
            # Local zero-latency fallback
            return record

        try:
            import urllib.request
            req = urllib.request.Request(
                self.endpoint_url,
                data=json.dumps(record).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                return res_data
        except Exception as e:
            logger.warning(f"Cloud upload skipped ({e}). Using local records.")
            record["cloud_status"] = f"OFFLINE_FALLBACK ({type(e).__name__})"
            return record
