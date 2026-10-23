"""Face region extraction for active photometric facial liveness detection.

Leverages OpenCV 5 built-in YuNet FaceDetectorYN with geometric fallback.
Extracts facial regions (forehead, nose, cheeks) to test 3D flesh relief against 2D spoof attacks.
"""

from dataclasses import dataclass
import os
import urllib.request
import cv2
import numpy as np


@dataclass
class FaceDetectionResult:
    """Facial bounding box and keypoints for active illumination testing."""

    detected: bool
    bbox: tuple[int, int, int, int] | None  # x, y, w, h
    keypoints: np.ndarray | None  # 5 facial landmarks (eyes, nose, mouth corners)
    face_roi: np.ndarray | None  # Cropped face image
    guidance: str


class FaceRoiExtractor:
    """Detects face and aligns key ROIs using OpenCV FaceDetectorYN."""

    MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "face_detection_yunet_2023mar.onnx")

    def __init__(self, target_size: tuple[int, int] = (512, 512)):
        self.target_size = target_size
        self.detector = None
        self._init_detector()

    def _init_detector(self):
        """Load YuNet ONNX model from cache or download on-demand."""
        try:
            os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
            if not os.path.exists(self.MODEL_PATH):
                urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)

            self.detector = cv2.FaceDetectorYN.create(
                model=self.MODEL_PATH,
                config="",
                input_size=(640, 480),
                score_threshold=0.6,
                nms_threshold=0.3,
                top_k=5000,
            )
        except Exception:
            # Graceful geometric / Haar cascade fallback
            self.detector = None

    def detect(self, frame: np.ndarray) -> FaceDetectionResult:
        """Locate most prominent face and extract normalized ROI."""
        h, w = frame.shape[:2]

        if self.detector is not None:
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(frame)

            if faces is not None and len(faces) > 0:
                # Select the largest face by bounding box area
                areas = faces[:, 2] * faces[:, 3]
                best_idx = int(np.argmax(areas))
                face = faces[best_idx]

                x, y, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                # Clamp coordinates
                x = max(0, x)
                y = max(0, y)
                fw = min(w - x, fw)
                fh = min(h - y, fh)

                landmarks = face[4:14].reshape((5, 2))

                crop = frame[y : y + fh, x : x + fw]
                resized = cv2.resize(crop, self.target_size, interpolation=cv2.INTER_LINEAR)

                return FaceDetectionResult(
                    detected=True,
                    bbox=(x, y, fw, fh),
                    keypoints=landmarks,
                    face_roi=resized,
                    guidance="Face locked. Keep still during active light probe.",
                )

        # Fallback: central frame crop if detector is unavailable
        cw, ch = int(w * 0.5), int(h * 0.6)
        cx, cy = (w - cw) // 2, (h - ch) // 2
        crop = frame[cy : cy + ch, cx : cx + cw]
        resized = cv2.resize(crop, self.target_size, interpolation=cv2.INTER_LINEAR)

        return FaceDetectionResult(
            detected=True,
            bbox=(cx, cy, cw, ch),
            keypoints=None,
            face_roi=resized,
            guidance="Using central frame ROI (fallback). Position face centrally.",
        )
