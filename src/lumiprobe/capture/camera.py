"""Camera acquisition manager with Windows DirectShow/UVC lock and simulator fallback."""

import logging
import cv2
import numpy as np
from lumiprobe.sim.mock_camera import SimulatedCameraStream

logger = logging.getLogger("lumiprobe.capture")


class CameraDevice:
    """Manages physical webcam video capture with UVC lock and simulation support."""

    def __init__(self, camera_index: int = 0, force_sim: bool = True, sim_target: str = "coin"):
        self.camera_index = camera_index
        self.force_sim = force_sim
        self.sim_target = sim_target
        self.cap: cv2.VideoCapture | None = None
        self.sim_stream: SimulatedCameraStream | None = None
        self.is_simulated = force_sim
        self.open()

    def open(self):
        """Initialize simulation or physical camera based on configuration."""
        if self.force_sim or self.is_simulated:
            self._init_simulation()
            return

        self._init_webcam()

    def _init_webcam(self) -> bool:
        """Attempt to open physical hardware camera."""
        try:
            if self.cap is not None:
                self.cap.release()

            # DirectShow on Windows for UVC exposure lock
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_index)

            if not self.cap.isOpened():
                logger.warning("No physical webcam detected. Falling back to physics simulator.")
                self._init_simulation()
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                logger.warning("Webcam failed initial read. Using simulator.")
                self._init_simulation()
                return False

            self.is_simulated = False
            logger.info("Physical camera successfully initialized.")
            return True

        except Exception as e:
            logger.warning(f"Camera open error ({e}). Using simulator.")
            self._init_simulation()
            return False

    def _init_simulation(self):
        self.is_simulated = True
        if self.sim_stream is None:
            self.sim_stream = SimulatedCameraStream(self.sim_target)
        else:
            self.sim_stream._set_surface(self.sim_target)
        logger.info(f"Physics simulator active with target '{self.sim_target}'")

    def set_source(self, source: str, target: str = "coin"):
        """Explicitly switch between 'sim' (physics engine) and 'webcam' (hardware)."""
        if source == "webcam":
            self.is_simulated = False
            success = self._init_webcam()
            if not success:
                logger.info("Webcam unavailable; staying on simulator.")
        else:
            self.sim_target = target
            self.is_simulated = True
            self._init_simulation()

    def set_pattern(self, pattern_id: str):
        """Update simulated pattern projection."""
        if self.sim_stream is not None:
            self.sim_stream.set_pattern(pattern_id)

    def set_sim_target(self, target_type: str):
        """Switch simulation surface between 'coin', 'flat_print', 'braille'."""
        self.sim_target = target_type
        self.is_simulated = True
        self._init_simulation()

    def read(self) -> tuple[bool, np.ndarray]:
        """Capture next frame from webcam or physics simulator."""
        if self.is_simulated:
            if self.sim_stream is None:
                self._init_simulation()
            return True, self.sim_stream.read_frame()

        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return True, frame

        # Fallback to simulation if physical read drops
        if self.sim_stream is None:
            self._init_simulation()
        return True, self.sim_stream.read_frame()

    def release(self):
        """Release hardware video capture."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
