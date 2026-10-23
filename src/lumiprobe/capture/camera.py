"""Camera acquisition manager with Windows DirectShow/UVC lock and simulator fallback."""

import logging
import cv2
import numpy as np
from lumiprobe.sim.mock_camera import SimulatedCameraStream

logger = logging.getLogger("lumiprobe.capture")


class CameraDevice:
    """Manages physical webcam video capture with UVC lock and simulation support."""

    def __init__(self, camera_index: int = 0, force_sim: bool = False, sim_target: str = "coin"):
        self.camera_index = camera_index
        self.force_sim = force_sim
        self.sim_target = sim_target
        self.cap: cv2.VideoCapture | None = None
        self.sim_stream: SimulatedCameraStream | None = None
        self.is_simulated = False
        self.open()

    def open(self):
        """Initialize physical camera or fall back to simulation."""
        if self.force_sim:
            self._init_simulation()
            return

        try:
            # On Windows, DirectShow (CAP_DSHOW) allows direct control over exposure and gain
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                # Fallback to default backend
                self.cap = cv2.VideoCapture(self.camera_index)

            if not self.cap.isOpened():
                logger.warning("No physical webcam detected. Falling back to high-fidelity physics simulator.")
                self._init_simulation()
                return

            # Request 1280x720 resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            # Attempt to disable auto-exposure (UVC standard: 0.25 or 0 is manual mode)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

            # Verify read
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                logger.warning("Webcam failed initial frame read. Using simulator.")
                self._init_simulation()
            else:
                self.is_simulated = False
                logger.info(f"Physical camera initialized at {test_frame.shape[1]}x{test_frame.shape[0]}")

        except Exception as e:
            logger.warning(f"Camera open error ({e}). Using simulator.")
            self._init_simulation()

    def _init_simulation(self):
        self.is_simulated = True
        self.sim_stream = SimulatedCameraStream(self.sim_target)
        logger.info(f"Physics simulator active with target '{self.sim_target}'")

    def set_pattern(self, pattern_id: str):
        """Update simulated pattern projection."""
        if self.sim_stream is not None:
            self.sim_stream.set_pattern(pattern_id)

    def set_sim_target(self, target_type: str):
        """Switch simulation surface between 'coin', 'flat_print', 'braille'."""
        if self.sim_stream is not None:
            self.sim_stream._set_surface(target_type)

    def read(self) -> tuple[bool, np.ndarray]:
        """Capture next frame from webcam or physics simulator."""
        if self.is_simulated and self.sim_stream is not None:
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
