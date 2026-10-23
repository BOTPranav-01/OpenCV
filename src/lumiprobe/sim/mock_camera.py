"""Mock camera device and session replay emulator."""

import cv2
import numpy as np
from lumiprobe.physics.patterns import PatternBank
from lumiprobe.sim.render import PhotometricForwardRenderer, ProceduralSurfaceGenerator, SyntheticSurface


class SimulatedCameraStream:
    """Emulates a live webcam observing a test surface under screen patterns."""

    def __init__(self, target_type: str = "coin", res: tuple[int, int] = (640, 480)):
        self.width, self.height = res
        self.renderer = PhotometricForwardRenderer()
        self.target_type = target_type
        self.current_pattern_id = "WHITE"
        self._set_surface(target_type)

    def _set_surface(self, target_type: str):
        self.target_type = target_type
        if target_type == "coin":
            self.surface = ProceduralSurfaceGenerator.generate_coin(512)
        elif target_type == "flat_print":
            self.surface = ProceduralSurfaceGenerator.generate_flat_print(512)
        elif target_type == "braille":
            self.surface = ProceduralSurfaceGenerator.generate_braille(512)
        else:
            self.surface = ProceduralSurfaceGenerator.generate_coin(512)

    def set_pattern(self, pattern_id: str):
        """Update the active projected light pattern."""
        self.current_pattern_id = pattern_id

    def read_frame(self) -> np.ndarray:
        """Render and return the simulated webcam camera frame."""
        pattern_img = PatternBank.get_pattern(self.current_pattern_id, 1920, 1080)
        roi_frame = self.renderer.render_frame(self.surface, pattern_img)

        # Place ROI inside a full simulated camera canvas with simulated ArUco markers
        canvas = np.full((self.height, self.width, 3), 40, dtype=np.uint8)

        # Draw ArUco markers at the 4 corners of the ROI
        rw, rh = 360, 360
        rx = (self.width - rw) // 2
        ry = (self.height - rh) // 2

        # Resize rendered ROI to fit inside the canvas
        resized_roi = cv2.resize(roi_frame, (rw, rh))
        canvas[ry : ry + rh, rx : rx + rw] = resized_roi

        # Draw ArUco markers (DICT_4X4_50) around the window
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker_size = 40

        m0 = cv2.aruco.generateImageMarker(aruco_dict, 0, marker_size)
        m1 = cv2.aruco.generateImageMarker(aruco_dict, 1, marker_size)
        m2 = cv2.aruco.generateImageMarker(aruco_dict, 2, marker_size)
        m3 = cv2.aruco.generateImageMarker(aruco_dict, 3, marker_size)

        def paste_marker(m_img, x, y):
            m_bgr = cv2.cvtColor(m_img, cv2.COLOR_GRAY2BGR)
            canvas[y : y + marker_size, x : x + marker_size] = m_bgr

        # Corners: 0=TL, 1=TR, 2=BR, 3=BL
        paste_marker(m0, rx - marker_size - 4, ry - marker_size - 4)
        paste_marker(m1, rx + rw + 4, ry - marker_size - 4)
        paste_marker(m2, rx + rw + 4, ry + rh + 4)
        paste_marker(m3, rx - marker_size - 4, ry + rh + 4)

        return canvas
