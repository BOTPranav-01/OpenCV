"""ArUco window card detector, pose estimator, and perspective normalizer.

Uses OpenCV 5 ArucoDetector to identify 4 corner markers framing the inspection window.
Computes projective homography for metric rectification and solvePnP for user positioning guidance.
"""

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class WindowDetectionResult:
    """Detection state of the physical ArUco window card."""

    detected: bool
    corner_points: np.ndarray | None  # (4, 2) [TL, TR, BR, BL] in image coordinates
    homography: np.ndarray | None  # (3, 3) warp perspective matrix
    pose_rvec: np.ndarray | None  # (3, 1) rotation vector from solvePnP
    pose_tvec: np.ndarray | None  # (3, 1) translation vector from solvePnP
    guidance_message: str  # Instruction for user (e.g. 'OK', 'Tilt left', 'Move closer')


class ArucoWindowTracker:
    """Tracks the 4-marker ArUco window card (IDs 0: TL, 1: TR, 2: BR, 3: BL)."""

    def __init__(self, target_roi_size: tuple[int, int] = (512, 512)):
        self.target_w, self.target_h = target_roi_size
        self.dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dict, self.params)

        # Canonical destination coordinates for rectified fronto-parallel window
        self.dst_pts = np.array(
            [
                [0.0, 0.0],
                [self.target_w - 1.0, 0.0],
                [self.target_w - 1.0, self.target_h - 1.0],
                [0.0, self.target_h - 1.0],
            ],
            dtype=np.float32,
        )

    def detect(self, frame: np.ndarray, camera_matrix: np.ndarray | None = None) -> WindowDetectionResult:
        """Detect the 4-corner window frame and return perspective transformation."""
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        corners, ids, rejected = self.detector.detectMarkers(gray)

        if ids is None or len(ids) < 4:
            count = 0 if ids is None else len(ids)
            return WindowDetectionResult(
                detected=False,
                corner_points=None,
                homography=None,
                pose_rvec=None,
                pose_tvec=None,
                guidance_message=f"Need 4 markers, found {count}. Position window card in camera view.",
            )

        # Match marker IDs 0, 1, 2, 3 (compatible with 1D and 2D ids in OpenCV 5/4)
        id_map = {int(marker_id.ravel()[0]): idx for idx, marker_id in enumerate(ids)}
        required_ids = [0, 1, 2, 3]

        if not all(k in id_map for k in required_ids):
            missing = [k for k in required_ids if k not in id_map]
            return WindowDetectionResult(
                detected=False,
                corner_points=None,
                homography=None,
                pose_rvec=None,
                pose_tvec=None,
                guidance_message=f"Missing markers: {missing}. Ensure card is fully visible.",
            )

        # Extract the inner corner of each marker framing the inner window:
        # Marker 0 (TL): bottom-right corner of marker is point index 2
        # Marker 1 (TR): bottom-left corner of marker is point index 3
        # Marker 2 (BR): top-left corner of marker is point index 0
        # Marker 3 (BL): top-right corner of marker is point index 1
        c0 = corners[id_map[0]][0][2]
        c1 = corners[id_map[1]][0][3]
        c2 = corners[id_map[2]][0][0]
        c3 = corners[id_map[3]][0][1]

        src_pts = np.array([c0, c1, c2, c3], dtype=np.float32)

        # Compute Homography
        h_matrix, _ = cv2.findHomography(src_pts, self.dst_pts)

        # Estimate card pose if camera intrinsics are provided
        rvec, tvec = None, None
        guidance = "Window card locked. Ready for capture."

        if camera_matrix is not None:
            # Assume 60mm square window in world coordinates
            w_size = 0.060
            obj_pts = np.array(
                [
                    [-w_size / 2.0, w_size / 2.0, 0.0],
                    [w_size / 2.0, w_size / 2.0, 0.0],
                    [w_size / 2.0, -w_size / 2.0, 0.0],
                    [-w_size / 2.0, -w_size / 2.0, 0.0],
                ],
                dtype=np.float32,
            )
            dist_coeffs = np.zeros((4, 1), dtype=np.float32)
            success, rvec, tvec = cv2.solvePnP(obj_pts, src_pts, camera_matrix, dist_coeffs)

            if success:
                dist = tvec[2, 0]
                if dist > 0.45:
                    guidance = "Move window card closer to screen (approx 20-30cm)."
                elif dist < 0.15:
                    guidance = "Move window card slightly further away."
                else:
                    guidance = "Pose optimal. Ready."

        return WindowDetectionResult(
            detected=True,
            corner_points=src_pts,
            homography=h_matrix,
            pose_rvec=rvec,
            pose_tvec=tvec,
            guidance_message=guidance,
        )

    def warp_roi(self, frame: np.ndarray, homography: np.ndarray) -> np.ndarray:
        """Rectify window frame region into fronto-parallel target resolution."""
        return cv2.warpPerspective(
            frame,
            homography,
            (self.target_w, self.target_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
