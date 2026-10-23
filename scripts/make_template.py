"""Generates printable ArUco window calibration template card.

Emits an A4/Letter size card with 4 corner ArUco markers (DICT_4X4_50)
and a 60mm x 60mm inspection window cutout in the center.
"""

import os
import cv2
import numpy as np


def generate_window_template(output_path: str = "window_card_template.png"):
    """Render high-resolution 300 DPI printable card image."""
    # A4 at 150 DPI = 1240 x 1754 px
    h, w = 1754, 1240
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    # ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_size = 140

    # Markers: 0 (TL), 1 (TR), 2 (BR), 3 (BL)
    m0 = cv2.aruco.generateImageMarker(aruco_dict, 0, marker_size)
    m1 = cv2.aruco.generateImageMarker(aruco_dict, 1, marker_size)
    m2 = cv2.aruco.generateImageMarker(aruco_dict, 2, marker_size)
    m3 = cv2.aruco.generateImageMarker(aruco_dict, 3, marker_size)

    # Center window: 450 x 450 px
    cx, cy = w // 2, h // 2
    win_half = 225

    # Corner positions for markers
    offset = 40
    pos_tl = (cx - win_half - marker_size - offset, cy - win_half - marker_size - offset)
    pos_tr = (cx + win_half + offset, cy - win_half - marker_size - offset)
    pos_br = (cx + win_half + offset, cy + win_half + offset)
    pos_bl = (cx - win_half - marker_size - offset, cy + win_half + offset)

    def draw_marker(m_img, pos):
        m_bgr = cv2.cvtColor(m_img, cv2.COLOR_GRAY2BGR)
        img[pos[1] : pos[1] + marker_size, pos[0] : pos[0] + marker_size] = m_bgr

    draw_marker(m0, pos_tl)
    draw_marker(m1, pos_tr)
    draw_marker(m2, pos_br)
    draw_marker(m3, pos_bl)

    # Draw dashed cut-out outline for the central window
    cv2.rectangle(
        img,
        (cx - win_half, cy - win_half),
        (cx + win_half, cy + win_half),
        (100, 100, 100),
        2,
        cv2.LINE_AA,
    )

    # Instructional text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        img,
        "LumiProbe-X | ArUco Metric Window Card",
        (cx - 300, cy - win_half - 220),
        font,
        0.9,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "CUT OUT CENTER WINDOW (Place target object inside)",
        (cx - 240, cy),
        font,
        0.55,
        (120, 120, 120),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "Align top of card towards laptop screen at 100-110 deg tilt",
        (cx - 280, cy + win_half + 240),
        font,
        0.65,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Generated template card: {output_path}")


if __name__ == "__main__":
    generate_window_template("scripts/window_card_template.png")
