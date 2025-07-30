
"""
metrics.py — MVP golf swing metrics from 2D pose keypoints (video frames)

Assumptions
-----------
- Input is a list of frames; each frame is a dict mapping body part names to (x, y) pixel coords.
  Required keys (Mediapipe/YOLOv8-Pose compatible): 
  'nose','left_shoulder','right_shoulder','left_hip','right_hip','left_elbow',
  'right_elbow','left_wrist','right_wrist','left_knee','right_knee','left_ankle','right_ankle'
- Camera is FACE-ON (target is to golfer's left for a right-handed player).
- Units are pixels; where needed we report pixels and relative changes. If you know a real-world scale,
  pass `px_per_cm` to get centimeters.
- Handedness defaults to 'right' but can be 'left'.

Outputs
-------
A dictionary with metrics ready to send back to the UI.

You can integrate this with your FastAPI endpoint by:
1) Running a pose model and building the `frames_keypoints` list.
2) Calling compute_metrics(frames_keypoints, fps=..., handedness='right').
"""

from typing import Dict, List, Tuple, Optional
import math
import numpy as np

Point = Tuple[float, float]
Frame = Dict[str, Point]

def _angle_deg(a: Point, b: Point, c: Point) -> float:
    """Return the angle ABC in degrees."""
    v1 = (a[0]-b[0], a[1]-b[1])
    v2 = (c[0]-b[0], c[1]-b[1])
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    n1 = math.hypot(*v1); n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return float('nan')
    cosang = max(-1.0, min(1.0, dot/(n1*n2)))
    return math.degrees(math.acos(cosang))

def _vec_angle_deg(p: Point, q: Point) -> float:
    """Angle of vector p->q relative to +x axis (degrees)."""
    return math.degrees(math.atan2(q[1]-p[1], q[0]-p[0]))

def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

def _mid(a: Point, b: Point) -> Point:
    return ((a[0]+b[0])/2.0, (a[1]+b[1])/2.0)

def _smooth(arr, k: int = 5):
    import numpy as np
    if len(arr) == 0:
        return np.array([])
    k = max(1, min(k, len(arr)))
    w = np.ones(k)/k
    return np.convolve(arr, w, mode='same')

def detect_top_frame(shoulder_turn_series):
    """Top of backswing = max absolute shoulder turn."""
    import numpy as np
    if not shoulder_turn_series: return 0
    return int(np.nanargmax(np.abs(shoulder_turn_series)))

def detect_impact_frame(wrist_speed_series):
    """Impact ~ peak wrist speed in latter half of swing."""
    import numpy as np
    n = len(wrist_speed_series)
    if n == 0: return 0
    start = n//2
    idx = int(start + np.nanargmax(wrist_speed_series[start:]))
    return idx

def compute_metrics(frames, fps: float, handedness: str = 'right', px_per_cm: Optional[float] = None):
    """Compute MVP swing metrics from pose keypoints across frames."""
    if not frames:
        return {}

    # Choose lead/trail sides
    lead = 'left' if handedness == 'right' else 'right'
    trail = 'right' if handedness == 'right' else 'left'

    # Build time series
    def series(name):
        return [f.get(name, (float('nan'), float('nan'))) for f in frames]

    Ls = series(f'{lead}_shoulder'); Rs = series(f'{trail}_shoulder')
    Lh = series(f'{lead}_hip');      Rh = series(f'{trail}_hip')
    Le = series(f'{lead}_elbow');    Re = series(f'{trail}_elbow')
    Lw = series(f'{lead}_wrist');    Rw = series(f'{trail}_wrist')
    nose = series('nose')

    # Shoulder & pelvis line angles relative to horizontal
    def line_angle_series(a_series, b_series):
        return [_vec_angle_deg(a, b) for a, b in zip(a_series, b_series)]

    shoulder_line = line_angle_series(Ls, Rs)
    pelvis_line   = line_angle_series(Lh, Rh)

    address_idx = 0  # assume first frame ~address
    shoulder_turn_rel = [ang - shoulder_line[address_idx] for ang in shoulder_line]
    pelvis_turn_rel   = [ang - pelvis_line[address_idx] for ang in pelvis_line]

    # Top & impact detection
    top_idx = detect_top_frame(shoulder_turn_rel)

    # Wrist (lead) speed magnitude (px/s)
    wrist_speed = []
    for i in range(1, len(Lw)):
        dx = Lw[i][0] - Lw[i-1][0]
        dy = Lw[i][1] - Lw[i-1][1]
        wrist_speed.append((dx**2 + dy**2) ** 0.5 * fps)
    wrist_speed = [wrist_speed[0] if wrist_speed else float('nan')] + wrist_speed
    wrist_speed_s = _smooth(wrist_speed, 7)
    impact_idx = detect_impact_frame(list(wrist_speed_s))

    # Head sway
    nose_x = [p[0] for p in nose]
    nose_x0 = nose_x[address_idx]
    head_sway_px = max(abs(x - nose_x0) for x in nose_x if not (x != x))  # exclude NaN
    head_sway_cm = head_sway_px/px_per_cm if px_per_cm else None

    # X-factor at top
    shoulder_turn_top = shoulder_turn_rel[top_idx]
    pelvis_turn_top   = pelvis_turn_rel[top_idx]
    x_factor = shoulder_turn_top - pelvis_turn_top

    # Spine tilt change (address -> impact)
    def spine_tilt(idx):
        s_mid = ((Ls[idx][0]+Rs[idx][0])/2.0, (Ls[idx][1]+Rs[idx][1])/2.0)
        h_mid = ((Lh[idx][0]+Rh[idx][0])/2.0, (Lh[idx][1]+Rh[idx][1])/2.0)
        ang = _vec_angle_deg(h_mid, s_mid)
        tilt_from_vertical = 90.0 - (ang % 180.0)
        return tilt_from_vertical
    spine_tilt_address = spine_tilt(address_idx)
    spine_tilt_impact  = spine_tilt(impact_idx)
    spine_tilt_change  = spine_tilt_impact - spine_tilt_address

    # Lead elbow angle (shoulder–elbow–wrist) as a lag proxy
    elbow_angle = []
    for i in range(len(frames)):
        elbow_angle.append(_angle_deg(Ls[i], Le[i], Lw[i]))
    # Define lag proxy at top
    lag_proxy_top = elbow_angle[top_idx]

    # Release timing = time from top to impact
    release_time_ms = max(0, (impact_idx - top_idx) / fps * 1000.0)

    # Swing plane variance proxy during downswing
    lead_arm_ang = []
    for i in range(len(frames)):
        lead_arm_ang.append(_vec_angle_deg(Ls[i], Lw[i]))
    downswing_start = max(address_idx, top_idx)
    import numpy as np
    plane_var = float(np.nanstd([lead_arm_ang[i] - shoulder_line[i] for i in range(downswing_start, impact_idx+1)]))

    # Forearm lean at impact (proxy for shaft lean)
    forearm_ang_impact = _vec_angle_deg(Le[impact_idx], Lw[impact_idx])
    forearm_lean_vs_vertical = 90.0 - (forearm_ang_impact % 180.0)

    return {
        "frames": len(frames),
        "fps": fps,
        "address_idx": address_idx,
        "top_idx": top_idx,
        "impact_idx": impact_idx,
        "head_sway_px": float(head_sway_px),
        "head_sway_cm": float(head_sway_cm) if head_sway_cm is not None else None,
        "shoulder_turn_top_deg": float(shoulder_turn_top),
        "pelvis_turn_top_deg": float(pelvis_turn_top),
        "x_factor_deg": float(x_factor),
        "spine_tilt_change_deg": float(spine_tilt_change),
        "lag_proxy_elbow_angle_top_deg": float(lag_proxy_top),
        "release_time_ms": float(release_time_ms),
        "swing_plane_var_deg": float(plane_var),
        "forearm_lean_vs_vertical_deg_at_impact": float(forearm_lean_vs_vertical),
    }
