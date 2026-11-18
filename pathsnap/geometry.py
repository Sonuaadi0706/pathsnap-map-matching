"""Planar segment geometry (metres, local projection)."""
import numpy as np


def point_segment(px, py, ax, ay, bx, by):
    """Distance from points p to segments a-b (all arrays broadcastable).
    Returns (distance, t) with t the clipped projection fraction in [0,1]."""
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    t = np.where(L2 > 1e-9, ((px - ax) * vx + (py - ay) * vy) / np.maximum(L2, 1e-9), 0.0)
    tc = np.clip(t, 0.0, 1.0)
    qx, qy = ax + tc * vx, ay + tc * vy
    return np.hypot(px - qx, py - qy), tc, t


def bearing(dx, dy):
    """Planar bearing in radians (atan2 of dy, dx)."""
    return np.arctan2(dy, dx)
