import numpy as np
from constants import *

def get_cd(speed):
    """ Vectorized drag coefficient calculation. """
    speed = np.atleast_1d(speed)
    Re = rho / g * speed * 2 * ball_r / mu
    CD = np.full_like(speed, ball_cd1)
    crisis = (Re > 1e5) & (Re < 2e5)
    CD[crisis] = ball_cd1 + (Re[crisis] - 1e5) * (ball_cd2 - ball_cd1) / 1e5
    CD[Re >= 2e5] = ball_cd2
    return CD

def get_acceleration(vel, omega):
    """ Calculate total continuous acceleration (Gravity + Drag + Magnus). """
    vel = np.atleast_2d(vel)
    omega = np.atleast_2d(omega)
    
    speed = np.linalg.norm(vel, axis=1)
    safe_speed = np.maximum(speed, 1e-6)
    
    # 1. Gravity
    accel = np.zeros_like(vel)
    accel[:, 2] -= g_eff
    
    # 2. Drag
    Re = rho / g * speed * 2 * ball_r / mu
    CD = get_cd(speed)
    Fd_mag = 0.5 * CD * rho * speed**2 * (np.pi * ball_r**2)
    accel -= (Fd_mag / (ball_m * safe_speed))[:, None] * vel
    
    # 3. Magnus Force
    omega_mag = np.linalg.norm(omega, axis=1)
    Sp = np.clip(omega_mag * ball_r / safe_speed, 0, 3)
    CL = Sp * 0.77 + 0.12
    Fm = 0.5 * CL[:, None] * np.pi * ball_r**3 * rho * np.cross(omega, vel)
    accel += Fm / ball_m
    
    return accel, (CD, Fd_mag, Re, Sp, CL, Fm)

def resolve_collisions(pos, vel, bb_cd, rim_cd, gr_cd, last_rim_pt):
    """ Detect and calculate impulse changes. Returns (new_vel, hit_masks, new_rim_pts). """
    pos = np.atleast_2d(pos)
    vel = np.atleast_2d(vel).copy()
    
    # 1. Backboard
    dx_bb = pos[:, 0] - bb_x
    dy_bb = np.maximum(0, np.maximum(-bb_l/2 - pos[:, 1], pos[:, 1] - bb_l/2))
    dz_bb = np.maximum(0, np.maximum(bb_z_bot - pos[:, 2], pos[:, 2] - bb_z_top))
    dist_bb = np.sqrt(dx_bb**2 + dy_bb**2 + dz_bb**2)
    
    hit_bb = (dist_bb < ball_r) & (bb_cd == 0)
    if np.any(hit_bb):
        straight = hit_bb & (pos[:, 2] > bb_z_bot) & (pos[:, 2] < bb_z_top) & (np.abs(pos[:, 1]) < bb_l/2)
        vel[straight, 0] = -vel[straight, 0] * ball_e2
        edge = hit_bb & ~straight
        if np.any(edge):
            pts = np.column_stack([np.full(np.sum(edge), bb_x), np.clip(pos[edge, 1], -bb_l/2, bb_l/2), np.clip(pos[edge, 2], bb_z_bot, bb_z_top)])
            n = pos[edge] - pts
            n_mag = np.linalg.norm(n, axis=1)
            n[n_mag > 0] /= n_mag[n_mag > 0][:, None]
            vel[edge] -= (1 + ball_e2) * np.sum(vel[edge] * n, axis=1)[:, None] * n

    # 2. Rim
    planar_dist = np.linalg.norm(pos[:, :2], axis=1)
    dist_to_rim = np.sqrt((planar_dist - rim_r)**2 + (pos[:, 2] - 10)**2)
    scale = rim_r / np.maximum(planar_dist, 1e-6)
    rim_pts = np.column_stack([pos[:, 0] * scale, pos[:, 1] * scale, np.full_like(planar_dist, 10.0)])
    
    not_recent = np.linalg.norm(rim_pts - last_rim_pt, axis=1) > 0.5
    # Exact parity with simulate.py: check x > -rim_r OR wide y
    hit_rim = (dist_to_rim < ball_r) & ((pos[:, 0] > -rim_r) | (np.abs(pos[:, 1]) > 0.25)) & not_recent & (rim_cd == 0)
    if np.any(hit_rim):
        n = pos[hit_rim] - rim_pts[hit_rim]
        n_mag = np.linalg.norm(n, axis=1)
        n[n_mag > 0] /= n_mag[n_mag > 0][:, None]
        vel[hit_rim] -= (1 + ball_e2) * np.sum(vel[hit_rim] * n, axis=1)[:, None] * n

    # 3. Rim Connector
    hit_conn = (pos[:, 0] > bb_x) & (pos[:, 0] < -rim_r) & (np.abs(pos[:, 1]) < 0.25) & \
               (pos[:, 2] > 10) & (pos[:, 2] < 10 + ball_r) & (rim_cd == 0)
    if np.any(hit_conn):
        vel[hit_conn, 2] = -vel[hit_conn, 2] * ball_e2

    # 4. Ground
    hit_gr = (pos[:, 2] <= ball_r) & (gr_cd == 0)
    if np.any(hit_gr):
        vel[hit_gr, 2] = -vel[hit_gr, 2] * ball_e1
        
    return vel, hit_bb, hit_rim, hit_conn, hit_gr, rim_pts

def step_rk4(pos, vel, omega, dt):
    """ Perform one RK4 integration step. """
    def f(v): return get_acceleration(v, omega)[0]
    k1_v, k1_p = f(vel), vel
    k2_v, k2_p = f(vel + k1_v * dt / 2), vel + k1_v * dt / 2
    k3_v, k3_p = f(vel + k2_v * dt / 2), vel + k2_v * dt / 2
    k4_v, k4_p = f(vel + k3_v * dt), vel + k3_v * dt
    return pos + (k1_p + 2*k2_p + 2*k3_p + k4_p) * dt / 6, vel + (k1_v + 2*k2_v + 2*k3_v + k4_v) * dt / 6
