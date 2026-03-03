import numpy as np
from constants import *

def get_cd(speed):
    """ Vectorized drag coefficient calculation including drag crisis.
    Returns (CD, Re).
    """
    speed = np.atleast_1d(speed)
    # Reynolds number calculation
    Re = rho / g * speed * 2 * ball_r / mu
    CD = np.full_like(speed, ball_cd1)
    
    # Drag crisis transition region (linear interpolation)
    crisis = (Re > 1e5) & (Re < 2e5)
    CD[crisis] = ball_cd1 + (Re[crisis] - 1e5) * (ball_cd2 - ball_cd1) / 1e5
    CD[Re >= 2e5] = ball_cd2
    
    return CD, Re

def get_acceleration(vel, omega):
    """ Calculate total continuous acceleration (Gravity + Drag + Magnus).
    Returns (accel, debug_info).
    """
    vel = np.atleast_2d(vel)
    omega = np.atleast_2d(omega)
    
    speed = np.linalg.norm(vel, axis=1)
    safe_speed = np.maximum(speed, 1e-6)
    
    # 1. Gravity
    accel = np.zeros_like(vel)
    accel[:, 2] -= g_eff
    
    # 2. Drag
    CD, Re = get_cd(speed)
    Fd_mag = 0.5 * CD * rho * speed**2 * (np.pi * ball_r**2)
    accel -= (Fd_mag / (ball_m * safe_speed))[:, None] * vel
    
    # 3. Magnus Force
    omega_mag = np.linalg.norm(omega, axis=1)
    Sp = np.clip(omega_mag * ball_r / safe_speed, 0, 3) # Spin factor
    CL = Sp * 0.77 + 0.12 # Lift coefficient from literature
    Fm = 0.5 * CL[:, None] * np.pi * ball_r**3 * rho * np.cross(omega, vel)
    accel += Fm / ball_m
    
    return accel, (CD, Fd_mag, Re, Sp, CL, Fm)

def apply_friction(v_rel, n, e, mu_f=0.4):
    """
    Applies impulse with normal reflection and tangential friction.
    ONLY triggers if v_rel is pointing INTO the surface (v_rel . n < 0).
    """
    # Ensure n is (N, 3) to match v_rel
    if n.shape[0] == 1 and v_rel.shape[0] > 1:
        n = np.tile(n, (v_rel.shape[0], 1))
        
    v_n_mag = np.sum(v_rel * n, axis=1)
    mask_towards = v_n_mag < -1e-6
    
    dv = np.zeros_like(v_rel)
    if not np.any(mask_towards):
        return dv, mask_towards

    vt = v_rel[mask_towards]
    nt = n[mask_towards]
    vn_mag_t = v_n_mag[mask_towards][:, None]
    
    # 1. Normal Impulse (Reflect with energy loss)
    # dv_n = -(1 + e) * (v . n) * n
    dv_n = -(1 + e) * vn_mag_t * nt
    
    # 2. Tangential Impulse (Friction)
    # Tangential velocity vector: v_t = v - (v . n) * n
    v_tan = vt - vn_mag_t * nt
    tan_mag = np.linalg.norm(v_tan, axis=1)
    tan_dir = np.zeros_like(v_tan)
    m = tan_mag > 1e-6
    tan_dir[m] = v_tan[m] / tan_mag[m][:, None]
    
    # Friction magnitude is proportional to normal impulse (mu * |dv_n|)
    # but capped by the current tangential speed to prevent unrealistic reversal.
    dv_t = -np.minimum(mu_f * np.abs(np.linalg.norm(dv_n, axis=1)), tan_mag)[:, None] * tan_dir
    
    dv[mask_towards] = dv_n + dv_t
    return dv, mask_towards

def resolve_collisions(pos, vel, omg, last_rim_pt):
    """ Detect and calculate impulse changes using velocity-direction masking.
    Returns (new_vel, last_rim_pt).
    """
    pos = np.atleast_2d(pos)
    vel = np.atleast_2d(vel).copy()
    omg = np.atleast_2d(omg)
    
    # 1. Backboard (modeled as a rectangle at x=bb_x)
    dx_bb = pos[:, 0] - bb_x
    dy_bb = np.maximum(0, np.maximum(-bb_l/2 - pos[:, 1], pos[:, 1] - bb_l/2))
    dz_bb = np.maximum(0, np.maximum(bb_z_bot - pos[:, 2], pos[:, 2] - bb_z_top))
    dist_bb = np.sqrt(dx_bb**2 + dy_bb**2 + dz_bb**2)
    
    mask_bb = dist_bb < ball_r
    if np.any(mask_bb):
        # Closest point on the backboard to determine normal
        pts = np.column_stack([
            np.full(np.sum(mask_bb), bb_x),
            np.clip(pos[mask_bb, 1], -bb_l/2, bb_l/2),
            np.clip(pos[mask_bb, 2], bb_z_bot, bb_z_top)
        ])
        n = pos[mask_bb] - pts
        n_mag = np.linalg.norm(n, axis=1)
        mask_on = n_mag < 1e-6
        n[mask_on] = [1, 0, 0] # Point away from board if center is inside
        n[~mask_on] /= n_mag[~mask_on][:, None]
        
        dv, hit = apply_friction(vel[mask_bb], n, ball_e2, mu_f=0.3)
        vel[np.where(mask_bb)[0][hit]] += dv[hit]

    # 2. Rim (Toroidal geometry)
    planar_dist = np.linalg.norm(pos[:, :2], axis=1)
    dist_to_rim = np.sqrt((planar_dist - rim_r)**2 + (pos[:, 2] - 10)**2)
    scale = rim_r / np.maximum(planar_dist, 1e-6)
    rim_pts = np.column_stack([pos[:, 0] * scale, pos[:, 1] * scale, np.full_like(planar_dist, 10.0)])
    
    # Memory: Skip if hitting exactly the same spot twice in a row (jitter prevention)
    not_recent = np.linalg.norm(rim_pts - last_rim_pt, axis=1) > 0.5
    mask_rim = (dist_to_rim < ball_r) & ((pos[:, 0] > -rim_r) | (np.abs(pos[:, 1]) > 0.25)) & not_recent
    if np.any(mask_rim):
        n = pos[mask_rim] - rim_pts[mask_rim]
        n_mag = np.linalg.norm(n, axis=1)
        n[n_mag > 0] /= n_mag[n_mag > 0][:, None]
        
        dv, hit = apply_friction(vel[mask_rim], n, ball_e2, mu_f=0.5)
        hit_indices = np.where(mask_rim)[0][hit]
        vel[hit_indices] += dv[hit]
        last_rim_pt[hit_indices] = rim_pts[hit_indices]

    # 3. Rim Connector (Horizontal Box)
    mask_conn = (pos[:, 0] > bb_x) & (pos[:, 0] < -rim_r) & (np.abs(pos[:, 1]) < 0.25) & \
                (pos[:, 2] > 10 - ball_r) & (pos[:, 2] < 10 + ball_r)
    if np.any(mask_conn):
        n = np.array([0, 0, 1]) # Simplification: assume flat top surface
        dv, hit = apply_friction(vel[mask_conn], n[None, :], ball_e2, mu_f=0.3)
        vel[np.where(mask_conn)[0][hit]] += dv[hit]

    # 4. Ground (Infinite Plane at z=0)
    mask_gr = (pos[:, 2] <= ball_r)
    if np.any(mask_gr):
        n = np.array([0, 0, 1])
        dv, hit = apply_friction(vel[mask_gr], n[None, :], ball_e1, mu_f=0.6)
        vel[np.where(mask_gr)[0][hit]] += dv[hit]
        
    return vel, last_rim_pt

def step_rk4(pos, vel, omega, dt):
    """ Perform one RK4 integration step for continuous forces. """
    def f(v): return get_acceleration(v, omega)[0]
    k1_v, k1_p = f(vel), vel
    k2_v, k2_p = f(vel + k1_v * dt / 2), vel + k1_v * dt / 2
    k3_v, k3_p = f(vel + k2_v * dt / 2), vel + k2_v * dt / 2
    k4_v, k4_p = f(vel + k3_v * dt), vel + k3_v * dt
    
    p_new = pos + (k1_p + 2*k2_p + 2*k3_p + k4_p) * dt / 6
    v_new = vel + (k1_v + 2*k2_v + 2*k3_v + k4_v) * dt / 6
    return p_new, v_new
