import numpy as np
from constants import *

def get_cd(speed):
    """
    Calculates the drag coefficient based on the Reynolds number (Drag Crisis).
    
    Args:
        speed (float or ndarray): Translational speed of the ball [ft/s].
        
    Returns:
        tuple: (CD, Re) where CD is the drag coefficient and Re is the Reynolds number.
    """
    speed = np.atleast_1d(speed)
    Re = rho / g * speed * 2 * ball_r / mu
    CD = np.full_like(speed, ball_cd1)
    
    # Linear transition for drag crisis between Re 1e5 and 2e5
    crisis = (Re > 1e5) & (Re < 2e5)
    CD[crisis] = ball_cd1 + (Re[crisis] - 1e5) * (ball_cd2 - ball_cd1) / 1e5
    CD[Re >= 2e5] = ball_cd2
    
    return CD, Re

def get_acceleration(vel, omega):
    """
    Calculates total continuous acceleration from Gravity, Drag, and Magnus forces.
    
    Args:
        vel (ndarray): (N, 3) array of velocities [ft/s].
        omega (ndarray): (N, 3) array of angular velocities [rad/s].
        
    Returns:
        tuple: (accel, debug_info) where accel is (N, 3) and debug_info contains coefficients.
    """
    vel = np.atleast_2d(vel)
    omega = np.atleast_2d(omega)
    
    speed = np.linalg.norm(vel, axis=1)
    safe_speed = np.maximum(speed, 1e-6)
    
    # 1. Gravity
    accel = np.zeros_like(vel)
    accel[:, 2] -= g_eff
    
    # 2. Drag Force
    CD, Re = get_cd(speed)
    Fd_mag = 0.5 * CD * rho * speed**2 * (np.pi * ball_r**2)
    accel -= (Fd_mag / (ball_m * safe_speed))[:, None] * vel
    
    # 3. Magnus Force (Lift)
    omega_mag = np.linalg.norm(omega, axis=1)
    Sp = np.clip(omega_mag * ball_r / safe_speed, 0, 3) # Spin factor
    CL = Sp * 0.77 + 0.12 # CL from literature
    Fm = 0.5 * CL[:, None] * np.pi * ball_r**3 * rho * np.cross(omega, vel)
    accel += Fm / ball_m
    
    return accel, (CD, Fd_mag, Re, Sp, CL, Fm)

def apply_friction(v_cm, omg, n, e, mu_f):
    """
    Applies an impulse with normal restitution and tangential friction.
    
    Args:
        v_cm (ndarray): (N, 3) velocities of center of mass.
        omg (ndarray): (N, 3) angular velocities.
        n (ndarray): (N, 3) surface normal vectors.
        e (float): Coefficient of restitution.
        mu_f (float): Coefficient of friction.
        
    Returns:
        tuple: (delta_v, delta_w, hit_mask)
    """
    # Ensure n matches v_cm shape for broadcasting (needed for ground/connector)
    if n.shape[0] == 1 and v_cm.shape[0] > 1:
        n = np.tile(n, (v_cm.shape[0], 1))

    v_n_mag = np.sum(v_cm * n, axis=1)
    mask_towards = v_n_mag < -1e-6
    
    dv = np.zeros_like(v_cm)
    dw = np.zeros_like(omg)
    if not np.any(mask_towards):
        return dv, dw, mask_towards

    vt_cm = v_cm[mask_towards]
    ot = omg[mask_towards]
    nt = n[mask_towards]
    vn_mag_t = v_n_mag[mask_towards][:, None]

    # Normal Impulse (Restitution)
    dv_n = -(1 + e) * vn_mag_t * nt
    
    # Contact Point Tangential Velocity
    r_contact = -ball_r * nt
    v_pt = vt_cm + np.cross(ot, r_contact)
    v_tan_pt = v_pt - vn_mag_t * nt
    
    tan_mag = np.linalg.norm(v_tan_pt, axis=1)
    tan_dir = np.zeros_like(v_tan_pt)
    m = tan_mag > 1e-6
    tan_dir[m] = v_tan_pt[m] / tan_mag[m][:, None]
    
    # Tangential Impulse (Friction)
    dv_t = -np.minimum(mu_f * np.abs(np.linalg.norm(dv_n, axis=1)), tan_mag)[:, None] * tan_dir
    
    dv[mask_towards] = dv_n + dv_t
    # 4. Angular Momentum Change (Torque from friction)
    dw[mask_towards] = -np.cross(nt, dv_t) / (alpha * ball_r)
    
    return dv, dw, mask_towards

def resolve_collisions(pos, vel, omg, last_rim_pt):
    """
    Detects and resolves collisions with the backboard, rim, connector, and ground.
    
    Returns:
        tuple: (new_vel, new_omg, last_rim_pt)
    """
    pos = np.atleast_2d(pos)
    vel = np.atleast_2d(vel).copy()
    omg = np.atleast_2d(omg).copy()
    
    # 1. Backboard
    dx_bb = pos[:, 0] - bb_x
    dy_bb = np.maximum(0, np.maximum(-bb_l/2 - pos[:, 1], pos[:, 1] - bb_l/2))
    dz_bb = np.maximum(0, np.maximum(bb_z_bot - pos[:, 2], pos[:, 2] - bb_z_top))
    dist_bb = np.sqrt(dx_bb**2 + dy_bb**2 + dz_bb**2)
    
    mask_bb = dist_bb < ball_r
    if np.any(mask_bb):
        # Determine normal vector based on closest point on the board
        pts = np.column_stack([np.full(np.sum(mask_bb), bb_x), np.clip(pos[mask_bb, 1], -bb_l/2, bb_l/2), np.clip(pos[mask_bb, 2], bb_z_bot, bb_z_top)])
        n = pos[mask_bb] - pts
        n_mag = np.linalg.norm(n, axis=1)
        mask_on = n_mag < 1e-6
        n[mask_on] = [1, 0, 0]
        n[~mask_on] /= n_mag[~mask_on][:, None]
        
        dv, dw, hit = apply_friction(vel[mask_bb], omg[mask_bb], n, ball_e2, mu_bb)
        indices = np.where(mask_bb)[0][hit]
        vel[indices] += dv[hit]
        omg[indices] += dw[hit]

    # 2. Rim
    planar_dist = np.linalg.norm(pos[:, :2], axis=1)
    dist_to_rim = np.sqrt((planar_dist - rim_r)**2 + (pos[:, 2] - 10)**2)
    scale = rim_r / np.maximum(planar_dist, 1e-6)
    rim_pts = np.column_stack([pos[:, 0] * scale, pos[:, 1] * scale, np.full_like(planar_dist, 10.0)])
    
    # Skip if hitting exactly the same spot twice in a row (jitter prevention)
    not_recent = np.linalg.norm(rim_pts - last_rim_pt, axis=1) > 0.5
    mask_rim = (dist_to_rim < ball_r) & ((pos[:, 0] > -rim_r) | (np.abs(pos[:, 1]) > 0.25)) & not_recent
    if np.any(mask_rim):
        n = pos[mask_rim] - rim_pts[mask_rim]
        n_mag = np.linalg.norm(n, axis=1)
        n[n_mag > 0] /= n_mag[n_mag > 0][:, None]
        
        dv, dw, hit = apply_friction(vel[mask_rim], omg[mask_rim], n, ball_e2, mu_rim)
        hit_indices = np.where(mask_rim)[0][hit]
        vel[hit_indices] += dv[hit]
        omg[hit_indices] += dw[hit]
        last_rim_pt[hit_indices] = rim_pts[hit_indices]

    # 3. Rim Connector
    mask_conn = (pos[:, 0] > bb_x) & (pos[:, 0] < -rim_r) & (np.abs(pos[:, 1]) < 0.25) & \
                (pos[:, 2] > 10 - ball_r) & (pos[:, 2] < 10 + ball_r)
    if np.any(mask_conn):
        n = np.zeros((np.sum(mask_conn), 3))
        # Normal is up if hitting top, down if hitting bottom
        n[:, 2] = np.where(pos[mask_conn, 2] > 10.0, 1.0, -1.0)
        dv, dw, hit = apply_friction(vel[mask_conn], omg[mask_conn], n, ball_e2, mu_rim)
        indices = np.where(mask_conn)[0][hit]
        vel[indices] += dv[hit]
        omg[indices] += dw[hit]

    # 4. Ground
    mask_gr = (pos[:, 2] <= ball_r)
    if np.any(mask_gr):
        n = np.array([[0, 0, 1]])
        dv, dw, hit = apply_friction(vel[mask_gr], omg[mask_gr], n, ball_e1, mu_ground)
        indices = np.where(mask_gr)[0][hit]
        vel[indices] += dv[hit]
        omg[indices] += dw[hit]
        
    return vel, omg, last_rim_pt

def step_rk4(pos, vel, omega, dt):
    """ Performs a single 4th-order Runge-Kutta integration step. """
    def f(v): return get_acceleration(v, omega)[0]
    k1_v, k1_p = f(vel), vel
    k2_v, k2_p = f(vel + k1_v * dt / 2), vel + k1_v * dt / 2
    k3_v, k3_p = f(vel + k2_v * dt / 2), vel + k2_v * dt / 2
    k4_v, k4_p = f(vel + k3_v * dt), vel + k3_v * dt
    
    p_new = pos + (k1_p + 2*k2_p + 2*k3_p + k4_p) * dt / 6
    v_new = vel + (k1_v + 2*k2_v + 2*k3_v + k4_v) * dt / 6
    return p_new, v_new
