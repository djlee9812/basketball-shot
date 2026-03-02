import numpy as np
import time
import matplotlib.pyplot as plt
from constants import *

class VectorizedSimulator:
    def __init__(self, speeds, phis, h, thetas, omegas, dt=0.002):
        self.dt = dt
        S, P = np.meshgrid(speeds, phis)
        self.N = S.size
        
        # Position (N, 3)
        self.pos = np.zeros((self.N, 3))
        self.pos[:, 0] = 15.0
        self.pos[:, 2] = h
        
        # Velocity (N, 3)
        phi_rad = np.radians(P.flatten())
        theta_rad = np.radians(thetas.flatten())
        v_mag = S.flatten()
        
        self.vel = np.zeros((self.N, 3))
        self.vel[:, 0] = -v_mag * np.cos(phi_rad) * np.cos(theta_rad)
        self.vel[:, 1] = -v_mag * np.cos(phi_rad) * np.sin(theta_rad)
        self.vel[:, 2] = v_mag * np.sin(phi_rad)
        
        # Omega (N, 3)
        v_dir = self.vel / np.linalg.norm(self.vel, axis=1)[:, None]
        up = np.array([0, 0, 1])
        omg_dir = np.cross(v_dir, up)
        omg_norm = np.linalg.norm(omg_dir, axis=1)
        valid_omg = omg_norm > 0
        omg_dir[valid_omg] /= omg_norm[valid_omg][:, None]
        self.omg = omg_dir * (2 * np.pi * omegas.flatten()[:, None])
        
        # State tracking
        self.active = np.ones(self.N, dtype=bool)
        self.scored = np.zeros(self.N, dtype=bool)
        
        # Collision Queues (using counts instead of arrays for performance)
        self.bb_cooldown = np.zeros(self.N, dtype=int)
        self.rimsq_cooldown = np.zeros(self.N, dtype=int)
        self.ground_cooldown = np.zeros(self.N, dtype=int)
        self.last_rim_pt = np.full((self.N, 3), np.inf)

    def get_accel(self, vel, active_mask):
        """ Continuous forces: Gravity, Drag, Magnus """
        n_active = np.sum(active_mask)
        accel = np.zeros((n_active, 3))
        speed = np.linalg.norm(vel, axis=1)
        safe_speed = np.maximum(speed, 1e-6)
        
        # Gravity
        accel[:, 2] -= g_eff
        
        # Drag (Re-dependent)
        Re = rho / g * speed * 2 * ball_r / mu
        CD = np.full(n_active, ball_cd1)
        crisis = (Re > 1e5) & (Re < 2e5)
        CD[crisis] = ball_cd1 + (Re[crisis] - 1e5) * (ball_cd2 - ball_cd1) / 1e5
        CD[Re >= 2e5] = ball_cd2
        Fd_mag = 0.5 * CD * rho * speed**2 * (np.pi * ball_r**2)
        accel -= (Fd_mag / (ball_m * safe_speed))[:, None] * vel
        
        # Magnus Force
        active_omg = self.omg[active_mask]
        omega_mag = np.linalg.norm(active_omg, axis=1)
        Sp = np.clip(omega_mag * ball_r / safe_speed, 0, 3)
        CL = Sp * 0.77 + 0.12
        Fm = 0.5 * CL[:, None] * np.pi * ball_r**3 * rho * np.cross(active_omg, vel)
        accel += Fm / ball_m
        
        return accel

    def apply_collisions(self, active_mask):
        idx = active_mask
        p = self.pos[idx]
        v = self.vel[idx]
        
        # 1. Backboard
        dx_bb = p[:, 0] - bb_x
        dy_bb = np.maximum(0, np.maximum(-bb_l/2 - p[:, 1], p[:, 1] - bb_l/2))
        dz_bb = np.maximum(0, np.maximum(bb_z_bot - p[:, 2], p[:, 2] - bb_z_top))
        dist_bb = np.sqrt(dx_bb**2 + dy_bb**2 + dz_bb**2)
        
        hit_bb = (dist_bb < ball_r) & (self.bb_cooldown[idx] == 0)
        if np.any(hit_bb):
            # Straight on backboard check
            straight_hit = hit_bb & (p[:, 2] > bb_z_bot) & (p[:, 2] < bb_z_top) & (np.abs(p[:, 1]) < bb_l/2)
            v[straight_hit, 0] = -v[straight_hit, 0] * ball_e2
            
            # Corner/Edge bounce
            edge_hit = hit_bb & ~straight_hit
            if np.any(edge_hit):
                ypt = np.clip(p[edge_hit, 1], -bb_l/2, bb_l/2)
                zpt = np.clip(p[edge_hit, 2], bb_z_bot, bb_z_top)
                normvec = p[edge_hit] - np.column_stack([np.full(np.sum(edge_hit), bb_x), ypt, zpt])
                n_mag = np.linalg.norm(normvec, axis=1)
                valid = n_mag > 0
                normvec[valid] /= n_mag[valid][:, None]
                v_dot_n = np.sum(v[edge_hit] * normvec, axis=1)
                v[edge_hit] -= (1 + ball_e2) * v_dot_n[:, None] * normvec
            
            self.bb_cooldown[idx] = np.where(hit_bb, 5, self.bb_cooldown[idx])

        # 2. Rim
        planar_dist = np.linalg.norm(p[:, :2], axis=1)
        dist_to_rim = np.sqrt((planar_dist - rim_r)**2 + (p[:, 2] - 10)**2)
        
        # Calculate rim point
        scale = rim_r / np.maximum(planar_dist, 1e-6)
        rim_pts = np.column_stack([p[:, 0] * scale, p[:, 1] * scale, np.full(len(p), 10.0)])
        
        # Match simulate.py "Requires new rim point" logic
        not_recent_pt = np.linalg.norm(rim_pts - self.last_rim_pt[idx], axis=1) > 0.5
        hit_rim = (dist_to_rim < ball_r) & (p[:, 0] > -rim_r) & not_recent_pt & (self.rimsq_cooldown[idx] == 0)
        
        if np.any(hit_rim):
            normvec = p[hit_rim] - rim_pts[hit_rim]
            n_mag = np.linalg.norm(normvec, axis=1)
            normvec[n_mag > 0] /= n_mag[n_mag > 0][:, None]
            v_dot_n = np.sum(v[hit_rim] * normvec, axis=1)
            v[hit_rim] -= (1 + ball_e2) * v_dot_n[:, None] * normvec
            self.last_rim_pt[np.where(idx)[0][hit_rim]] = rim_pts[hit_rim]

        # 3. Rim Connector
        hit_conn = (p[:, 0] > bb_x) & (p[:, 0] < -rim_r) & (np.abs(p[:, 1]) < 0.25) & \
                   (p[:, 2] > 10) & (p[:, 2] < 10 + ball_r) & (self.rimsq_cooldown[idx] == 0)
        if np.any(hit_conn):
            v[hit_conn, 2] = -v[hit_conn, 2] * ball_e2
            self.rimsq_cooldown[idx] = np.where(hit_conn, 10, self.rimsq_cooldown[idx])

        # 4. Ground
        hit_ground = (p[:, 2] <= ball_r) & (self.ground_cooldown[idx] == 0)
        if np.any(hit_ground):
            v[hit_ground, 2] = -v[hit_ground, 2] * ball_e1
            self.ground_cooldown[idx] = np.where(hit_ground, 5, self.ground_cooldown[idx])

        self.vel[idx] = v

    def step(self):
        idx = self.active
        if not np.any(idx): return
        
        # 1. Apply Discrete Collisions
        self.apply_collisions(idx)
        
        p = self.pos[idx]
        v = self.vel[idx]
        
        # 2. RK4 Continuous Integration
        k1_p = v
        k1_v = self.get_accel(v, idx)
        
        k2_p = v + k1_v * self.dt/2
        k2_v = self.get_accel(v + k1_v * self.dt/2, idx)
        
        k3_p = v + k2_v * self.dt/2
        k3_v = self.get_accel(v + k2_v * self.dt/2, idx)
        
        k4_p = v + k3_v * self.dt
        k4_v = self.get_accel(v + k3_v * self.dt, idx)
        
        p_new = p + (k1_p + 2*k2_p + 2*k3_p + k4_p) * self.dt / 6
        v_new = v + (k1_v + 2*k2_v + 2*k3_v + k4_v) * self.dt / 6
        
        # 3. Exact Linear Interp Scoring Logic
        passed_plane = (p[:, 2] >= 10.0) & (p_new[:, 2] < 10.0)
        if np.any(passed_plane):
            f = (p[passed_plane, 2] - 10.0) / (p[passed_plane, 2] - p_new[passed_plane, 2] + 1e-10)
            x_at_rim = p[passed_plane, 0] + f * (p_new[passed_plane, 0] - p[passed_plane, 0])
            y_at_rim = p[passed_plane, 1] + f * (p_new[passed_plane, 1] - p[passed_plane, 1])
            inside = (x_at_rim**2 + y_at_rim**2) < rim_r**2
            self.scored[np.where(idx)[0][passed_plane]] |= inside
        
        # Update state
        self.pos[idx] = p_new
        self.vel[idx] = v_new
        self.bb_cooldown[idx] = np.maximum(0, self.bb_cooldown[idx] - 1)
        self.rimsq_cooldown[idx] = np.maximum(0, self.rimsq_cooldown[idx] - 1)
        self.ground_cooldown[idx] = np.maximum(0, self.ground_cooldown[idx] - 1)
        
        # Deactivate
        self.active[idx] &= (p_new[:, 2] > -1) & (p_new[:, 0] > -5) & (p_new[:, 0] < 50) & (np.abs(p_new[:, 1]) < court_w/2)

def run_analysis():
    speeds = np.linspace(23, 34, 100)
    phis = np.linspace(35, 70, 100)
    sim = VectorizedSimulator(speeds, phis, 6.0, np.zeros((100,100)), 5 * np.ones((100,100)))
    
    t0 = time.time()
    for _ in range(2500):
        if not np.any(sim.active): break
        sim.step()
    t1 = time.time()
    
    print(f"Vectorized simulation: {t1-t0:.4f}s for {sim.N} shots.")
    print(f"Efficiency: {sim.N / (t1-t0):.1f} shots/sec")
    
    plt.figure(figsize=(8,6))
    X, Y = np.meshgrid(speeds, phis)
    plt.pcolormesh(X, Y, sim.scored.reshape(100, 100), cmap='magma', shading='auto')
    plt.colorbar(label='Scored')
    plt.xlabel("Speeds [ft/s]")
    plt.ylabel("Launch Angle [deg]")
    plt.title("Vectorized Success Map")
    plt.show()

if __name__ == "__main__":
    run_analysis()
