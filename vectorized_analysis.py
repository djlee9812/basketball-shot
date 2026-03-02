import numpy as np
import time
import matplotlib.pyplot as plt
from constants import *
import engine

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
        v_dir = self.vel / np.maximum(np.linalg.norm(self.vel, axis=1)[:, None], 1e-6)
        omg_dir = np.cross(v_dir, [0, 0, 1])
        omg_norm = np.linalg.norm(omg_dir, axis=1)
        valid = omg_norm > 0
        omg_dir[valid] /= omg_norm[valid][:, None]
        self.omg = omg_dir * (2 * np.pi * omegas.flatten()[:, None])
        
        # State tracking
        self.active = np.ones(self.N, dtype=bool)
        self.scored = np.zeros(self.N, dtype=bool)
        
        # Cooldowns
        self.bb_cd = np.zeros(self.N, dtype=int)
        self.rim_cd = np.zeros(self.N, dtype=int)
        self.gr_cd = np.zeros(self.N, dtype=int)
        self.last_rim_pt = np.full((self.N, 3), np.inf)

    def step(self):
        idx = self.active
        if not np.any(idx): return
        
        # 1. Physics Step via Engine
        p, v, o = self.pos[idx], self.vel[idx], self.omg[idx]
        
        v, h_bb, h_rim, h_conn, h_gr, r_pts = engine.resolve_collisions(
            p, v, self.bb_cd[idx], self.rim_cd[idx], self.gr_cd[idx], self.last_rim_pt[idx]
        )
        
        # Update cooldowns and rim points based on hits
        active_indices = np.where(idx)[0]
        self.bb_cd[active_indices[h_bb]] = 5
        self.rim_cd[active_indices[h_conn]] = 10
        self.gr_cd[active_indices[h_gr]] = 5
        self.last_rim_pt[active_indices[h_rim]] = r_pts[h_rim]
        
        p_new, v_new = engine.step_rk4(p, v, o, self.dt)
        
        # 2. Scoring Logic (Exact Parity)
        passed_plane = (p[:, 2] >= 10.0) & (p_new[:, 2] < 10.0)
        if np.any(passed_plane):
            f = (p[passed_plane, 2] - 10.0) / (p[passed_plane, 2] - p_new[passed_plane, 2] + 1e-10)
            x_rim = p[passed_plane, 0] + f * (p_new[passed_plane, 0] - p[passed_plane, 0])
            y_rim = p[passed_plane, 1] + f * (p_new[passed_plane, 1] - p[passed_plane, 1])
            inside = (x_rim**2 + y_rim**2) < rim_r**2
            self.scored[active_indices[passed_plane]] |= inside
        
        # 3. Update State
        self.pos[idx], self.vel[idx] = p_new, v_new
        self.bb_cd[idx] = np.maximum(0, self.bb_cd[idx] - 1)
        self.rim_cd[idx] = np.maximum(0, self.rim_cd[idx] - 1)
        self.gr_cd[idx] = np.maximum(0, self.gr_cd[idx] - 1)
        
        # Deactivate
        self.active[idx] &= (p_new[:, 2] > -1) & (p_new[:, 0] > -5) & (p_new[:, 0] < 45) & (np.abs(p_new[:, 1]) < court_w/2)

def run_analysis(save=False):
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
    
    scored_map = sim.scored.reshape(100, 100)
    
    if save:
        np.savez("analysis_results.npz", speeds=speeds, phis=phis, scored=scored_map)
        print("Results saved to analysis_results.npz")
    
    plt.figure(figsize=(8,6))
    X, Y = np.meshgrid(speeds, phis)
    plt.pcolormesh(X, Y, scored_map, cmap='magma', shading='auto')
    plt.colorbar(label='Scored')
    plt.xlabel("Speeds [ft/s]")
    plt.ylabel("Launch Angle [deg]")
    plt.title("Vectorized Success Map")
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Vectorized Basketball Analysis")
    parser.add_argument("--save", action="store_true", help="Save results to analysis_results.npz")
    args = parser.parse_args()
    
    run_analysis(save=args.save)
