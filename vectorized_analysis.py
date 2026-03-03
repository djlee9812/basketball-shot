import numpy as np
import time
import matplotlib.pyplot as plt
from constants import *
import engine

class VectorizedSimulator:
    """
    A basketball physics simulator that uses NumPy vectorization
    to simulate thousands of shots simultaneously.
    """
    def __init__(self, speeds, phis, h, thetas, omegas, dt=0.002):
        """
        Initialize the simulator with grids of shooting parameters.
        
        Args:
            speeds (ndarray): 1D array of launch speeds [ft/s].
            phis (ndarray): 1D array of vertical launch angles [deg].
            h (float): Release height [ft].
            thetas (ndarray): 2D grid of side angle deviations [deg].
            omegas (ndarray): 2D grid of backspin values [rev/s].
            dt (float): Integrator timestep [s].
        """
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
        
        # Memory for rim hits (prevents micro-jitters)
        self.last_rim_pt = np.full((self.N, 3), np.inf)

    def step(self):
        """
        Advance the simulation by one timestep for all active balls.
        Handles collisions, integration, and scoring detection.
        """
        idx = self.active
        if not np.any(idx): return
        
        # 1. Physics Step via Engine
        p, v, o = self.pos[idx], self.vel[idx], self.omg[idx]
        
        v, o, self.last_rim_pt[idx] = engine.resolve_collisions(
            p, v, o, self.last_rim_pt[idx]
        )
        
        p_new, v_new = engine.step_rk4(p, v, o, self.dt)
        
        # 2. Scoring Logic (Exact Parity)
        passed_plane = (p[:, 2] >= 10.0) & (p_new[:, 2] < 10.0)
        if np.any(passed_plane):
            f = (p[passed_plane, 2] - 10.0) / (p[passed_plane, 2] - p_new[passed_plane, 2] + 1e-10)
            x_rim = p[passed_plane, 0] + f * (p_new[passed_plane, 0] - p[passed_plane, 0])
            y_rim = p[passed_plane, 1] + f * (p_new[passed_plane, 1] - p[passed_plane, 1])
            inside = (x_rim**2 + y_rim**2) < rim_r**2
            active_indices = np.where(idx)[0]
            self.scored[active_indices[passed_plane]] |= inside
        
        # 3. Update State
        self.pos[idx], self.vel[idx], self.omg[idx] = p_new, v_new, o
        
        # Deactivate
        self.active[idx] &= (p_new[:, 2] > -1) & (p_new[:, 0] > -5) & (p_new[:, 0] < 45) & (np.abs(p_new[:, 1]) < court_w/2)

def run_analysis(nx, ny, save=False):
    """
    Executes a parameter sweep over launch speeds and angles,
    generates a success map, and optionally saves results.
    
    Args:
        nx (int): Number of speed points in the grid.
        ny (int): Number of angle points in the grid.
        save (bool): If True, saves speeds, angles, and score map to analysis_results.npz.
    """
    speeds = np.linspace(25, 31, nx)
    phis = np.linspace(37, 68, ny)
    sim = VectorizedSimulator(speeds, phis, 6.0, np.zeros((nx, ny)), 5 * np.ones((nx, ny)))
    
    t0 = time.time()
    for _ in range(int(sim_duration/sim.dt)):
        if not np.any(sim.active): break
        sim.step()
    t1 = time.time()
    
    print(f"Vectorized simulation: {t1-t0:.4f}s for {sim.N} shots.")
    print(f"Efficiency: {sim.N / (t1-t0):.1f} shots/sec")
    
    # Reshape scores back to the 2D grid (y-axis is phis, x-axis is speeds)
    scored_map = sim.scored.reshape(ny, nx)
    
    if save:
        np.savez("analysis_results.npz", speeds=speeds, phis=phis, scored=scored_map)
        print("Results saved to analysis_results.npz")
    
    plt.figure(figsize=(8,6))
    X, Y = np.meshgrid(speeds, phis)
    plt.pcolormesh(X, Y, scored_map, cmap='magma', shading='auto')
    plt.colorbar(label='Scored')
    plt.xlabel("Speeds [ft/s]")
    plt.ylabel("Launch Angle [deg]")
    plt.title("Shot Success Map")
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Vectorized Basketball Analysis")
    parser.add_argument("-nx", type=int, default=100, help="Grid resolution for speeds (default: 100)")
    parser.add_argument("-ny", type=int, default=100, help="Grid resolution for angles (default: 100)")
    parser.add_argument("--save", action="store_true", help="Save results to analysis_results.npz")
    args = parser.parse_args()
    
    run_analysis(nx=args.nx, ny=args.ny, save=args.save)
