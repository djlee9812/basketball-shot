import numpy as np
import time
import matplotlib.pyplot as plt
from constants import *
import engine

class VectorizedSimulator:
    """
    Physics driver for simulating many shots in parallel.
    Uses NumPy arrays to process all balls in a single vectorized step.
    """
    def __init__(self, speeds, phis, h, thetas, omegas, dt=timestep):
        """
        Initializes the state for all balls in the grid.
        
        Args:
            speeds (ndarray): 1D array of launch speeds [ft/s].
            phis (ndarray): 1D array of launch angles [deg].
            h (float): Release height [ft].
            thetas (ndarray): 2D grid of side deviation angles [deg].
            omegas (ndarray): 2D grid of backspin values [rev/s].
            dt (float): Timestep [sec].
        """
        self.dt = dt
        self.N = speeds.size * phis.size
        S, P = np.meshgrid(speeds, phis)
        
        # Position (N, 3)
        self.pos = np.zeros((self.N, 3))
        self.pos[:, 0] = 15.0 # Fixed X starting position
        self.pos[:, 2] = h    # Fixed height
        
        # Initial velocity (N, 3)
        phi_rad = np.radians(P.flatten())
        theta_rad = np.radians(thetas.flatten())
        v_mag = S.flatten()
        
        self.vel = np.zeros((self.N, 3))
        self.vel[:, 0] = -v_mag * np.cos(phi_rad) * np.cos(theta_rad)
        self.vel[:, 1] = -v_mag * np.cos(phi_rad) * np.sin(theta_rad)
        self.vel[:, 2] = v_mag * np.sin(phi_rad)
        
        # Initial angular velocity (N, 3)
        v_norm = np.maximum(np.linalg.norm(self.vel, axis=1)[:, None], 1e-6)
        v_dir = self.vel / v_norm
        omg_dir = np.cross(v_dir, [0, 0, 1])
        omg_norm = np.linalg.norm(omg_dir, axis=1)[:, None]
        valid = omg_norm.flatten() > 0
        omg_dir[valid] /= omg_norm[valid]
        self.omg = omg_dir * (2 * np.pi * omegas.flatten()[:, None])
        
        self.active = np.ones(self.N, dtype=bool)
        self.scored = np.zeros(self.N, dtype=bool)
        self.last_rim_pt = np.full((self.N, 3), np.inf)

    def step(self):
        """ Advances all active balls by one physics step. """
        idx = self.active
        if not np.any(idx): return
        
        # 1. Collision Handling
        p, v, o = self.pos[idx], self.vel[idx], self.omg[idx]
        v, o, self.last_rim_pt[idx] = engine.resolve_collisions(p, v, o, self.last_rim_pt[idx])
        
        # 2. Continuous Integration (RK4)
        p_new, v_new = engine.step_rk4(p, v, o, self.dt)
        
        # 3. Scoring Detection (Rim Plane Crossing)
        passed_plane = (p[:, 2] >= 10.0) & (p_new[:, 2] < 10.0)
        if np.any(passed_plane):
            f = (p[passed_plane, 2] - 10.0) / (p[passed_plane, 2] - p_new[passed_plane, 2] + 1e-10)
            x_rim = p[passed_plane, 0] + f * (p_new[passed_plane, 0] - p[passed_plane, 0])
            y_rim = p[passed_plane, 1] + f * (p_new[passed_plane, 1] - p[passed_plane, 1])
            inside = (x_rim**2 + y_rim**2) < rim_r**2
            self.scored[np.where(idx)[0][passed_plane]] |= inside
        
        # 4. State Committal & Deactivation
        self.pos[idx], self.vel[idx], self.omg[idx] = p_new, v_new, o
        self.active[idx] &= (p_new[:, 2] > -1) & (p_new[:, 0] > -5) & (p_new[:, 0] < 45) & (np.abs(p_new[:, 1]) < court_w/2)

def run_analysis(nx, ny, save=False, plot=True):
    """ Executes a parameter sweep and generates a success map. """
    speeds = np.linspace(25, 31, nx)
    phis = np.linspace(37, 68, ny)
    # Dummy grids for initialization
    thetas = np.zeros((ny, nx))
    omegas = 5 * np.ones((ny, nx))
    
    sim = VectorizedSimulator(speeds, phis, 6.0, thetas, omegas)
    
    t0 = time.time()
    max_iters = int(sim_duration / timestep)
    for _ in range(max_iters):
        if not np.any(sim.active): break
        sim.step()
    t1 = time.time()
    
    print(f"Vectorized simulation: {t1-t0:.4f}s for {sim.N} shots.")
    print(f"Efficiency: {sim.N / (t1-t0):.1f} shots/sec")
    
    scored_map = sim.scored.reshape(ny, nx)
    
    if save:
        np.savez("analysis_results.npz", speeds=speeds, phis=phis, scored=scored_map)
        print("Results saved to analysis_results.npz")
    
    if plot:
        plt.figure(figsize=(8,6))
        X, Y = np.meshgrid(speeds, phis)
        plt.pcolormesh(X, Y, scored_map, cmap='magma', shading='auto')
        plt.colorbar(label='Scored')
        plt.xlabel("Launch Speed [ft/s]")
        plt.ylabel("Launch Angle [deg]")
        plt.title("Shot Success Map")
        plt.show()
    
    return speeds, phis, scored_map

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Vectorized Basketball Analysis")
    parser.add_argument("-nx", type=int, default=100, help="Speed resolution")
    parser.add_argument("-ny", type=int, default=100, help="Angle resolution")
    parser.add_argument("--save", action="store_true", help="Save to .npz")
    args = parser.parse_args()
    
    run_analysis(nx=args.nx, ny=args.ny, save=args.save)
