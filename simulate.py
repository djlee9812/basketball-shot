import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from constants import *
import engine

plt.style.use('seaborn-darkgrid')

# Sources:
# https://www.sciencedirect.com/science/article/pii/S1877705810003991
# http://www.physics.usyd.edu.au/~cross/Gripslip.pdf

class Ball:
    def __init__(self, x, y, z, v, phi, theta, omega_rev):
        """ Initialize a basketball shot and show simulation results
        :params x: Distance in front of rim [ft]
        :params y: Distance to the right of rim while facing it [ft]
        :params z: Distance from ground / shot height [ft]
        :params v: Shot launch speed [ft/s]
        :params phi: Shot launch angle from the horizontal [deg]
        :params theta: Shot angle deviation to the side [deg]
        :params omega: Backspin [revolution/second]
        """
        # Initial position (1, 3)
        self.pos = np.array([[x, y, z]])
        
        # Initial velocity (1, 3)
        phi_rad, theta_rad = np.radians(phi), np.radians(theta)
        vx = -v * np.cos(phi_rad) * np.cos(theta_rad)
        vy = -v * np.cos(phi_rad) * np.sin(theta_rad)
        vz = v * np.sin(phi_rad)
        self.vel = np.array([[vx, vy, vz]])

        # Initial omega (1, 3)
        v_norm = np.linalg.norm(self.vel)
        v_dir = self.vel / v_norm if v_norm > 0 else np.array([[1, 0, 0]])
        omg_dir = np.cross(v_dir, [0, 0, 1])
        omg_norm = np.linalg.norm(omg_dir)
        if omg_norm > 0:
            omg_dir /= omg_norm
        self.omg = omg_dir * (2 * np.pi * omega_rev)

        self.states = []
        self.data = []
        self.score = False
        self.end = False
        
        # Cooldowns (1-element arrays for engine)
        self.bb_cd = np.zeros(1, dtype=int)
        self.rim_cd = np.zeros(1, dtype=int)
        self.gr_cd = np.zeros(1, dtype=int)
        self.last_rim_pt = np.full((1, 3), np.inf)

        self.shot()

    def shot(self):
        """
        Simulate a shot from (x,y,z) with speed v, vertical launch angle phi,
        horizontal launch deviation theta
        """
        nit = 0
        # Terminate either when out of bounds or 10 seconds passed
        while not self.end and nit < 3/timestep:
            # Save state
            self.states.append(np.concatenate([self.pos[0], self.vel[0], self.omg[0]]))
            
            # Physics Step (Engine)
            self.vel, h_bb, h_rim, h_conn, h_gr, r_pts = engine.resolve_collisions(
                self.pos, self.vel, self.bb_cd, self.rim_cd, self.gr_cd, self.last_rim_pt
            )
            
            # Commit hits to cooldowns
            if h_bb[0]: self.bb_cd[0] = 5
            if h_conn[0]: self.rim_cd[0] = 10
            if h_gr[0]: self.gr_cd[0] = 5
            if h_rim[0]: self.last_rim_pt[0] = r_pts[0]
            
            # Log forces for debug mode (using current k1 state)
            accel, debug_info = engine.get_acceleration(self.vel, self.omg)
            CD, Fd, Re, Sp, CL, Fm = debug_info
            # self.data expects: Fg, CD, Fd, Re, Sp, CL, Fmx, Fmy, Fmz
            self.data.append([ball_m * g_eff, CD[0], Fd[0], Re[0], Sp[0], CL[0], Fm[0,0], Fm[0,1], Fm[0,2]])

            p_new, v_new = engine.step_rk4(self.pos, self.vel, self.omg, timestep)
            
            # Scoring & Termination logic
            self.check_scoring(self.pos[0], p_new[0])
            self.pos, self.vel = p_new, v_new
            self.bb_cd = np.maximum(0, self.bb_cd - 1)
            self.rim_cd = np.maximum(0, self.rim_cd - 1)
            self.gr_cd = np.maximum(0, self.gr_cd - 1)
            
            # Deactivate
            if self.pos[0, 2] < -1 or (not -5 < self.pos[0, 0] < 45) or np.abs(self.pos[0, 1]) > court_w/2:
                self.end = True
            nit += 1
            
        self.states = np.array(self.states)
        self.data = np.array(self.data)

    def check_scoring(self, p_prev, p_curr):
        if not self.score and p_prev[2] >= 10 and p_curr[2] < 10:
            frac = (p_prev[2] - 10) / (p_prev[2] - p_curr[2])
            x_rim = p_prev[0] + frac * (p_curr[0] - p_prev[0])
            y_rim = p_prev[1] + frac * (p_curr[1] - p_prev[1])
            if (x_rim**2 + y_rim**2) < rim_r**2:
                self.score = True

    def visualize(self, debug=False, animate=False):
        """ Visualize court and ball trajectory in pyplot.
        """
        import matplotlib.animation as animation
        
        x, y, z = self.states[:,0:3].T
        vx, vy, vz = self.states[:,3:6].T
        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(111, projection='3d')
        
        # Static Court Elements
        # Position of shooter
        position = Circle((x[0],y[0]),.5, color="gray", alpha=0.3)
        ax.add_patch(position)
        art3d.pathpatch_2d_to_3d(position)

        # Plot backboard and rim
        backboard = Rectangle((-bb_l/2, bb_z_bot), bb_l, bb_h, fill=False, linewidth=1)
        ax.add_patch(backboard)
        art3d.pathpatch_2d_to_3d(backboard, z=bb_x, zdir="x")
        rim = Circle((0, 0), rim_r, fill=False, edgecolor="red", linewidth=1)
        ax.add_patch(rim)
        art3d.pathpatch_2d_to_3d(rim, z=rim_h)

        # Plot backboard inside box
        bb_box = Rectangle((-1, bb_z_bot+.5), 2, 1.5, fill=False, linewidth=1)
        ax.add_patch(bb_box)
        art3d.pathpatch_2d_to_3d(bb_box, z=bb_x, zdir="x")

        # Plot rim connector to backboard
        rim_sq = Rectangle((bb_x, -.25), .5, .5, fill=True, color="red")
        ax.add_patch(rim_sq)
        art3d.pathpatch_2d_to_3d(rim_sq, z=10)

        # Plot half court sideline and baseline
        court = Rectangle((-4, -court_w/2), court_l/2, court_w, fill=False, linewidth=2)
        ax.add_patch(court)
        art3d.pathpatch_2d_to_3d(court, z=0)
        
        # Plot key box
        key = Rectangle((-4, -6), 19, 12, fill=False, linewidth=1)
        ax.add_patch(key)
        art3d.pathpatch_2d_to_3d(key, z=0)
        
        # Plot circle around free throw
        key_circle = Circle((15, 0), 6, fill=False, linewidth=1)
        ax.add_patch(key_circle)
        art3d.pathpatch_2d_to_3d(key_circle, z=0)
        
        # Plot 3 point line
        ax.plot([-4, 10], [22, 22], [0, 0], linewidth=1, color="black")
        ax.plot([-4, 10], [-22, -22], [0, 0], linewidth=1, color="black")
        
        # Plot 3 point arc
        # Arc radius is 23.75ft from the center of the rim (0,0)
        ys_3 = np.linspace(-22, 22, 100)
        xs_3 = np.sqrt(np.maximum(23.75**2 - ys_3**2, 0))
        ax.plot(xs_3, ys_3, np.zeros(len(ys_3)), linewidth=1, color="black")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(-5, 45)
        ax.set_ylim(-25, 25)
        ax.set_zlim(0, 50)
        
        if not animate:
            ax.plot(x, y, z, color="blue", alpha=0.7)
            ax.scatter(x[0], y[0], z[0], color="orange", s=100)
            plt.show()
        else:
            # Animation
            line, = ax.plot([], [], [], color="blue", alpha=0.7, lw=2)
            ball_pt, = ax.plot([], [], [], "o", color="orange", markersize=10)
            
            # Step size for animation (every 10 frames to speed up)
            step = 10
            
            def update(i):
                end_idx = i * step
                if end_idx >= len(x):
                    end_idx = len(x) - 1
                line.set_data(x[:end_idx], y[:end_idx])
                line.set_3d_properties(z[:end_idx])
                ball_pt.set_data(x[end_idx], y[end_idx])
                ball_pt.set_3d_properties(z[end_idx])
                return line, ball_pt
            
            num_frames = len(x) // step
            ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=20, blit=True)
            plt.show()

        # If debug True, plot time graphs of position and velocity
        if debug:
            # Reconstruct time array
            t = np.arange(len(x)) * timestep
            
            # Extract data from self.data (Fg, CD, Fd, Re, Sp, CL, Fmx, Fmy, Fmz, ...)
            data_arr = np.array(self.data)
            Fg, CD, Fd, Re, Sp, CL, Fmx, Fmy, Fmz = data_arr[:, 0:9].T
            
            fig_dbg, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8,10))
            fig_dbg.suptitle("Ball Physics Debug Variables")
            
            # Subplot 1: Position
            ax1.plot(t, x, label="x")
            ax1.plot(t, y, label="y")
            ax1.plot(t, z, label="z")
            ax1.set_ylabel("Position [ft]")
            ax1.legend()
            
            # Subplot 2: Velocity
            ax2.plot(t, vx, label=r"$v_x$")
            ax2.plot(t, vy, label=r"$v_y$")
            ax2.plot(t, vz, label=r"$v_z$")
            ax2.set_ylabel("Velocity [ft/s]")
            ax2.legend()
            
            # Subplot 3: Nondimensional Coefficients (Aerodynamics)
            # self.data is logged only on k1 steps, so it matches the length of states
            # (minus 1 if the simulation ended exactly on a step)
            t_data = t[:len(CD)] 
            ax3.plot(t_data, CD, label=r"$C_D$ (Drag)")
            ax3.plot(t_data, Sp, label=r"$Sp$ (Spin)")
            ax3.plot(t_data, CL, label=r"$C_L$ (Lift)")
            ax3.set_ylabel("Coefficients")
            ax3.set_xlabel("Time [sec]")
            ax3.legend()
            
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Basketball Shot Simulator")
    
    # Position and Velocity Arguments
    parser.add_argument("-x", type=float, default=15.0, help="Distance in front of rim [ft] (default: 15.0)")
    parser.add_argument("-y", type=float, default=0.0, help="Distance to the side of rim [ft] (default: 0.0)")
    parser.add_argument("-z", type=float, default=6.0, help="Shot release height [ft] (default: 6.0)")
    parser.add_argument("-v", "--speed", type=float, default=26.0, help="Launch speed [ft/s] (default: 26.0)")
    parser.add_argument("-a", "--angle", type=float, default=56.0, help="Vertical launch angle [deg] (default: 56.0)")
    parser.add_argument("-s", "--side", type=float, default=0.0, help="Side angle deviation [deg] (default: 0.0)")
    parser.add_argument("-w", "--spin", type=float, default=5.0, help="Backspin [rev/s] (default: 5.0)")
    
    # Visuals
    parser.add_argument("--debug", action="store_true", help="Show detailed physics debug plots")
    parser.add_argument("--animate", action="store_true", help="Animate the shot trajectory")
    
    args = parser.parse_args()

    # Initialize ball object
    ball = Ball(args.x, args.y, args.z, args.speed, args.angle, args.side, args.spin)
    
    print("-------------------------")
    print(f"Result: {'SCORE!' if ball.score else 'MISSED'}")
    print("-------------------------")
    
    ball.visualize(debug=args.debug, animate=args.animate)
