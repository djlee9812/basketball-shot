import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from constants import *

plt.style.use('seaborn-darkgrid')

# Sources:
# https://www.sciencedirect.com/science/article/pii/S1877705810003991
# http://www.physics.usyd.edu.au/~cross/Gripslip.pdf

class Ball:
    def __init__(self, x, y, z, v, phi, theta, omega):
        """ Initialize a basketball shot and show simulation results
        :params x: Distance in front of rim [ft]
        :params y: Distance to the right of rim while facing it [ft]
        :params z: Distance from ground / shot height [ft]
        :params v: Shot launch speed [ft/s]
        :params phi: Shot launch angle from the horizontal [deg]
        :params theta: Shot angle deviation to the side [deg]
        :params omega: Backspin [revolution/second]
        """
        # Units: [ft], [ft/s], [rad], [rad], [rad/s]
        self.initial = {"position": [x,y,z],
                       "speed": v,
                       "phi": np.radians(phi),
                       "theta": np.radians(theta),
                       "omega": 2 * np.pi * omega}
        self.states = []
        self.data = []
        self.score = False
        self.end = False
        # Queue for recent collisions with ground / backboard (avoid oscillations)
        # If collision within last t time steps, don't collide again
        self.ground = np.zeros(5)
        self.rimpt = np.zeros(3)
        self.rimsq = np.zeros(10)
        self.bb = np.zeros(5)
        # Shoot after initialization
        self.shot()

    def shot(self):
        """
        Simulate a shot from (x,y,z) with speed v, vertical launch angle phi,
        horizontal launch deviation theta
        """
        # Get initial state in global hoop frame
        x, y, z = self.initial["position"]
        # theta0 = angle to goal
        theta0 = np.arctan2(y, x)
        # theta = angle includingn launch deviation
        theta = theta0 - self.initial["theta"]
        vx = -self.initial["speed"] * np.cos(self.initial["phi"]) * np.cos(theta)
        vy = -self.initial["speed"] * np.cos(self.initial["phi"]) * np.sin(theta)
        vz = self.initial["speed"] * np.sin(self.initial["phi"])

        omega = self.initial["omega"]
        # Omega for backspin - to the right in body frame (v x z)
        omega_dir = np.cross([vx, vy, vz], [0, 0, 1])
        omega_dir_norm = np.linalg.norm(omega_dir)
        if omega_dir_norm > 0:
            omega_dir /= omega_dir_norm
        else:
            omega_dir = np.array([0, 0, 0])
        wx, wy, wz = omega * omega_dir
        
        current_state = [x, y, z, vx, vy, vz, wx, wy, wz]
        self.states.append(current_state)
        
        nit = 0
        # Terminate either when out of bounds or 10 seconds passed
        while not self.end and nit < 3/timestep:
            # 1. Handle Discrete Collisions (Impulse Changes)
            # This directly modifies the current_state velocity if a collision is detected
            current_state = self.apply_collisions(current_state)
            
            # 2. Continuous Integration (RK4 for smooth forces: Gravity, Drag, Magnus)
            state = np.array(current_state)
            k1 = np.array(self.dynamics(state))
            k2 = np.array(self.dynamics(state + k1 * timestep / 2))
            k3 = np.array(self.dynamics(state + k2 * timestep / 2))
            k4 = np.array(self.dynamics(state + k3 * timestep))
            
            new_state = state + (k1 + 2*k2 + 2*k3 + k4) * timestep / 6
            current_state = new_state.tolist()
            self.states.append(current_state)
            
            # 3. Update collision queues and check end conditions
            self.update_queues()
            self.check_end()
            nit += 1
            
        self.states = np.array(self.states)
        self.data = np.array(self.data)

    def apply_collisions(self, state):
        """ Check for and apply instantaneous velocity changes from collisions.
        Returns the modified state.
        """
        x, y, z, vx, vy, vz, wx, wy, wz = state
        v = np.array([vx, vy, vz])
        
        # Backboard collision
        if self.dist_to_bb(state) < ball_r and np.count_nonzero(self.bb) == 0:
            # Straight on backboard
            if bb_z_bot < z < bb_z_top and -bb_l/2 < y < bb_l/2:
                vx = -vx * ball_e2
            # Edge of backboard
            else:
                ypt = max(-bb_l/2, min(bb_l/2, y))
                zpt = max(bb_z_bot, min(bb_z_top, z))
                normvec = np.array([x,y,z]) - np.array([bb_x, ypt, zpt])
                normvec /= np.linalg.norm(normvec)
                v = v - (1 + ball_e2) * np.dot(v, normvec) * normvec
                vx, vy, vz = v
            self.bb[-1] = 1

        # Rim collision
        rim_dist, rim_pt = self.dist_to_rim(state)
        # New rim point and no recent collision with rim connector
        if rim_dist < ball_r and (x > -rim_r or np.abs(y) > .25) \
           and not np.allclose(rim_pt, self.rimpt, atol=0.5) and np.count_nonzero(self.rimsq) == 0:
            normvec = np.array([x,y,z]) - np.array(rim_pt)
            normvec /= np.linalg.norm(normvec)
            v = v - (1 + ball_e2) * np.dot(v, normvec) * normvec
            vx, vy, vz = v
            self.rimpt = rim_pt

        # Rim connector collision
        if bb_x < x < -rim_r and np.abs(y) < .25 and 10 < z < 10 + ball_r \
           and np.count_nonzero(self.rimsq) == 0:
            vz = -vz * ball_e2
            self.rimsq[-1] = 1

        # Ground collision
        if z <= ball_r and np.count_nonzero(self.ground) == 0:
            vz = -vz * ball_e1
            self.ground[-1] = 1

        return [x, y, z, vx, vy, vz, wx, wy, wz]

    def update_queues(self):
        """ Shift collision queues once per full timestep.
        """
        self.bb = np.concatenate([self.bb[1:], [0]])
        self.rimsq = np.concatenate([self.rimsq[1:], [0]])
        self.ground = np.concatenate([self.ground[1:], [0]])

    def dynamics(self, state):
        """ Return time derivative of state variables for continuous forces.
        """
        x, y, z, vx, vy, vz, wx, wy, wz = state
        dx, dy, dz = (vx, vy, vz)
        speed = np.linalg.norm([vx, vy, vz])
        if speed == 0:
            return [0, 0, 0, 0, 0, 0, 0, 0, 0]

        # Continuous forces
        Fg = ball_m * g_eff
        CD, Re = self.get_cd(speed)
        Fd = 0.5 * CD * rho * speed**2 * (np.pi*ball_r**2)
        
        # Magnus force
        omega = np.linalg.norm([wx, wy, wz])
        Sp = max(0, min(omega * ball_r / speed, 3)) # Clip to [0,3]
        CL = Sp * 0.77 + 0.12
        Fmx, Fmy, Fmz = 0.5 * CL * np.pi * ball_r**3 * rho \
                        * np.cross([wx, wy, wz], [vx, vy, vz])

        Fx = -Fd * vx / speed + Fmx
        Fy = -Fd * vy / speed + Fmy
        Fz = -Fd * vz / speed - Fg + Fmz
        
        # Logging for k1 step (primary state)
        if np.array_equal(state, self.states[-1]):
            self.data.append([Fg, CD, Fd, Re, Sp, CL, Fmx, Fmy, Fmz, 0, 0, 0])

        return [dx, dy, dz, Fx/ball_m, Fy/ball_m, Fz/ball_m, 0, 0, 0]

    def get_cd(self, speed):
        """ Calculate drag coefficient of basketball including drag crisis effects
        between Re 1e5 and 2e5 where CD decreases linearly from ball_cd1 to ball_cd2
        Re = 1e5 corresponds to approximately 20ft/s
        """
        Re = rho / g * speed * 2 * ball_r / mu
        CD = ball_cd1
        if 1e5 < Re < 2e5:
            slope = (ball_cd2 - ball_cd1) / 1e5
            CD += (Re - 1e5) * slope
        elif Re >= 2e5:
            CD = ball_cd2
        return CD, Re

    def dist_to_bb(self, state):
        x, y, z = state[:3]
        dx = x - bb_x
        dy = np.max([-bb_l/2 - y, 0, y - bb_l/2])
        dz = np.max([bb_z_bot - z, 0, z - bb_z_top])
        return np.linalg.norm([dx, dy, dz])

    def dist_to_rim(self, state):
        x, y, z = state[:3]
        planar_d = np.linalg.norm([x,y]) - rim_r
        z_dist = z - 10
        dist = np.linalg.norm([planar_d, z_dist])
        scale = rim_r / max(1e-6, np.linalg.norm([x,y]))
        pt = [x * scale, y *scale, 10]
        return dist, pt

    def check_end(self):
        """ Termination conditions if out of bounds and check if scored.
        """
        x, y, z = self.states[-1][:3]
        if z < -1 or (not -5 < x < 45) or np.abs(y) > court_w/2:
            self.end = True
        
        if not self.score and len(self.states) > 1:
            prev_z = self.states[-2][2]
            curr_z = self.states[-1][2]
            if prev_z >= 10 and curr_z < 10:
                frac = (prev_z - 10) / (prev_z - curr_z)
                x_at_rim = self.states[-2][0] + frac * (self.states[-1][0] - self.states[-2][0])
                y_at_rim = self.states[-2][1] + frac * (self.states[-1][1] - self.states[-2][1])
                if np.linalg.norm([x_at_rim, y_at_rim]) < rim_r:
                    self.score = True

    def visualize(self, debug=False):
        """ Visualize court and ball trajectory in pyplot.
        """
        x, y, z = self.states[:,0:3].T
        vx, vy, vz = self.states[:,3:6].T
        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z)
        ax.scatter(x[0], y[0], z[0], color="orange", s=100)

        position = Circle((x[0],y[0]),.5, color="gray")
        ax.add_patch(position)
        art3d.pathpatch_2d_to_3d(position)

        # Plot court elements
        backboard = Rectangle((-bb_l/2, bb_z_bot), bb_l, bb_h, fill=False, linewidth=1)
        ax.add_patch(backboard)
        art3d.pathpatch_2d_to_3d(backboard, z=bb_x, zdir="x")
        rim = Circle((0, 0), rim_r, fill=False, edgecolor="red", linewidth=1)
        ax.add_patch(rim)
        art3d.pathpatch_2d_to_3d(rim, z=rim_h)
        
        court = Rectangle((-4, -court_w/2), court_l/2, court_w, fill=False, linewidth=2)
        ax.add_patch(court)
        art3d.pathpatch_2d_to_3d(court, z=0)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim(-5, 45)
        ax.set_ylim(-25, 25)
        ax.set_zlim(0, 50)
        plt.show()

if __name__ == "__main__":
    ball = Ball(15, 0, 6, 26, 56, 0, 5)
    print("Score!" if ball.score else "Missed!")
    ball.visualize()
