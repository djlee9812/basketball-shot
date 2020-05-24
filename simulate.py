import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
plt.style.use('seaborn')

# Constants (Imperial Units)
ft2m = 0.3048 # [m/ft]
m2ft = 1 / ft2m # [ft/m]
kg2lb = 2.20462 # [lb/kg]
g = 32.174 # 9.81 * m2ft # [ft/s^2]
g_eff = g * .985 # Effective gravity after 1.5% buoyancy force
rho = 0.0764 # 1.225 * kg2lb / m2ft**3 # Air Density [lb/ft^3]
dt = 0.002 # Integrator time step [sec]
# NBA court dimensions
court_l = 94
court_w = 50
# line distances from rim center
center_3 = 23.75
corner_3 = 22
corner_l = 10  # how to account for behind rim? (x<0) -> 14
free = 15
# rim - origin
rim_h = 10 # Rim height 10ft
rim_r = 1.5/2 # Rim diameter 1.5ft
rim_t = 0.02 * m2ft # Rim thicknenss 2cm
rim_bb_dist = 0.5 # Rim end to backboard distance 6 inches
# backboard
bb_l = 6 # Backboard width 6ft
bb_h = 3.5 # Backboard height 3.5ft
bb_z_bot = 9 # Backboard bottom z coord
bb_z_top = bb_z_bot + bb_h # backboard top z coord
bb_x = -rim_r - rim_bb_dist # backboard x coordinate
# Ball
ball_r = 29.5 / 12 / (2 * np.pi) # 29.5" circumference
ball_m = 0.620 * kg2lb # Weight (567-650g - Wikipedia) [grams]
ball_e1 = 0.84 # Coefficient of Restitution - Ground (v2/v1)
ball_e2 = 0.65 # Coefficient of Restitution - Backboard/rim (v2/v1)
ball_cd = 0.47 # Coefficient of drag
ball_contact_t = 0.01 # Impact time on contact [seconds]

class Ball:
    def __init__(self, x, y, z, v, phi, theta):
        self.position = [x,y,z]
        self.speed = v
        self.phi = np.radians(phi)
        self.theta = np.radians(theta)
        self.states = []
        self.score = False
        self.end = False
        # Queue for recent collisions with ground / backboard (avoid oscillations)
        self.ground = np.zeros(5)
        self.bb = np.zeros(5)
        self.shot()

    def shot(self):
        """
        Simulate a shot from (x,y,z) with speed v, vertical launch angle phi,
        horizontal launch deviation theta
        """
        # Get initial state in global hoop frame
        x, y, z = self.position
        ratio = x/y if y != 0 else np.inf
        theta0 = np.arctan(ratio)
        theta = theta0 - self.theta
        vx = -self.speed * np.cos(self.phi) * np.sin(theta)
        vy = self.speed * np.cos(self.phi) * np.cos(theta)
        vz = self.speed * np.sin(self.phi)
        ics = [x, y, z, vx, vy, vz]
        # Save ICs in state vector
        self.states.append(ics)
        nit = 0
        while not self.end and nit < 2000:
            derivs = self.dynamics()
            self.integrate(derivs)
            self.check_end()
            nit += 1
        self.states = np.array(self.states)
        self.visualize()


    def dynamics(self):
        """ Return time derivative of state variables
        :params x: State variable vector
            position, velocity
        """
        x, y, z, vx, vy, vz = self.states[-1]
        # dx/dt = v
        dx, dy, dz = (vx, vy, vz)
        speed = np.linalg.norm([vx, vy, vz])
        # Gravity + buoyancy (z)
        Fg = -ball_m * g_eff
        # Drag (-v)
        Fd = 0.5 * ball_cd * rho * speed**2 * (np.pi*ball_r**2)

        # Magnus (omega x v)
        # Fm = 16 / 3 * np.pi**2 * ball_r**3 * rho * omega * v

        dvx = 0
        dvy = 0
        dvz = Fg/ball_m
        Fcoll = self.collision()
        if np.count_nonzero(Fcoll) > 0:
            dvx, dvy, dvz = Fcoll / ball_m

        return [dx, dy, dz, dvx, dvy, dvz]

    def collision(self):
        """ Calculate impulse force from collision
        """
        x, y, z, vx, vy, vz = self.states[-1]
        speed = np.linalg.norm([vx, vy, vz])
        delta_p = np.zeros(3)

        # Shift collision backboard, ground queues
        self.bb = np.concatenate([self.bb[1:], [0]])
        self.ground = np.concatenate([self.ground[1:], [0]])
        # If collision with backboard or ground in last 5 time steps: skip

        if self.dist_to_bb() < ball_r and np.count_nonzero(self.bb) == 0:
            # handle collision with backboard
            delta_p[0] += (1 + ball_e2) * (-vx * ball_m)
            self.bb[-1] = 1
            print("BB Bounce!")

        if self.dist_to_rim() < ball_r:
            # handle collision with rim
            pass

        if z <= ball_r and np.count_nonzero(self.ground) == 0:
            delta_p[2] += (1 + ball_e1) * (-vz * ball_m)
            self.ground[-1] = 1

        return delta_p / dt

    def dist_to_bb(self):
        """ Calculate projection distance from ball to backboard
        """
        x, y, z, vx, vy, vz = self.states[-1]
        dx = x - bb_x
        dy = np.max([-bb_l/2 - y, 0, y - bb_l/2])
        dz = np.max([bb_z_bot - z, 0, z - bb_z_top])
        return np.linalg.norm([dx, dy, dz])

    def dist_to_rim(self):
        """ Calculate projection distance from ball to rim
        """
        x, y, z, vx, vy, vz = self.states[-1]
        planar_d = np.linalg.norm([x,y]) - rim_r
        z_dist = z - 10
        return np.linalg.norm([planar_d, z_dist])

    def integrate(self, derivs):
        state = self.states[-1]
        new_state = [state[i] + derivs[i] * dt for i in range(len(state))]
        self.states.append(new_state)

    def check_end(self):
        x, y, z, vx, vy, vz = self.states[-1]
        if z < -1 or x < -5 or np.abs(y) > court_w/2:
            self.end = True
        # TODO: Implement check condition for scoring

    def visualize(self):
        x, y, z = self.states[:,0:3].T
        vx, vy, vz = self.states[:,3:6].T
        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z)
        ax.scatter(x[0], y[0], z[0], color="orange", s=100)

        backboard = Rectangle((-bb_l/2, bb_z_bot), bb_l, bb_h, fill=False, linewidth=1)
        ax.add_patch(backboard)
        art3d.pathpatch_2d_to_3d(backboard, z=bb_x, zdir="x")

        bb_box = Rectangle((-1, bb_z_bot+.5), 2, 1.5, fill=False, linewidth=1)
        ax.add_patch(bb_box)
        art3d.pathpatch_2d_to_3d(bb_box, z=bb_x, zdir="x")

        rim = Circle((0, 0), rim_r, fill=False, edgecolor="red", linewidth=1)
        ax.add_patch(rim)
        art3d.pathpatch_2d_to_3d(rim, z=rim_h)

        court = Rectangle((-4, -court_w/2), court_l/2, court_w, fill=False, linewidth=2)
        ax.add_patch(court)
        art3d.pathpatch_2d_to_3d(court, z=0)

        key = Rectangle((-4, -6), 19, 12, fill=False, linewidth=1)
        ax.add_patch(key)
        art3d.pathpatch_2d_to_3d(key, z=0)

        key_circle = Circle((15, 0), 6, fill=False, linewidth=1)
        ax.add_patch(key_circle)
        art3d.pathpatch_2d_to_3d(key_circle, z=0)

        ax.plot([-4, 10], [22, 22], [0, 0], linewidth=1, color="black")
        ax.plot([-4, 10], [-22, -22], [0, 0], linewidth=1, color="black")

        ys_3 = np.arange(0, 22, 0.05)
        xs_3 = np.sqrt(23.75**2 - ys_3**2)
        ys_3 = np.concatenate([-ys_3[::-1], ys_3])
        xs_3 = np.concatenate([xs_3[::-1], xs_3])
        zs_3 = np.zeros(len(ys_3))
        ax.plot(xs_3, ys_3, zs_3, linewidth=1, color="black")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Shot trajectory")
        ax.set_xlim(-5, 40)
        ax.set_ylim(-court_w/2-1, court_w/2+1)
        ax.set_zlim(0, 20)

        # ax.grid(None)
        plt.tight_layout()
        # plt.show()

        fig, (ax1, ax2) = plt.subplots(2, 1)
        t = [dt*i for i in range(len(x))]
        ax1.plot(t, x, label="x")
        ax1.plot(t, y, label="y")
        ax1.plot(t, z, label="z")
        ax2.plot(t, vx, label="vx")
        ax2.plot(t, vy, label="vy")
        ax2.plot(t, vz, label="vz")
        ax2.set_xlabel("Time")
        ax1.legend()
        ax2.legend()
        plt.show()


if __name__ == "__main__":
    ball = Ball(15, 0, 5.5, 26, 50, 0)
