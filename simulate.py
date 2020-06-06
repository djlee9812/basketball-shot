import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
plt.style.use('seaborn')

# Constants (Imperial Units)
ft2m = 0.3048 # [m/ft]
m2ft = 1 / ft2m # [ft/m]
kg2lb = 2.20462 # [lbm/kg]
g = 32.174 # 9.81 * m2ft # [ft/s^2]
g_eff = g * .985 # Effective gravity after 1.5% buoyancy force
rho = 0.0764 # 1.225 * kg2lb / m2ft**3 # Air Density [lbm/ft^3]
mu = 3.737e-7 # Dynamic viscosity [lbm * s/ft^2]
timestep = 0.002 # Integrator time step [sec]
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
ball_cd1 = 0.5 # Coefficient of drag (low Re)
ball_cd2 = 0.2 # Coefficient of drag (Drag crisis)
ball_contact_t = 0.01 # Impact time on contact [seconds]
alpha = 0.66
ball_I = alpha * ball_m * ball_r ** 2 # Moment of inertia of basketball [ft^3]
mu_bb = 0.6
mu_rim = 0.5

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
        omega_dir /= np.linalg.norm(omega_dir)
        wx, wy, wz = omega * omega_dir
        ics = [x, y, z, vx, vy, vz, wx, wy, wz]
        # Save ICs in state vector
        self.states.append(ics)
        nit = 0
        # Terminate either when out of bounds or 10 seconds passed
        while not self.end and nit < 3/timestep:
            derivs = self.dynamics()
            self.integrate(derivs)
            self.check_end()
            nit += 1
        self.states = np.array(self.states)
        self.data = np.array(self.data)

    def dynamics(self):
        """ Return time derivative of state variables
        :params x: State variable vector
            position, velocity
        """
        x, y, z, vx, vy, vz, wx, wy, wz = self.states[-1]
        # dx/dt = v
        dx, dy, dz = (vx, vy, vz)
        speed = np.linalg.norm([vx, vy, vz])
        # Gravity + buoyancy (z)
        Fg = ball_m * g_eff
        # Drag - Dx = D * vx/speed
        CD, Re = self.get_cd(speed)
        Fd = 0.5 * CD * rho * speed**2 * (np.pi*ball_r**2)

        # Magnus (omega x v)
        omega = np.linalg.norm([wx, wy, wz])
        # Clip Sp to range [0,3]
        Sp = max(0, min(omega * ball_r / speed, 3))
        CL = Sp * 0.77 + 0.12
        Fmx, Fmy, Fmz = 0.5 * CL * np.pi * ball_r**3 * rho \
                        * np.cross([wx, wy, wz], [vx, vy, vz])

        Fx = -Fd * vx / speed + Fmx
        Fy = -Fd * vy / speed + Fmy
        Fz = -Fd * vz / speed - Fg + Fmz

        dwx, dwy, dwz = np.zeros(3)

        Fcoll = self.collision()
        # If collision, ignore other forces for this time step
        if np.count_nonzero(Fcoll) > 0:
            Fx, Fy, Fz = Fcoll
        # Acceleration = force / mass
        dvx, dvy, dvz = np.array([Fx, Fy, Fz]) / ball_m
        data = [Fg, CD, Fd, Re, Sp, CL, Fmx, Fmy, Fmz]
        data.extend(Fcoll)
        self.data.append(data)

        return [dx, dy, dz, dvx, dvy, dvz, dwx, dwy, dwz]

    def get_cd(self, speed):
        """ Calculate drag coefficient of basketball including drag crisis effects
        between Re 1e5 and 2e5 where CD decreases linearly from ball_cd1 to ball_cd2
        Re = 1e5 corresponds to approximately 20ft/s
        """
        Re = rho / g * speed * 2 * ball_r / mu
        CD = ball_cd1
        if 1e5 < Re < 2e5:
            # CD drop over transition to drag crisis
            slope = (ball_cd2 - ball_cd1) / 1e5
            CD += (Re - 1e5) * slope
        if Re >= 2e5:
            CD = ball_cd2
        return CD, Re

    def collision(self):
        """ Calculate impulse force from collision.
        Handles backboard, rim, and ground collisions separately and superposes
        if multiple collisions at a time step. Returns impulse vector dp/dt
        """
        x, y, z, vx, vy, vz, wx, wy, wz = self.states[-1]
        speed = np.linalg.norm([vx, vy, vz])
        delta_p = np.zeros(3)

        # Shift collision backboard, ground queues
        self.bb = np.concatenate([self.bb[1:], [0]])
        self.rimsq = np.concatenate([self.rimsq[1:], [0]])
        self.ground = np.concatenate([self.ground[1:], [0]])
        # If collision with x in last 5 time steps: skip x
        # Backboard collision
        if self.dist_to_bb() < ball_r and np.count_nonzero(self.bb) == 0:
            # print("Backboard: t = {:.3f}; pos = ({:.2f}, {:.2f}, {:.2f})".format(\
                # timestep*len(self.states), x, y, z))
            # If collision straight on backboard
            if bb_z_bot < z < bb_z_top and -bb_l/2 < y < bb_l/2:
                delta_p[0] += (1 + ball_e2) * (-vx * ball_m)
            # If collision makes it bounce away
            else:
                ypt = max(-bb_l/2, min(bb_l/2, y))
                zpt = max(bb_z_bot, min(bb_z_top, z))
                normvec = np.array([x,y,z]) - np.array([bb_x, ypt, zpt])
                normvec /= np.linalg.norm(normvec)
                delta_p -= (1 + ball_e2) * np.dot([vx, vy, vz],normvec) * normvec * ball_m

            # Set backboard queue
            self.bb[-1] = 1
        # Rim collision
        rim_dist, rim_pt = self.dist_to_rim()
        # Requires new rim point and not recent collision with connector
        if rim_dist < ball_r and (x > -rim_r or np.abs(y) > .25) \
           and not np.allclose(rim_pt, self.rimpt, atol=0.5) and np.count_nonzero(self.rimsq) == 0:
            # print("Rim: t = {:.3f}; pos = ({:.2f}, {:.2f}, {:.2f})".format(\
            #     timestep*len(self.states), x, y, z))
            normvec = np.array([x,y,z]) - np.array(rim_pt)
            normvec /= np.linalg.norm(normvec)
            delta_p -= (1 + ball_e2) * np.dot([vx, vy, vz],normvec) * normvec * ball_m
            # Set last rim collision point
            self.rimpt = rim_pt
        # Rim connector collision
        if bb_x < x < -rim_r and np.abs(y) < .25 and 10 < z < 10 + ball_r \
           and np.count_nonzero(self.rimsq) == 0:
            # print("Connector: t = {:.3f}; pos = ({:.2f}, {:.2f}, {:.2f})".format(\
            #     timestep*len(self.states), x, y, z))
            delta_p[2] += (1 + ball_e2) * (-vz * ball_m)
            self.rimsq[-1] = 1
        # Ground collision
        if z <= ball_r and np.count_nonzero(self.ground) == 0:
            # print("Ground: t = {:.3f}; pos = ({:.2f}, {:.2f}, {:.2f})".format(\
            #     timestep*len(self.states), x, y, z))
            delta_p[2] += (1 + ball_e1) * (-vz * ball_m)
            # vx1 = np.linalg.norm([vx, vy])
            # vy1 = vz
            # omega1 =
            # vx2, omega2 = self.calc_friction(vx1, vy1, omega1, ball_e1)
            # Set ground queue
            self.ground[-1] = 1

        return delta_p / timestep

    def dist_to_bb(self):
        """ Calculate projection distance from ball to backboard
        """
        x, y, z, vx, vy, vz, wx, wy, wz = self.states[-1]
        dx = x - bb_x
        # 0 if within (a, b) range, positive otherwise
        dy = np.max([-bb_l/2 - y, 0, y - bb_l/2])
        dz = np.max([bb_z_bot - z, 0, z - bb_z_top])
        return np.linalg.norm([dx, dy, dz])

    def dist_to_rim(self):
        """ Calculate projection distance from ball to rim and collision point
        on the rim
        """
        x, y, z, vx, vy, vz, wx, wy, wz = self.states[-1]
        planar_d = np.linalg.norm([x,y]) - rim_r
        z_dist = z - 10
        dist = np.linalg.norm([planar_d, z_dist])
        scale = rim_r / np.linalg.norm([x,y])
        pt = [x * scale, y *scale, 10]
        return dist, pt

    def calc_friction(self, vx1, vy1, omega1, ey):
        A = 1 + 1/alpha
        ratio = np.inf if vx1 == 0 else vy1/vx1
        theta1 = np.arctan(np.abs(ratio))
        ex = 0.1 if theta1 > np.radians(35) else -0.2
        D = 1.1e-3 * m2ft if theta > np.radians(45) else 3e-3 * m2ft
        vx2 = vx1 - (1+ex)*(vx1-ball_r*omega1)/A - D*(1+ey)*vy1/(alpha*ball_r*A)
        omega2 = omega1 + (vx1-vx2)/(alpha*ball_r) - D*(1+ey)*vy1/(alpha*ball_r**2)
        return vx2, omega2

    def integrate(self, derivs):
        """ Euler method to integrate derivatives with given timestep dt
        """
        state = self.states[-1]
        new_state = [state[i] + derivs[i] * timestep for i in range(len(state))]
        self.states.append(new_state)

    def check_end(self):
        """ Termination conditions if out of bounds and check if scored
        """
        x, y, z, vx, vy, vz, wx, wy, wz = self.states[-1]
        # Check out of bound condition
        if z < -1 or (not -5 < x < 45) or np.abs(y) > court_w/2:# or (self.score and z < 1):
            self.end = True
        # Scoring condition: slightly below rim and inside rim circle
        if np.linalg.norm([x,y]) < rim_r and 9.5 < z < 9.95:
            self.score = True

    def visualize(self, debug=False):
        """ Visualize court and ball trajectory in pyplot. If debug is True,
        shows time graphs of position and velocity
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

        # Plot backboard rectangle
        backboard = Rectangle((-bb_l/2, bb_z_bot), bb_l, bb_h, fill=False, linewidth=1)
        ax.add_patch(backboard)
        art3d.pathpatch_2d_to_3d(backboard, z=bb_x, zdir="x")
        # Plot backboard inside box
        bb_box = Rectangle((-1, bb_z_bot+.5), 2, 1.5, fill=False, linewidth=1)
        ax.add_patch(bb_box)
        art3d.pathpatch_2d_to_3d(bb_box, z=bb_x, zdir="x")
        # Plot rim circle
        rim = Circle((0, 0), rim_r, fill=False, edgecolor="red", linewidth=1)
        ax.add_patch(rim)
        art3d.pathpatch_2d_to_3d(rim, z=rim_h)
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
        ax.set_xlim(-5, 30)
        ax.set_ylim(-court_w/2-1, court_w/2+1)
        ax.set_zlim(0, 20)

        ax.grid(None)
        plt.tight_layout()
        # If debug True, plot time graphs of position and velocity
        if debug:
            Fg, CD, Fd, Re, Sp, CL, Fmx, Fmy, Fmz, Fcx, Fcy, Fcz = self.data.T
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8,7))
            fig.suptitle("Ball time variables")
            t = [timestep*i for i in range(len(x))]
            ax1.plot(t, x, label="x")
            ax1.plot(t, y, label="y")
            ax1.plot(t, z, label="z")
            ax1.set_ylabel("Position [ft]")
            ax2.plot(t, vx, label=r"$v_x$")
            ax2.plot(t, vy, label=r"$v_y$")
            ax2.plot(t, vz, label=r"$v_z$")
            ax2.set_ylabel("Velocity [ft/s]")
            # ax3.plot(Re, CD, label="CD")
            # ax3.plot(Re, CL, label="CL")
            ax3.plot(t[1:], CD, label=r"$C_D$")
            ax3.plot(t[1:], Sp, label=r"$Sp$")
            ax3.plot(t[1:], CL, label=r"$C_L$")
            # ax3.plot(t[1:], Fmx)
            # ax3.plot(t[1:], Fmy)
            # ax3.plot(t[1:], Fmz)
            ax3.set_ylabel("Nondim forces")
            ax3.set_xlabel("Time [sec]")
            ax1.legend()
            ax2.legend()
            ax3.legend()

        plt.show()


if __name__ == "__main__":
    # Initialize ball object with
    # (x, y, z, speed, launch_angle, side_angle, backspin)
    ball = Ball(15, -15, 6, 31, 60, -8.5, 5)
    msg = "Score!" if ball.score else "Missed!"
    print(msg)
    ball.visualize(True)
