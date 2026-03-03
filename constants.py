import numpy as np

# Physical Conversions
ft2m = 0.3048 # [m/ft]
m2ft = 1 / ft2m # [ft/m]
kg2lb = 2.20462 # [lbm/kg]

# Environment Constants
g = 32.174 # [ft/s^2]
g_eff = g * .985 # Effective gravity after 1.5% buoyancy force
rho = 0.0764 # Air Density [lbm/ft^3]
mu = 3.737e-7 # Dynamic viscosity [lbm * s/ft^2]
timestep = 0.002 # Integrator time step [sec]
sim_duration = 5.0 # Maximum simulation time [sec]

# NBA Court Dimensions
court_l = 94.0
court_w = 50.0

# Rim & Backboard Geometry (Origin is center of the rim)
rim_h = 10.0 # Rim height 10ft
rim_r = 1.5/2 # Rim diameter 1.5ft
rim_bb_dist = 0.5 # Rim end to backboard distance 6 inches

bb_l = 6.0 # Backboard width 6ft
bb_h = 3.5 # Backboard height 3.5ft
bb_z_bot = 9.0 # Backboard bottom z coord
bb_z_top = bb_z_bot + bb_h # Backboard top z coord
bb_x = -rim_r - rim_bb_dist # Backboard x coordinate

# Ball Properties
ball_r = 29.5 / 12 / (2 * np.pi) # 29.5" circumference
ball_m = 0.620 * kg2lb # Weight (567-650g - Wikipedia) [grams]
ball_e1 = 0.84 # Coefficient of Restitution - Ground
ball_e2 = 0.65 # Coefficient of Restitution - Backboard/rim
mu_ground = 0.6 # Friction - Ground
mu_bb = 0.2     # Friction - Backboard
mu_rim = 0.3    # Friction - Rim
ball_cd1 = 0.5 # Coefficient of drag (low Re)
ball_cd2 = 0.2 # Coefficient of drag (Drag crisis)
alpha = 0.66
ball_I = alpha * ball_m * ball_r ** 2 # Moment of inertia
