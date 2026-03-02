# Basketball Shot Simulation

Dynamical simulation and visualization of a basketball shot using a 4th-order Runge-Kutta (RK4) integrator. This codebase models the complex physics of a basketball flight, including gravity, drag crisis, Magnus effect, and discrete collisions.

![Image of 3 Point Shot](https://raw.githubusercontent.com/djlee9812/basketball-shot/master/plots/demo.png)

## Features
- **High-Accuracy Physics**: Uses an **RK4 Integrator** and discrete impulse handlers for realistic trajectory and bounces.
- **Vectorized Analysis**: Run parameter sweeps at **1,000+ shots per second** using NumPy.
- **Interactive Visualization**: 3D court rendering with optional animation (`--animate`) and physics debugging (`--debug`).
- **Command-Line Interface**: Easily test different shot parameters from the terminal.
- **Unit Tests**: Built-in physics validation for energy conservation and scoring logic.

---

## How to Use

### 1. Single Shot Simulation (`simulate.py`)
Run a single shot with custom parameters directly from the CLI:

```bash
# Run a standard free throw
python3 simulate.py -v 26 -a 56

# Animate a high-arc 3-pointer
python3 simulate.py -x 23.75 -v 30 -a 60 --animate

# Debug physics (drag, lift, and spin coefficients)
python3 simulate.py --debug
```

**Available Flags:**
- `-x`, `-y`, `-z`: Initial position [ft] (Origin is the center of the rim).
- `-v`, `--speed`: Launch speed [ft/s].
- `-a`, `--angle`: Vertical launch angle [deg].
- `-s`, `--side`: Side angle deviation [deg].
- `-w`, `--spin`: Backspin [rev/s].
- `--animate`: Toggles 3D animation.
- `--debug`: Shows time-series plots of position, velocity, and aerodynamic forces.

### 2. High-Speed Analysis (`vectorized_analysis.py`)
To map out the "success space" (which combinations of speed and angle result in a score), use the vectorized simulator:

```bash
python3 vectorized_analysis.py
```
This script bypasses slow Python loops and uses NumPy matrix operations to simulate thousands of shots simultaneously, producing a success heat map in seconds.

### 3. Running Tests
Validate the physics engine using the automated test suite:
```bash
python3 -m unittest discover tests
```

---

## Model
The current version calculates gravity, air resistance (drag), Magnus force, and momentum impulse from collision with the ground, backboard, and rim. Future updates will include friction during collisions and change in spin over time.

Drag coefficient decreases from 0.5 to 0.2 at Re = 1e5 to Re = 2e5. The drag crisis drop are estimated numbers from **"On the Size of Sport Fields (Texier et al)"** which use numbers from soccer balls. **"Identification of basketball parameters for a simulation model (Okubo, Hubbard)"** puts a basketball CD at free fall (low Re) at 0.54.

The Magnus force data relies on lift coefficient $C_L = aSp + b$, which is an affine function with the spin factor $Sp = \frac{\omega r}{v}$. The spin factor Sp defines the ratio between spin tangential velocity and translational velocity. Because the spin factor is divided by the velocity (which can be quite low at certain instances), the Sp term is clipped to a maximum of 3.

All physical constants (ball weight, rim radius, air density, etc.) are centralized in `constants.py`.
