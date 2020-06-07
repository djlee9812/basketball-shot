# Basketball Shot Simulation

Dynamical simulation and visualization of a basketball shot given starting position, throw speed, launch angle, and deviation angle.

![Image of 3 Point Shot](https://raw.githubusercontent.com/djlee9812/basketball-shot/master/plots/demo.png)

## How to Use
### `simulate.py`
#### Initialize
The Ball class contains all necessary methods for the dynamical simulation of the basketball shot as well as visualization of the shot trajectory and the basketball court. To use, simply initialize the object as follows and the code will run the dynamical simulation:

```
ball = Ball(x, y, z, speed, launch_angle, side_angle, backspin)
```
#### Position
All units are imperial. The initial position of the shot is set with variables `x, y, z`, where `x` is the distance in front of the rim, `y` is the distance to the right of the rim and `z` is height from ground.

For example, a free throw shot for a person shooting at 6ft would have coordinates `(x,y,z) = (15, 0, 6)` and a right corner 3 point shot would be `(x,y,z) = (0, 23.75, 6)`. The shot in the example image above is at `(x,y,z) = (24, 10, 5.5)`, where the orange sphere shows this coordinate in 3D and the gray circle is the x, y position at z=0.
#### Shot Parameters
The speed is the speed of the thrown ball in ft/s.  `launch_angle` is the angle from the ground plane. The angle to the rim is automatically calculated, but can be offset by setting `side_angle`. A shot straight to the hoop should have a `side_angle` of 0. All angles are given in degrees. The `backspin` parameter specifies the backspin speed in revolutions per second. The spin vector is calculated within the `Ball()` class.

#### Score
The class contains a variable `Ball.score` which is a Boolean that determines whether the given shot made or not.

### `analysis.py`

This file contains functions to iterate over a range of shot speeds and angles to see which combinations of shot parameters score baskets. We can see optimal shot angles at various positions and which shots are most resistance to noise or human errors. This file uses multiprocessing to speed up computation since each shot instance doesn't rely on the results of other shots.

## Model
The current version calculates gravity, air resistance (drag), Magnus force, and momentum impulse from collision with the ground, backboard, and rim. Future updates will include friction during collisions and change in spin over time.

Drag coefficient decreases from 0.5 to 0.2 at Re = 1e5 to Re = 2e5. The drag crisis drop are estimated numbers from "On the Size of Sport Fields (Texier et al)" which use numbers from soccer balls. "Identification of basketball parameters for a simulation model (Okubo, Hubbard)" puts a basketball CD at free fall (low Re) at 0.54.

The Magnus force data relies on lift coefficient ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Csmall%20%28C_L%3DaSp&plus;b%29), which is an affine function with the spin factor ![equation](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Csmall%20Sp%3D%20%5Cfrac%7B%5Comega%20r%20%7D%7Bv%7D). The spin factor Sp defines the ratio between spin tangential velocity and translational velocity. Because the spin factor is divided by the velocity (which can be quite low at certain instances), the Sp term is clipped to a maximum of 3.
