# Basketball Shot Simulation

Dynamical simulation and visualization of a basketball shot given starting position, throw speed, launch angle, and deviation angle.

![Image of 3 Point Shot](https://raw.githubusercontent.com/djlee9812/basketball-shot/master/demo.png)

## How to Use
#### Initialize
The Ball class contains all necessary methods for the dynamical simulation of the basketball shot as well as visualization of the shot trajectory and the basketball court. To use, simply initialize the object as follows and the code will run the dynamical simulation:

```
ball = Ball(x, y, z, speed, launch_angle, side_angle)
```
#### Position
All units are imperial. The initial position of the shot is set with variables `x, y, z`, where `x` is the distance in front of the rim, `y` is the distance to the right of the rim and `z` is height from ground.

For example, a free throw shot for a person shooting at 6ft would have coordinates `(x,y,z) = (15, 0, 6)` and a right corner 3 point shot would be `(x,y,z) = (0, 23.75, 6)`.
#### Shot Parameters
The speed is the speed of the thrown ball in ft/s.  `launch_angle` is the angle from the ground plane. The angle to the rim is automatically calculated, but can be offset by setting `side_angle`. A shot straight to the hoop should have a `side_angle` of 0. All angles are given in degrees.

#### Score
The class contains a variable `Ball.score` which is a Boolean that determines whether the given shot made or not.

## Effects Included
The current version calculates gravity, air resistance (drag), and momentum impulse from collision with the ground, backboard, and rim. Future updates will include the Magnus effect with a set `spin` parameter and friction during collisions.

Drag coefficient decreases from 0.5 to 0.2 at Re = 1e5 to Re = 2e5. The drag crisis drop are estimated numbers from "On the Size of Sport Fields (Texier et al)" which use numbers from soccer balls. "Identification of basketball parameters for a simulation model (Okubo, Hubbard)" puts a basketball CD at free fall (low Re) at 0.54.
