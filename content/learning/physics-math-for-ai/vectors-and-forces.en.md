## Same numbers, different directions

If an object is pushed with 3 N to the right and 4 N upward, is the total 7 N? Force has both magnitude and direction. **Add components in a common coordinate system before computing magnitude.**

The goal is to distinguish components, direction, and magnitude, then relate net force to acceleration. Start with coordinates and basic arithmetic.

## How to read this unit

Start by connecting a list of numbers to an arrow, then connect force to motion. Previous courses in calculus or linear algebra are not assumed. Finally, reuse the arithmetic for feature vectors and weighted combinations in AI.

- **Explain:** Distinguish components, magnitude and direction.
- **Calculate:** Add vectors, find their length, and calculate acceleration, velocity and position from force.
- **Interpret:** Give a counterexample showing why zero force and zero velocity are different statements.

## Scalars and vectors

A scalar answers “how much?” with one number and a unit: mass 2 kg or time 3 s. A vector also specifies direction. Speed 5 m/s is a scalar; a velocity of 5 m/s to the right includes a direction.

| Term | Meaning | Example in this unit |
| --- | --- | --- |
| Scalar | A number specifying an amount | Mass $m=2$ kg |
| Vector | Components relative to chosen axes | Force $\mathbf F=(3,4,0)$ N |
| Component | A signed amount along an axis | $F_x=3$, $F_y=4$, $F_z=0$ |
| Magnitude or norm | Length of the represented arrow | $\|\mathbf F\|=5$ N |
| Zero vector | Every component is zero | $(0,0,0)$ |

N denotes the newton, a unit of force. A net force of 1 N gives a mass of 1 kg an acceleration of 1 m/s². Equal numerical values in N, m/s and m still describe different physical quantities.

A negative component means the negative direction along that axis. The magnitude of `(-3,0,0)` N is 3 N, not −3 N. **Choose the axes and units before interpreting components.**

## Coordinates are a record separated by direction

With x pointing right and y pointing up, `(3,4,0)` records 3 along x, 4 along y and zero along z. Initially, study the plane z=0. Changing z extends exactly the same rule into space.

The unit vectors describing the reference directions are

$$
\mathbf e_x=(1,0,0),\quad \mathbf e_y=(0,1,0),\quad \mathbf e_z=(0,0,1).
$$

Thus $\mathbf F=(3\mathbf e_x+4\mathbf e_y+0\mathbf e_z)\,\mathrm N$. The coefficients 3 and 4 specify how much of each reference direction to use. This prepares the idea of a basis in the next unit.

Walking 3 m east and then 4 m north gives a path length of 7 m, but the straight-line displacement has magnitude 5 m. This analogy separates components from length. Adding forces does not itself mean that an object traveled along those two segments.

## Add the components

Write the forces as

$$
\mathbf F_1=(3,0,0)\;\mathrm N,\qquad \mathbf F_2=(0,4,0)\;\mathrm N.
$$

Adding matching components gives

$$
\mathbf F_{\rm net}=\mathbf F_1+\mathbf F_2=(3,4,0)\;\mathrm N.
$$

Use the Pythagorean theorem for its magnitude:

$$
\|\mathbf F_{\rm net}\|=\sqrt{3^2+4^2+0^2}=5\;\mathrm N.
$$

![Perpendicular forces of 3 N along x and 4 N along y add to a diagonal of magnitude 5 N.](/assets/learning/vectors-and-forces.svg)

Translate the second arrow to the tip of the first. The arrow from the initial tail to the final tip is the sum. Translating an arrow to illustrate addition is different from changing the physical point of application. Here, both forces act on a particle.

## Split the hand calculation into three steps

1. Add matching components: `(3+0, 0+4, 0+0)=(3,4,0)`.
2. Only when finding magnitude, square and sum: `3²+4²+0²=25`.
3. Take the square root to get 5 N. Do not begin by adding the magnitudes 3 N and 4 N.

Why can an arrow move in this construction? Vector addition uses each arrow's **direction and length**. A parallel translation preserves its components. Following the translated second arrow from the first endpoint arrives at the coordinatewise sum.

Adding magnitudes directly works for vectors pointing in the same direction. Opposite vectors can cancel. In general,

$$
\|\mathbf u+\mathbf v\|\leq\|\mathbf u\|+\|\mathbf v\|.
$$

This triangle inequality connects to the geometric fact that a straight shortcut is no longer than a route through an intermediate point.

## Try three cases

In **Force addition · N** view, switch presets and compare your prediction with the result.

1. Perpendicular: `(3,0,0)` and `(0,4,0)` → magnitude 5 N.
2. Same direction: `(3,0,0)` and `(3,0,0)` → magnitude 6 N.
3. Cancellation: `(3,0,0)` and `(-3,0,0)` → zero net force.

Rotating the camera does not change the components in the table. Changing the viewpoint does not change the coordinate system or the forces. Use the labels `F₁`, `F₂`, and `Fnet` as well as the colors.

## From force to acceleration

For a particle of constant mass, Newton's second law gives

$$
\mathbf a=\frac{\mathbf F_{\rm net}}{m}.
$$

With mass 2 kg, acceleration is `(1.5,2,0)` m/s² with magnitude 2.5 m/s². Doubling only the mass halves acceleration under the same force. Distinguish N from m/s².

**Model assumptions:** A particle in an inertial coordinate system has constant mass and two constant external forces. Gravity, friction, collisions, and rotation are excluded. Editing force or mass restarts from the selected initial state at time zero.

## Motion and time

**Particle motion · m** view shows position and trajectory in metres instead of force arrows. Starting with zero position and velocity, constant acceleration gives

$$
\mathbf v(t)=\mathbf a t,\qquad \mathbf r(t)=\tfrac12\mathbf a t^2.
$$

At two seconds in the reference example, velocity is `(3,4,0)` m/s and position is `(3,4,0)` m. Matching numbers represent different quantities. Check the grid units and display scale in each view. Fitting the scene on screen does not change the calculation.

**Counterexample:** Zero net force need not mean rest. The **Opposing · moving** preset starts with velocity `(1,0,0)` m/s. Acceleration is zero, yet position becomes `(2,0,0)` m after two seconds.

## Check against a time table

Acceleration `(1.5,2,0)` m/s² means that **velocity changes by that vector each second**. Acceleration is not position or velocity. With both initial position and initial velocity zero, the reference example gives:

| Time t (s) | Velocity v (m/s) | Position r (m) |
| --- | --- | --- |
| 0 | `(0,0,0)` | `(0,0,0)` |
| 1 | `(1.5,2,0)` | `(0.75,1,0)` |
| 2 | `(3,4,0)` | `(3,4,0)` |

During the first second, velocity increases uniformly from zero to `(1.5,2,0)`. Its average is half the final velocity, giving displacement `(0.75,1,0)` m. This explains the factor 1/2 in the position equation. From rest under constant acceleration, doubling time quadruples displacement.

If initial velocity is present, include its contribution:

$$
\mathbf v(t)=\mathbf v_0+\mathbf a t,\qquad
\mathbf r(t)=\mathbf r_0+\mathbf v_0t+\tfrac12\mathbf a t^2.
$$

Select **Opposing · moving** to see that $\mathbf v_0t$ remains even when $\mathbf a=0$. Rotating the camera changes none of these terms.

### A second check: energy

After two seconds, speed is 5 m/s, so kinetic energy is $\tfrac12\times2\times5^2=25$ J. Work by the constant net force is $\mathbf F\cdot\Delta\mathbf r=3\times3+4\times4=25$ J. The dot product multiplies matching components and adds them. Under these assumptions, the work equals the increase in kinetic energy.

## Connecting to vectors in AI

Feature-vector addition and weighted sums use the same component-wise operations. A feature vector need not represent a physical force: its meaning and units depend on the problem.

Dividing `(3,4,0)` N by its magnitude, 5 N, gives the dimensionless direction `(0.6,0.8,0)`. Normalizing direction means that the original magnitude must be kept separately if it is needed.

### A small AI example: mixing two representations

Suppose two models express features in the same two-dimensional coordinate system as $\mathbf u=(2,0)$ and $\mathbf v=(0,4)$. Give the first 25% weight and the second 75%:

$$
0.25\mathbf u+0.75\mathbf v=(0.5,3).
$$

The coefficients scale the arrows; addition combines the representations. This invented example illustrates weighted-sum arithmetic, not an experimental claim that representation quality improves. If the models' coordinates mean different things, first reconsider whether combining matching components is meaningful.

Normalization by length can help compare directions. The zero vector has length zero, so this operation cannot define its direction. Features with different units also require attention to preprocessing and scale. A numerically large contribution to length does not automatically mean greater importance for the research question.

## Common misconceptions

| Thought | Counterexample or check |
| --- | --- |
| Force magnitudes can always be added | Perpendicular forces of 3 N and 4 N have resultant magnitude 5 N |
| Zero net force means rest | An initial velocity persists as uniform motion |
| Rotating the camera changes force components | This camera changes the viewing direction only |
| Every vector is a direction in physical space | Feature coordinates are defined for the task |

When comparing research representations, record the coordinate system, norm and normalization. Two quantities called “distance” need not support the same interpretation when feature scales and definitions differ.

## Challenge and check

Keep `F₁=(3,0,0)` N and change `F₂` so that the net force is `(0,4,0)` N. Verify the components, then try the questions.

**Explanation:** Choose `F₂=(-3,4,0)` N. Matching magnitude alone does not determine the target direction.

Physical reference: [OpenStax — Newton's Second Law](https://openstax.org/books/university-physics-volume-1/pages/5-3-newtons-second-law).
