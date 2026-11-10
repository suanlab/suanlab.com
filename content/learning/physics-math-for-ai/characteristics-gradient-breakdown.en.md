## Follow labels before reconstructing a field

For linear transport u_t+c u_x=0, a characteristic satisfies X'=c. Along it, d/dt u(X(t),t)=u_t+c u_x=0. A label xi initially at position xi therefore carries its original value u0(xi) to X=xi+ct. Parallel characteristic lines preserve the ordering of labels and the initial profile's shape.

Burgers' equation changes the speed to the carried value itself: u_t+u u_x=0. While a classical solution is smooth, each characteristic remains straight because its value stays constant, but neighboring lines can have different slopes. Faster material can catch slower material.

## An affine profile exposes the inverse map

The experiment uses u0(xi)=a-b xi. For Burgers,

$$
X(\xi,t)=\xi+t(a-b\xi)=(1-bt)\xi+at,
\qquad \partial_\xi X=1-bt.
$$

Before this Jacobian vanishes, solving for xi reconstructs the Eulerian field. Differentiating the carried value gives

$$
u_x=\frac{u_0'(\xi)}{\partial_\xi X}
=-\frac b{1-bt}.
$$

If b is positive, the first forward breakdown time is 1/b. For b=1 and t=1/2 the gradient is minus two. At t=1, every affine characteristic in this model meets at the same position a. Finite carried values coexist with an unbounded spatial gradient, so bounded amplitude does not guarantee continued classical smoothness.

## Do not draw crossed paths as a classical solution

The upper diagram shows raw straight characteristic lines over times zero to two and highlights the selected current time. Their intersections are a statement about the label construction, not a valid continuation of a single-valued smooth field. At or after breakdown, the lab leaves the classical Eulerian gradient absent.

The lower graph deliberately plots current position against initial label, with identity as a reference. It is not a plot of u against spatial x. When its slope becomes zero or negative, continuing to use a pre-breakdown inverse formula would hide the loss of the original classical solution. Entropy weak solutions require an additional selection principle, introduced next.

## Not every profile breaks forward in time

For b zero, all Burgers characteristics are parallel. For negative b, the Jacobian grows rather than shrinks over positive time, and this affine profile has no positive-time gradient breakdown. In constant-speed transport the Jacobian is always one regardless of b. Distinguishing equations is as important as inspecting a plotted intersection.

## Exercise: compute and interpret the crossing

Choose a=1/2, b=1, mode Burgers. At label zero and t=1/2, position is 1/4, carried value is 1/2 and gradient is -2. At t=1, the current-position map collapses to X=1/2 for every label. Explain why the carried values can no longer define one classical value at that position. Then switch to constant speed 1/2 and verify the same initial labels remain ordered for every displayed time.

Further reading: [Clawpack authors' Burgers chapter](https://www.clawpack.org/riemann_book/html/Burgers.html). The affine example and validity checks are derived independently.
