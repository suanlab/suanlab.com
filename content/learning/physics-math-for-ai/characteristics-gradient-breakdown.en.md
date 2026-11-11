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

## Invertibility, slope blow-up, and inverse sensitivity

For C¹ bounded Burgers initial data with bounded derivative, X(ξ,t)=ξ+tu₀(ξ) has Jacobian 1+tu₀′(ξ). While this remains uniformly positive, the inverse map exists globally on the real line: bounded u₀ makes X proper and the positive derivative makes it strictly increasing. Along each characteristic,

$$
u_x(X(\xi,t),t)=\frac{u_0'(\xi)}{1+tu_0'(\xi)},\qquad
\frac d{dt}u_x=-u_x^2.
$$

If m=inf u₀′<0, the first loss of a uniform gradient bound is T*=−1/m. If the infimum is attained, a characteristic has infinite slope at that time. For u₀(ξ)=−tanh ξ, m=−1 at ξ=0, so T*=1 although the transported values remain between −1 and 1. With nonnegative initial slope there is no forward compression breakdown of this kind.

The inverse derivative ∂ξ/∂x=1/(1+tu₀′) also measures sensitivity: a position error δx near the compressive center creates about δx/(1−t) label error. Thus an apparently accurate characteristic position can reconstruct a poor Eulerian field near breakdown. The affine live lab has the same Jacobian exactly, but is not globally bounded initial data; its explicit formula supplies its own pre-breakdown inverse.

| t at the tanh center | Jacobian | Inverse sensitivity |
|---|---:|---:|
| 1/2 | 1/2 | 2 |
| 9/10 | 1/10 | 10 |
| 99/100 | 1/100 | 100 |

```python
from math import tanh, cosh
for t in (.5, .9, .99):
    slope0 = -1.
    J = 1+t*slope0
    slope = slope0/J
    slope_time_derivative = -slope0**2/J**2
    assert abs(slope_time_derivative+slope*slope) < 1e-9
    for xi in (-1., 0., 1.):
        u0 = -tanh(xi)
        derivative = -1/cosh(xi)**2
        assert 1+t*derivative > 0
        position = xi+t*u0
        assert abs(position-xi) <= t
    print(t, J, 1/J, slope)
```

The derivative and Riccati identities are analytic; the code verifies representative values before T*. After breakdown a crossed label plot is not a single-valued classical solution. Entropy selection, not numerical inversion of a folded map, supplies the physical continuation.

## Exercise: compute and interpret the crossing

Choose a=1/2, b=1, mode Burgers. At label zero and t=1/2, position is 1/4, carried value is 1/2 and gradient is -2. At t=1, the current-position map collapses to X=1/2 for every label. Explain why the carried values can no longer define one classical value at that position. Then switch to constant speed 1/2 and verify the same initial labels remain ordered for every displayed time.

Further reading: [Clawpack authors' Burgers chapter](https://www.clawpack.org/riemann_book/html/Burgers.html). The affine example and validity checks are derived independently.
