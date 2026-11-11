## 1. Differentiate a vector whose basis moves

On a curved surface the tangent plane depends on position. Even in a flat plane a polar basis turns as the coordinate angle changes. The covariant derivative compares vectors using a connection. For a vector $V$, its components are
$$
(\nabla_i V)^k=\partial_i V^k+\Gamma^k_{ij}V^j.
$$
The connection term corrects for the variation of the basis. The Levi-Civita connection is determined by the metric together with metric compatibility and zero torsion. [Tong, Introducing Riemannian Geometry §3.2](https://davidtong.org/teaching/general-relativity/grhtml/S3)

## 2. Compare three rotationally symmetric surfaces

Use the metric $ds^2=dr^2+f(r)^2d\theta^2$ where $f(r)>0$ away from the center. The functions for the plane, sphere of radius $R$, and hyperbolic plane of curvature radius $R$ are $r$, $R\sin(r/R)$, and $R\sinh(r/R)$. Direct substitution into the Levi-Civita formula gives
$$
\Gamma^r_{\theta\theta}=-ff',\qquad
\Gamma^\theta_{r\theta}=\Gamma^\theta_{\theta r}=f'/f.
$$
The plane has nonzero connection coefficients in these coordinates.

## 3. Compute curvature from the connection

Use the convention in which the round sphere has positive Gaussian curvature. In this metric the curvature is
$$
K=-f''/f.
$$
For example, the radial-angle component $R^r{}_{\theta r\theta}$ equals $-ff''$. Terms proportional to $(f')^2$ cancel. The plane has $K=0$ even though its connection coefficients are nonzero. The sphere and hyperbolic plane give $K=1/R^2$ and $K=-1/R^2$. The scalar curvature in two dimensions is $2K$.

## 4. Carry a vector around a circle

Use the orthonormal frame $e_r=\partial_r$, $e_\theta=f^{-1}\partial_\theta$. At fixed $r$ it obeys $\nabla_\theta e_r=f'e_\theta$ and $\nabla_\theta e_\theta=-f'e_r$. For a parallel vector whose angle in this frame is $\alpha$, the equation is $d\alpha/d\theta=-f'$. After one positively oriented loop, its angle changes by $-2\pi f'$, equivalent modulo $2\pi$ to $2\pi(1-f')$.

For the plane the basis turns once, and the carried vector returns unchanged. On the sphere the residual rotation is $2\pi(1-\cos(r/R))$, equal to the curvature times the enclosed cap area. The analogous hyperbolic value is negative. The signed expression is a continuous representative starting at zero for small loops. A single measured final angle is defined modulo $2\pi$. [Tong, Introducing Riemannian Geometry §3.3](https://davidtong.org/teaching/general-relativity/grhtml/S3)

## 5. Check the small-loop limit

```python
from math import cos, cosh, isclose, pi

R = 2
r = 0.7
sphere_area = 2*pi*R**2*(1-cos(r/R))
hyperbolic_area = 2*pi*R**2*(cosh(r/R)-1)
sphere_turn = 2*pi*(1-cos(r/R))
hyperbolic_turn = 2*pi*(1-cosh(r/R))
assert isclose(sphere_turn, sphere_area/R**2)
assert isclose(hyperbolic_turn, -hyperbolic_area/R**2)
assert sphere_turn > 0 > hyperbolic_turn
small = 0.001
assert isclose(2*(1-cos(small/R))/small**2, 1/R**2, rel_tol=1e-6)
print(sphere_turn, hyperbolic_turn)
```

## 6. Inspect geometry without relying on a coordinate symbol

- Compare the connection coefficients and Gaussian curvature for all three surfaces.
- On the plane follow a vector in the turning polar basis and compare its final direction to its starting direction.
- Increase the loop radius on the sphere and inspect both the rotation and the area.
- In general relativity the same connection and curvature ideas apply to a Lorentzian metric. This experiment uses spatial surfaces so that the path and rotation are easy to display.
