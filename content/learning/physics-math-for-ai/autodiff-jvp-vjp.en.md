## Differentiate an executed graph

Consider F(x,y)=sin(xy)+x². Its graph contains two inputs, a product a=xy, b=sin(a), another product c=x*x and the sum F=b+c. The same input x participates through more than one route. A correct derivative must include every route, including both arguments of the square multiplication.

The lab creates a primitive tape with values, parent indices and local partial derivatives. The graph labels switch among primal values, forward tangents and reverse adjoints. These labels come from actual propagation rather than substituting a closed-form gradient into each node. The analytic gradient is retained separately as a verification reference.

## Forward mode pushes a direction

Choose an input direction v=(vx,vy). Propagate its tangent through the chain rule:

$$
\dot a=yv_x+xv_y,\quad
\dot b=\cos(a)\dot a,\quad
\dot c=2xv_x,\quad
\dot F=\dot b+\dot c=Jv.
$$

One forward sweep computes the Jacobian action for this direction. For a general function with many outputs, it produces their directional changes together. Constructing a full Jacobian would require multiple independent directions, so the number of inputs and desired derivative products matters when selecting a mode.

## Reverse mode pulls a cotangent

Seed the scalar output adjoint with w. Traversing the tape backward adds local contributions to each parent. In particular, x receives the route through a and both parent entries of c=x*x:

$$
\bar x=w\left(y\cos(xy)+2x\right),\qquad
\bar y=w x\cos(xy).
$$

Overwriting a shared parent's adjoint would lose a path. The drawing collapses the two identical x-to-square edges for readability, while the table lists both local parent contributions and the reverse implementation accumulates both.

For a scalar output, one reverse sweep obtains all input partials multiplied by w. Retaining or recomputing forward states creates a memory tradeoff, developed in the checkpointing unit. AD applies derivative rules to the executed program; it is neither symbolic simplification of every expression nor subtraction of nearby function values.

## Pairing gives an independent consistency check

The scalar identity w(Jv)=(wJ)v compares forward and reverse products. At (x,y)=(1,0), the gradient is (2,1). With v=(1,1) and w=1 both sides equal three. Changing v to (0,1) changes JVP to one while leaving primal and VJP unchanged. Setting w=0 zeros every reverse adjoint while preserving the primal and forward tangent.

The lower plot compares F along the selected input direction with its first-order tangent. A tangent agrees locally, not necessarily over a finite step. Curvature explains their separation away from zero.

## Differentiate the reverse calculation to obtain a Hessian action

At a smooth point, a direction v can be propagated through the reverse-gradient program itself. This forward-over-reverse operation computes Hv without storing the whole Hessian. For this function the independent matrix is

$$
H=\begin{pmatrix}
2-y^2\sin(xy)&\cos(xy)-xy\sin(xy)\\
\cos(xy)-xy\sin(xy)&-x^2\sin(xy)
\end{pmatrix}.
$$

In the program below each tape value and each reverse adjoint is a dual number with a value and a tangent. Reverse accumulation therefore differentiates both the local partials and the child adjoints. Omitting either term would miss a Hessian contribution. Shared parents, including both square inputs, still receive separate additions.

At x=y=0 the gradient is zero, but the Hessian is not:

| v | Hv |
|---|---|
| (1,0) | (2,1) |
| (0,1) | (1,0) |
| (1,−1) | (1,1) |

```python
from math import sin, cos
class D:
    def __init__(self, value, tangent=0.): self.v, self.t = value, tangent
    def __add__(self, other): return D(self.v+other.v, self.t+other.t)
    def __mul__(self, other): return D(self.v*other.v, self.t*other.v+self.v*other.t)
def sine(x): return D(sin(x.v), cos(x.v)*x.t)
def cosine(x): return D(cos(x.v), -sin(x.v)*x.t)
def hvp(x, y, vx, vy):
    values, parents = [D(x, vx), D(y, vy)], [[], []]
    def node(value, edges):
        values.append(value); parents.append(edges); return len(values)-1
    a = node(values[0]*values[1], [(0, values[1]), (1, values[0])])
    b = node(sine(values[a]), [(a, cosine(values[a]))])
    c = node(values[0]*values[0], [(0, values[0]), (0, values[0])])
    node(values[b]+values[c], [(b, D(1)), (c, D(1))])
    bars = [D(0) for _ in values]; bars[-1] = D(1)
    for i in range(len(values)-1, -1, -1):
        for parent, partial in parents[i]: bars[parent] = bars[parent]+partial*bars[i]
    return (bars[0].t, bars[1].t)
for x, y in ((0., 0.), (1., .5)):
    for vx, vy in ((1., 0.), (0., 1.), (1., -1.)):
        actual = hvp(x, y, vx, vy)
        cross = cos(x*y)-x*y*sin(x*y)
        expected = ((2-y*y*sin(x*y))*vx+cross*vy, cross*vx-x*x*sin(x*y)*vy)
        assert max(abs(a-b) for a, b in zip(actual, expected)) < 1e-14
        print((x, y), (vx, vy), actual)
```

The code also checks a nonzero point against the displayed analytic matrix. It is an extra second-order tape example, not a new mode in the existing first-order UI. Hessian symmetry relies on smoothness; custom derivative rules or branch conventions at nonsmooth points require separate reasoning.

## Exercise: remove a path on paper

Pretend the square node sends only one x contribution. At (1,0) the resulting x derivative would be one rather than two, and JVP/VJP consistency against the correct tape would fail. List the contributions separately, then verify the true directional derivative with a moderate centered difference. Agreement near a smooth point provides evidence, while the next unit explains why finite-difference checks alone are insufficient at a kink.

Further reading: [JAX automatic differentiation](https://docs.jax.dev/en/latest/automatic-differentiation.html). This tape and graph are implemented directly for the stated function.
