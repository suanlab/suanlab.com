## Preserving a value differs from transforming an output

An invariant scalar satisfies q(gx)=q(x). An equivariant map satisfies f(ρin(g)x)=ρout(g)f(x), with specified input and output representations. Invariance is a special equivariance condition with trivial output action. A vector should often rotate with its input rather than stay numerically fixed.

For planar rotations Rθ, squared norm is invariant because RθᵀRθ=I. The experiment separately compares two vector-output paths: rotate the input then apply W, or apply W then rotate the output. Blue and orange endpoints coincide exactly when those paths agree for the selected input and angle.

## Solve a small linear family

Use x=(1,0.5) and

$$
W=\begin{pmatrix}a+c&-b\\b&a-c\end{pmatrix}
=aI+bJ+c\operatorname{diag}(1,-1),\quad
J=\begin{pmatrix}0&-1\\1&0\end{pmatrix}.
$$

When c=0, W is a scaling-plus-rotation map and commutes with every Rθ. Conversely, commuting with the quarter-turn J forces equal diagonal entries and opposite off-diagonal entries. Thus within this family the all-angle SO(2) equivariance condition is exactly c=0.

This is a statement about real vector representations. It does not imply that only scalar multiples of identity are allowed: the J term survives for SO(2). Requiring reflections as well would remove it, as the Reynolds-projection unit demonstrates.

## A numerical counterexample

Set a=1,b=0,c=0.5 and θ=π/2. Then W=diag(1.5,0.5), giving

$$
WR_\theta x=(-0.75,0.5),\qquad
R_\theta Wx=(-0.25,1.5),\qquad
\|WR_\theta x-R_\theta Wx\|=\sqrt{1.25}.
$$

The unit reference circle is only an output-coordinate guide. It is not a claim that either endpoint must have unit norm. The separate squared-norm readout checks invariance of the input norm, not invariance of arbitrary anisotropic W outputs.

## Why one successful test is insufficient

At θ=π the rotation is −I, which commutes with every linear W. The selected discrepancy becomes zero even with c=0.5. The all-angle indicator correctly remains false. More generally, testing one x only probes a commutator on that vector; a nonzero matrix can annihilate a particular input. Universal equivariance quantifies over both inputs and group elements.

## Derive all continuous planar radial vector laws

For a nonzero input, the pair x,Jx is a basis. Write an SO(2)-equivariant output uniquely in that basis. Rotating x rotates both basis vectors, so their coefficients must be constant on a circle. Therefore the general form away from the origin is

$$
f(x)=\alpha(\|x\|^2)x+\beta(\|x\|^2)Jx,\qquad x\ne0.
$$

At zero, equivariance forces f(0)=0 because no nonzero vector is fixed by every rotation. Continuity there additionally requires the output norm to tend to zero; continuity of α and β at zero is sufficient but not necessary. Requiring reflection equivariance removes β, since a reflection anticommutes with J. A norm-dependent scalar gate is thus compatible with vector rotations, while coordinatewise ReLU generally is not.

For the linear family in this lab, the exact commutator norm on x=(1,1/2) is √5 |c sinθ|. The readout now evaluates this identity directly. With c=10⁻²⁰ and θ=π/2, the true model discrepancy is √5·10⁻²⁰ even though forming 1±c rounds both diagonal entries to one. The displayed matrix products are separate floating-point evaluations, so subtracting their rounded endpoints can erase the discrepancy.

| Map | SO(2) equivariant | O(2) equivariant |
|---|---|---|
| (1+‖x‖²)x | Yes | Yes |
| Jx | Yes | No |
| Coordinatewise ReLU | No | No |

The code tests a mixed radial law at several inputs and angles, then exhibits reflection failure of its J component. The derivation, rather than finite testing alone, establishes the universal law.

## Reproduce the calculation in Python

```python
from math import sin, cos, hypot
def rot(t,x): return [cos(t)*x[0]-sin(t)*x[1],sin(t)*x[0]+cos(t)*x[1]]
def law(x):
    r2 = sum(v*v for v in x)
    return [(1+r2)*x[0]-.3*x[1], (1+r2)*x[1]+.3*x[0]]
errors = []
for x in [[0,0],[1,.5],[-.2,2]]:
    for t in [.2,1.1,2.4]:
        a,b = law(rot(t,x)),rot(t,law(x))
        errors.append(hypot(a[0]-b[0],a[1]-b[1]))
assert max(errors) < 1e-13
x=[1,.5]; left=law([x[0],-x[1]]); right=law(x); right[1]*=-1
assert hypot(left[0]-right[0],left[1]-right[1]) > .5
print('max rotation error:', max(errors))
print('reflection discrepancy:', hypot(left[0]-right[0],left[1]-right[1]))
```

## Exercise and explained answer

Start from the broken quarter-turn example, change only θ to π, and explain the zero discrepancy using −I. Then restore π/2 and set c=0. The repaired map now satisfies a proved matrix identity for every angle. Try a nonzero b to see that equivariance does not require outputs to equal inputs.

## Research connection

Architectural equivariance can encode known structure, while numerical tests catch implementation mistakes. Neither replaces choosing the correct task symmetry. Boundary handling, interpolation, discretization and nonlinear layers can break a mathematical continuum equivariance condition. Report the precise representations and tested transformations alongside any measured equivariance error.

Further reading: [Cohen and Welling, Group Equivariant Convolutional Networks](https://proceedings.mlr.press/v48/cohenc16.html).
