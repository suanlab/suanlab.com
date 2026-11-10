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

## Exercise and explained answer

Start from the broken quarter-turn example, change only θ to π, and explain the zero discrepancy using −I. Then restore π/2 and set c=0. The repaired map now satisfies a proved matrix identity for every angle. Try a nonzero b to see that equivariance does not require outputs to equal inputs.

## Research connection

Architectural equivariance can encode known structure, while numerical tests catch implementation mistakes. Neither replaces choosing the correct task symmetry. Boundary handling, interpolation, discretization and nonlinear layers can break a mathematical continuum equivariance condition. Report the precise representations and tested transformations alongside any measured equivariance error.

Further reading: [Cohen and Welling, Group Equivariant Convolutional Networks](https://proceedings.mlr.press/v48/cohenc16.html).
