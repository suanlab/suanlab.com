## When can function search become finite coefficient search?

The representer theorem applies when loss depends on finitely many function evaluations and regularization grows appropriately with RKHS norm. If a minimizer exists, one can choose it in the observed kernel-section span. A strictly increasing penalty removes unnecessary orthogonal components. Existence, uniqueness and coefficient uniqueness need separate assumptions.

$$
f=f_\parallel+f_\perp,\qquad f_\perp\perp\operatorname{span}\{k_{x_i}\}.
$$

Reproduction gives f⊥(xᵢ)=0, leaving data loss unchanged. Squared norm splits orthogonally, so discarding the complement reduces penalty.

## Construct an invisible direction

Use features (1,x,x²), kernel 1+xz+x²z² and observations (−1,1),(1,1). The unit coefficient vector v=(1,0,−1)/√2 is orthogonal to both observed features:

$$
v\cdot\varphi(x)=\frac{1-x^2}{\sqrt2},\qquad v\cdot\varphi(\pm1)=0.
$$

Adding cv preserves training predictions but changes the central value by c/√2. Equal observations do not identify an entire function.

## Solve the penalized example

The objective is mean squared residual plus λ‖w‖² with λ>0. Symmetry and stationarity give w=(t,0,t), t=1/(2+λ). Adding the complement yields norm squared 2t²+c²:

$$
J(w+cv)-J(w)=\lambda c^2.
$$

At λ=1, t=1/3, training MSE is 1/9 and minimum objective 1/3. Adding c=1 gives objective 4/3 without changing observations. At λ=.1, the same perturbation costs .1. Blue is the minimizer, orange the alternative and dots the data.

## What the theorem does not guarantee

It does not establish accurate extrapolation. Derivative or integral observations require checking their continuous representatives. Singular Gram matrices can also allow different coefficient vectors representing one function: function and coefficient uniqueness differ.

This finite RKHS exposes the same orthogonal argument used in infinite-dimensional kernel regression. Numerical conditioning and statistical generalization remain separate.

## Exercises with explanations

1. Double c: added cost? **Four times larger**, because the penalty is quadratic.
2. Does invisibility at training points imply invisibility everywhere? **No:** the central change is a counterexample.
3. Must the complement vanish at λ=0? **No.** Without a penalty multiple minimizers can remain. The controls here keep λ positive.

Further reading: [Schölkopf, Herbrich and Smola, A Generalized Representer Theorem](https://alex.smola.org/papers/2001/SchHerSmo01.pdf).
