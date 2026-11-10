## A slope can support a function without being its derivative

For a convex function f, a subgradient v at u defines an affine global lower bound. The condition is not merely a local tangent approximation:

$$
v\in\partial f(u)\quad\Longleftrightarrow\quad
f(x)\ge f(u)+v(x-u)\quad\text{for every }x\in\operatorname{dom}f.
$$

When f is differentiable, the subdifferential contains its single gradient. At a cusp it may contain many slopes. These are not guesses of a missing classical derivative; they are valid global support certificates with their own definition.

## Derive the subdifferential in the lab

Take f(x)=|x|+γx²/2 with γ≥0. Away from zero, the derivative is sign(x)+γx. At zero, the quadratic contributes derivative zero while |x| permits slopes between −1 and one:

$$
\partial f(u)=\begin{cases}[-1,1],&u=0,\\
\{\operatorname{sign}(u)+\gamma u\},&u\ne0.
\end{cases}
$$

For γ=0,u=0,v=0.5, the line 0.5x lies below |x| everywhere. At query x=2 the residual is 2−1=1, yet the global minimum residual is zero at the touching point. With slope 1.5, residual becomes −0.5x for positive x and tends to negative infinity. The finite plot is not the domain of the definition.

## Calculate the global residual through a conjugate

The Fenchel conjugate is f*(v)=supₓ(vx−f(x)). For positive γ, optimizing separately on the positive and negative half-lines gives

$$
f^*(v)=\frac{(\max\{|v|-1,0\})^2}{2\gamma},\qquad
\inf_x\{f(x)-f(u)-v(x-u)\}=vu-f(u)-f^*(v).
$$

For γ=0 the conjugate is zero on [−1,1] and positive infinity outside. A valid subgradient makes this infimum zero. An invalid slope can give a finite negative residual when γ>0, or an unbounded negative residual when γ=0 and |v|>1. The interface distinguishes these cases and hides a finite analytic value that exceeds numerical representation range instead of relabeling it mathematical infinity.

## Exercise and explained answer

Choose u=1,γ=1,v=2. The only subgradient is 1+1=2. At x=−1, f(−1)=f(1)=1.5 and the line value is 1.5+2(−2)=−2.5, giving residual four. The conjugate is 0.5 and the global minimum residual is 2−1.5−0.5=0. Verify each number, then perturb v and see why a single positive query residual need not certify support everywhere.

## Optimality and learning

For a proper convex objective, zero in the subdifferential characterizes a global minimizer. Constraints can be included through indicator functions and normal cones, subject to the appropriate calculus rules. Subgradient descent and proximal algorithms use this structure differently; an arbitrary slope chosen inside a cusp set is not automatically an optimal algorithmic update. Separate the geometric support definition from claims about convergence rates or learning performance.

Further reading: [Parikh and Boyd, Proximal Algorithms](https://stanford.edu/~boyd/papers/prox_algs.html).
