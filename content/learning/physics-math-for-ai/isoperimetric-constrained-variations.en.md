## A global integral constraint restricts directions

Minimize derivative energy J[y]=∫₀¹(y′)²/2 dt with y(0)=y(1)=0 and prescribed signed area ∫₀¹y dt=A. Negative A is allowed; it represents a curve below the baseline rather than a negative probability.

A candidate curve can satisfy the endpoints while violating area. Likewise, a direction can vanish at the endpoints while changing area. Feasibility of the current curve and admissibility of a variation are separate tests.

## Use a multiplier with an explicit sign

Define L[y,λ]=J[y]+λ(∫y−A). Integration by parts gives

$$
\delta L[y;h]=\int_0^1(-y''+\lambda)h\,dt,
\qquad -y''+\lambda=0.
$$

The endpoint conditions make y=−λt(1−t)/2. Since ∫₀¹t(1−t)dt=1/6, the area condition yields

$$
y^*(t)=6A\,t(1-t),\qquad \lambda^*=-12A,\qquad J[y^*]=6A^2.
$$

Changing the sign convention for L would change the multiplier's sign without changing the minimizing curve. Equality multipliers are not restricted to be nonnegative.

## An exact minimum, not just a stationary parabola

For a feasible difference h=y−y*, endpoints and area both vanish. The cross term is ∫y*′h′=−∫y*″h=12A∫h=0. Therefore J[y]−J[y*]=∫(h′)²/2≥0. Fixed endpoints eliminate a nonzero constant difference, giving uniqueness.

This proof uses feasibility. A lower energy at a curve with the wrong area is not a counterexample to the constrained optimum. The lab withholds a feasible objective-excess certificate for such candidates.

## Read the candidate and direction controls separately

The candidate is y=6At(1−t)+b sin 2πt+v sin πt. The second sine mode integrates to zero, while the first integrates to 2/π. Thus the candidate area residual is 2v/π and feasibility requires v=0.

The direction is h=sin 2πt+c sin πt, with area variation 2c/π. It is admissible exactly when c=0. At a constrained optimum, δJ must vanish on admissible directions; it need not vanish on arbitrary directions. The multiplier balances the latter through δL=δJ+λδG.

## Exercise and explained answer

Choose A=0.5,b=v=c=0. The curve is 3t(1−t), energy 1.5 and multiplier −6. Set b=0.2: area stays 0.5, but objective excess becomes 0.04π²≈0.394784. The variation along sin 2πt is 0.4π², so this oscillating feasible curve is not stationary.

Restore b=v=0 and set c=1. The direction is no longer area-preserving. Its objective variation is 12/π, canceled by λδG=−12/π. Explain why a nonzero δJ in this inadmissible direction does not contradict constrained optimality. Integral constraints of this type appear in mass conservation, normalized density estimation and variational physical models, with domains and signs adapted to each application.

Further reading: [MIT, Calculus of Variations](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec10.pdf).
