## A lower bound and an equality theorem are different claims

For feasible inequality multipliers, the Lagrangian infimum gives a lower bound on every feasible primal objective. Maximizing those bounds gives d*≤p*: weak duality. Strong duality means equality of optimal values. Whether a supremum is attained by a finite multiplier is a separate question.

Slater's condition is a sufficient qualification for convex problems with finite optimum: a feasible point in the appropriate relative interior strictly satisfies the nonlinear inequalities, with the usual affine-equality qualifications. Failure of this sufficient condition does not by itself determine whether the values coincide.

## Strict feasibility with an attained multiplier

The first mode uses min (x−2)²/2 subject to x≤1. Choosing x=0 is strictly feasible. Primal optimum is 0.5 at x=1, and

$$
d(\lambda)=\lambda-\lambda^2/2,\quad\lambda\ge0,
\qquad\lambda^*=1,\quad d^*=0.5.
$$

The plotted dual curve touches the primal reference level. The witness-size control is unused in this regular example.

## Equal values without a finite dual maximizer

Now minimize x subject to x²≤0. Only x=0 is feasible, so p*=0, and no strict feasible point exists. For λ>0, minimizing x+λx² gives

$$
x_\lambda=-1/(2\lambda),\qquad d(\lambda)=-1/(4\lambda).
$$

At λ=0 the infimum is negative infinity, witnessed by x=−M as M grows. For positive λ, d approaches zero only as λ→∞. Thus d*=p*=0, but no finite optimal multiplier exists. KKT stationarity at the primal optimum would require 1+2λ·0=0, which is impossible. Convexity alone did not guarantee multipliers.

## A genuine positive gap

The third example is adapted explicitly from Boyd and Vandenberghe's exercise 5.21:

$$
\min_{x\in\mathbb R,\ y>0} e^{-x}
\quad\text{subject to}\quad x^2/y\le0.
$$

The perspective quadratic x²/y is convex on y>0 and nonnegative. Feasibility forces x=0, giving attained primal value one. For every finite λ≥0 the Lagrangian is nonnegative, while the sequence x=M,y=M⁴ gives e⁻ᴹ+λ/M²→0. Therefore d(λ)=0 for every feasible λ and the optimal-value gap is one.

Those sequence points lie in the function domain but are not primal feasible. They witness an unconstrained Lagrangian infimum, not better feasible solutions. The dual supremum is attained here, although its inner Lagrangian infimum is not. Distinguish all three attainment statements.

## Exercise and interpretation

In mode two increase M from four to sixteen. The witness Lagrangian value decreases toward zero, while the primal optimum remains one. Explain why this supports the calculated dual infimum without refuting weak duality. Compare with the second mode, where the value gap is zero but multiplier attainment fails.

When using a dual certificate in AI optimization, state the primal/dual domains, convexity and qualification assumptions. A plotted finite grid cannot establish an unbounded-domain infimum; the explicit sequences and lower-bound arguments provide the missing reasoning.

Further reading: [Boyd and Vandenberghe, Convex Optimization](https://stanford.edu/~boyd/cvxbook/).
