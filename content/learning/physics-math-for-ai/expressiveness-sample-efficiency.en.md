## Perfect observations need not identify a function

Suppose n uniform locations in [−1,1] observe synthetic truth f*=x². Identification depends on the allowed function class and prior. Add a polynomial vanishing at every sample:

$$
p_n(x)=\prod_{i=1}^n(x-x_i),\qquad p_n(x_i)=0.
$$

This is a **constructed ambiguity example**, not a claim that a particular learner returns this alternative. Truth is available only because the experiment is synthetic.

## Normalize population ambiguity

Use population measure dx/2 and qₙ=pₙ/‖pₙ‖₂. Then alternatives fA=f*+Aqₙ have identical training values but

$$
E_{X\sim U[-1,1]}[(f_A(X)-f_*(X))^2]=A^2.
$$

At n=5,A=1, training MSE is zero and population MSE one. Halving amplitude leaves training error zero and reduces population error to 1/4. Compare every table row with the between-sample orange curve.

## Interpret growing sample count carefully

At n=9 the construction uses a new ninth-degree vanishing polynomial. The candidate family grows with n. This is not a learning curve for a fixed quadratic model or evidence against consistency of a specific learner. It shows that an unrestricted class permits new ambiguities.

Expressiveness asks whether a class contains a target. Sample efficiency asks how much data suffices under a class, distribution, noise model, tolerance and confidence level. Parameter count alone cannot equate them. Norm, smoothness, margin or symmetry can constrain between-sample behavior.

## Connect to kernels and generalization

This resembles the representer lesson's invisible directions, now measured by population risk. Norm regularization or implicit bias can select among interpolating functions, but useful generalization needs additional assumptions and evidence.

Interpolation is not inherently bad. It is simply insufficient by itself as a guarantee. Use independent test evidence and a complexity analysis appropriate to the chosen class.

## Exercises with explanations

1. What if A=0? **Both functions and risks coincide at zero error.**
2. Why does risk remain one with more samples? **The alternative family also expands.** This is not a fixed learner's failure curve.
3. Is population risk merely a grid estimate? **No.** Polynomial-square integration normalizes it analytically and independent quadrature verifies it.

Further reading: [MIT 9.520, Statistical Learning Theory](https://ocw.mit.edu/courses/9-520-statistical-learning-theory-and-applications-spring-2003/).
