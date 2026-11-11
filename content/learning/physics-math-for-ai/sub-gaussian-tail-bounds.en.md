## An MGF condition stronger than finite variance

This chapter calls a centered variable sub-Gaussian when log E exp(λX)≤σ²λ²/2 for every real λ. The quantity σ² is a variance proxy; conventions can use different constants. The condition controls variance, but finite variance alone does not imply the condition.

$$
E[X]=0,\qquad \log E[e^{\lambda X}]\leq\frac{\sigma^2\lambda^2}{2}\quad\text{for every }\lambda\in\mathbb R.
$$

## Two laws sharing a variance

One law places equal mass at ±σ; the other is N(0,σ²). Both have variance σ² and admit this proxy. Their MGFs are cosh(σλ) and exp(σ²λ²/2). After taking logarithms, the blue sign-law curve lies below the orange quadratic.

At σ=1,λ=1, log cosh(1)≈.43378 while the Gaussian value is .5. Both equal zero at λ=0. The drawing covers [−3,3], but the definition concerns all real arguments, not merely the displayed interval.

## Obtain tails through Chernoff's argument

Apply Markov's inequality to exp(λX) for λ>0 and optimize at λ=t/σ². This gives a one-sided bound exp(−t²/(2σ²)). Apply the same argument to −X and add the two tails.

$$
P(|X|\geq t)\leq\min\!\left(1,2e^{-t^2/(2\sigma^2)}\right),\qquad t>0.
$$

## Equal bounds do not imply equal tails

At σ=1,t=2, the sign law has tail zero because its only values are ±1. The Gaussian tail is about .0455, while their common bound is about .27067. At t=σ, the inclusive sign-law tail is one. Distinguish atoms and inclusive thresholds from continuous Gaussian tails.

## Scope in research

A Laplace law can have finite variance without an MGF existing for all real arguments. An estimated variance therefore does not certify sub-Gaussianity. State centering, scale and any independence assumptions required by subsequent sum inequalities. Gaussian tail values here use a numerical CDF; the plot is not a fixed-seed sampling experiment.

## Computing a small log MGF

When cosh(x) is extremely close to one, direct logarithms or subtraction of similar constants can lose a quadratic-size value. For |x|<1 the implementation uses the following identity with log1p.

$$
\log\cosh x=\log\!\left(1+2\sinh^2(x/2)\right).
$$

At σ=1,λ=10⁻⁸, the value is approximately 5×10⁻¹⁷. This distinguishes an exact zero at λ=0 from a small positive value. If a mathematically positive selected log MGF falls below numeric representation range, the readout and marker are withheld with a numeric-range notice. Larger |x| retains the overflow-avoiding expression. This is not arbitrary-precision evaluation across every representable-range boundary.

## Exercises with explanations

1. Must σ² be the unique optimal constant? **No: any valid proxy can be used.**
2. Does finite variance suffice? **No: global MGF control is stronger.**
3. What is the inclusive tail at the sign atom? **One, including both atoms.**

Further reading: [Vershynin, High-Dimensional Probability — publisher’s page](https://www.cambridge.org/core/books/highdimensional-probability/748AE7D1E34D1EB98E4D96E1659330FA).
