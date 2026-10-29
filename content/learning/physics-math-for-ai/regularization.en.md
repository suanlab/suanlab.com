## Fitting well can still be unstable

A weakly measured direction can demand a very large coefficient to explain a small observation. Slight noise then changes the recovered coefficient substantially. Regularization adds a preference among solutions. It can stabilize inversion, but this preference can introduce bias and cannot manufacture missing observations.

Use $X=\operatorname{diag}(3,s)$ and $y=(3,y_2)^T$. The first direction is strongly measured; the second has scale $s$. The lab varies this weak scale, its observed response and a nonnegative ridge penalty.

## Define the objective and units

Ridge minimizes

$$
J(w)=\frac12\lVert y-Xw\rVert^2+\frac\lambda2\lVert w\rVert^2.
$$

The residual term measures fit; the second term favors small coefficients in the chosen coordinates. Feature scaling changes this preference. If quantities have physical units, the penalty coefficient needs compatible units or an explicit nondimensionalization.

| Symbol | Meaning |
| --- | --- |
| $s$ | Weak measurement scale |
| $\lambda\ge0$ | Penalty strength |
| $w_\lambda$ | Penalized solution |
| $X^TX+\lambda I$ | Curvature matrix of the objective |

## Derive the solution

Differentiate: $\nabla J=X^T(Xw-y)+\lambda w$. For $\lambda>0$, the curvature matrix is positive definite even when $X$ is rank deficient, yielding a unique minimizer:

$$
w_1=\frac9{9+\lambda},\qquad
w_2=\frac{s y_2}{s^2+\lambda}.
$$

At $s=0.1,y_2=1$, unpenalized inversion gives $w_2=10$. At $\lambda=0.1$, it becomes $10/11\approx0.909091$. The first coefficient is $90/91\approx0.989011$. Shrinking the weak direction improves numerical stability while leaving a larger residual for the second observation.

The displayed curvature condition number is $(9+\lambda)/(s^2+\lambda)$ within the lab’s scale range. This is a condition number of the penalized normal matrix, not of the original observation matrix.

## Understand the spectral filter

Along a singular direction of scale $\sigma$, the ridge inverse factor is $\sigma/(\sigma^2+\lambda)$ rather than $1/\sigma$. The ratio of the ridge estimate to the unpenalized estimate is $\sigma^2/(\sigma^2+\lambda)$. Small singular directions are suppressed most strongly.

This filter gives a concrete connection to SVD. It does not prove that every suppressed direction is irrelevant to the scientific task. If the true signal lies in a weakly observed direction, shrinkage may remove valuable signal along with noise.

## Rank deficiency and a chosen solution

At $s=0,\lambda=0$, the second observation cannot constrain $w_2$. Every value gives the same data fit. The lab displays $w_2=0$ as a minimum-norm selection, with an explicit warning. It does not claim the data proved this value.

For positive $\lambda$, the penalized objective selects zero in that direction, but the missing information remains missing. Distinguish objective uniqueness from physical identifiability.

## What the chart compares

The chart plots both coefficient solutions over a range of penalty strengths for the current $s,y_2$. The numeric table evaluates the currently selected penalty and separately reports residual cost and penalty cost. A larger penalty can make the chosen objective easier to optimize while worsening unpenalized fit. Comparing only the combined objective across different penalty strengths is not a valid predictive comparison.

## Selection and research design

Choose a penalty using a declared validation procedure, without reusing the final test set. Compare prediction error, coefficient sensitivity and cost. Report feature scaling and whether the intercept is penalized; this example has no intercept. A Bayesian interpretation is possible under an explicitly Gaussian likelihood and prior, but arbitrary ridge use does not by itself establish those assumptions.

For a study, perturb the observation by a known amount, compare parameter changes with and without ridge, and evaluate against known synthetic truth. Then separately test whether the same trade-off helps held-out real observations. Numerical conditioning, estimator variance and generalization are related questions with different definitions.

## Exercises and source

1. With $s=0.1,y_2=1,\lambda=0.01$, compute $w_2$. **Answer:** 5.
2. Show that $\lambda\to\infty$ sends both coefficients to zero.
3. Explain why a unique penalized minimizer need not identify the true unobserved parameter.
4. Change the units of the second feature and state how an equivalent penalty would need to change.

For regression and regularization background, see the authors’ [Introduction to Statistical Learning resources](https://www.statlearning.com/). The diagonal experiment is an independently derived example, not a reproduced empirical result from that book.
