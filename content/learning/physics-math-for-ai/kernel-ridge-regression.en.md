## Fix the objective before solving

The inputs are −1,0,1 with labels 1,c,1. Synthetic truth x² is a reference; the learner uses only the kernel and observations. The RBF kernel is exp(−(x−z)²/(2ℓ²)), with unit amplitude.

$$
\min_{f\in H}\sum_i(f(x_i)-y_i)^2+\lambda\|f\|_H^2.
$$

This is a sum-loss convention. A mean-loss convention can introduce nλ in the Gram system. Align these scalings when comparing papers and code.

## Solve a finite system and predict

The representer theorem gives

$$
(K+\lambda I)\alpha=y,\qquad f(x)=k_x^T\alpha.
$$

The implementation uses Cholesky and triangular solves rather than an explicit inverse. It adds no hidden jitter. Distinct inputs and finite positive length scale make this RBF Gram positive definite even at λ=0. Other kernels or duplicate inputs require rechecking that assumption.

At ℓ=1, adjacent kernel values are exp(−1/2) and the endpoint value exp(−2). Zero regularization interpolates the three observations to numerical precision, not the entire quadratic. Positive regularization trades fit against norm. Inspect coefficients, predictions and linear-system residual together.

$$
\|f\|_H^2=\alpha^TK\alpha,\qquad S=K(K+\lambda I)^{-1}.
$$

The smoother maps labels to training predictions. Its trace is three at zero regularization and between zero and three otherwise. Blue is synthetic truth, orange regression and dots observations.

## Length scale differs from regularization

Length scale controls which locations interact. Lambda penalizes norm within the selected RKHS. Select them using held-out evidence or an explicit probabilistic model; fitting three points does not guarantee extrapolation or shifted-distribution performance.

Large datasets also incur quadratic Gram storage and cubic direct-factorization cost. Feature and Nyström approximations address computation but introduce a distinct approximation error.

## Exercises with explanations

1. Does larger lambda always improve truth approximation? **No.** It changes a bias–variance tradeoff.
2. Do negative coefficients invalidate PSD? **No.** Coefficient signs and kernel quadratic forms are different issues.
3. Does a small system residual imply small population risk? **No.** Numerical accuracy is separate from statistical performance.

Further reading: [Rasmussen and Williams, Regression](https://gaussianprocess.org/gpml/chapters/RW2.pdf).
