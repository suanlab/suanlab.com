## Build a second moment from observation rows

Let A have m observation rows and d variable columns. S=AᵀA/m is an uncentered sample second moment. The model mean is known to be zero, so this lab neither subtracts a sample mean nor divides by m−1. Distinguish this convention from covariance preprocessing on measured data.

$$
S=\frac1m A^\top A,\qquad v^\top Sv=\|Av\|^2/m\geq0.
$$

## Actual eigenvalues of a small matrix

The lab generates a reproducible pseudo-Gaussian matrix and uses Jacobi diagonalization. Blue points are sorted finite eigenvalues; connecting lines are not a spectral density. Their sum should match the trace.

Floating-point diagonalization introduces error. Values with absolute magnitude at most 10⁻¹⁰ times the larger of one and maximum eigenvalue magnitude are displayed as numerical zero. The tolerance is stated explicitly. Numerical rank does not symbolically certify rank for arbitrary data.

## Marchenko–Pastur as a limiting reference

Consider ideal iid centered unit-variance entries with m,d increasing together and d/m→γ>0. Under suitable assumptions, nonzero bulk edges are (1±√γ)². For γ>1, a separate zero atom has mass 1−1/γ.

$$
\lambda_-=(1-\sqrt\gamma)^2,\quad\lambda_+=(1+\sqrt\gamma)^2,\quad
\text{zero atom mass}=\max(0,1-1/\gamma).
$$

Orange lines are these asymptotic references, not confidence intervals or deterministic ranges for the tiny current matrix. A finite eigenvalue outside them does not contradict the limit theorem.

## Fewer samples than variables

Since rank(AᵀA)≤min(m,d), dimension eight with three rows forces nullity at least five. A positive lower edge of the nonzero bulk does not remove the separate zero atom. An ideal Gaussian matrix has rank min(m,d) almost surely; the screen instead checks one fixed pseudo-random realization with a numerical tolerance.

## Interpreting AI spectra

Feature, gradient and covariance spectra can illuminate conditioning and low-dimensional structure. Dependence, anisotropy and spikes in actual learned data differ from an iid baseline. Specify normalization, centering, sample/variable axes and finite-versus-asymptotic interpretation before comparison.

## Exercises with explanations

1. What is forced when d>m? **A nullspace.**
2. Does an outside-bulk value violate the theorem? **No: the reference is asymptotic.**
3. Why is S positive semidefinite? **Every quadratic form vᵀSv is a squared norm divided by m.**

Further reading: [Vershynin, High-Dimensional Probability — publisher’s page](https://www.cambridge.org/core/books/highdimensional-probability/748AE7D1E34D1EB98E4D96E1659330FA).

[Vershynin, Introduction to the Non-Asymptotic Analysis of Random Matrices](https://arxiv.org/abs/1011.3027).
