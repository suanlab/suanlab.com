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

## Exact Gaussian moments at finite sample size

Now assume every entry of A is an ideal independent N(0,1) variable. Positive integers m,d are finite, and the mean is known to be zero, so keep S=AᵀA/m. Do not substitute formulas for an estimator subtracting a fitted sample mean. Integration by parts gives E[G²]=1 and E[G⁴]=3.

Each diagonal S_ii averages m squared Gaussians, giving mean one and variance 2/m. For distinct columns i≠j, S_ij has mean zero; independence across rows and between columns gives E[S_ij²]=1/m. Since symmetry gives tr(S²)=Σ_i,j S_ij²,

$$
E\operatorname{tr}S=d,\qquad
E\operatorname{tr}(S^2)=d+\frac{d^2+d}{m},\qquad
\operatorname{Var}\!\left(\frac{\operatorname{tr}S}{d}\right)=\frac{2}{dm}.
$$

Do not confuse tr(S²) with (tr S)². The latter has expectation d²+2d/m. The displayed second spectral moment includes off-diagonal terms and equals the sum of squared eigenvalues.

## Trace concentration does not control the whole spectrum

For t>0, Chebyshev gives P(|tr S/d−1|≥t)≤min(1,2/(dmt²)). The mean eigenvalue concentrates, but this alone does not control each eigenvalue or conditioning. Deterministic PSD matrices I_3 and diag(3,0,0) both have trace three, yet operator norms one and three and tr(S²) values three and nine. These are comparison matrices, not claimed Gaussian samples.

![Two PSD matrices have trace 3 but eigenvalues 1,1,1 versus 3,0,0. Their squared spectral sums and operator norms differ.](/assets/learning/high-dimensional-trace-spectra-en.svg)

If m<d, at least d−m eigenvalues vanish; ideal Gaussian sampling gives exactly that nullity almost surely. This does not place every nonzero eigenvalue near one. When m,d grow with d/m→γ, E[tr(S²)/d]=1+(d+1)/m tends to 1+γ. This connects to an MP moment but does not itself prove spectral-distribution convergence or an extreme-eigenvalue theorem.

Below, Moment means E[tr(S²)/d] and Variance means Var(tr S/d).

| (d,m) | Moment | Variance |
|---|---|---|
| (2,1) | 4 | 1 |
| (2,2) | $5/2$ | $1/2$ |
| (3,2) | 3 | $1/3$ |

## Check polynomial integrals without Gaussian sampling

For each independent Gaussian coordinate use three nodes (−√3,0,√3) with weights (1/6,2/3,1/6). They exactly match moments through degree four: E[G]=E[G³]=0, E[G²]=1 and E[G⁴]=3. The trace statistics above are polynomials of degree at most four in each coordinate, so product quadrature integrates them exactly.

The code writes each node as √3·z, z∈{−1,0,1}, making Gram entries rational sums (3/m)Σz_ki z_kj. This three-point law is not the Gaussian law itself. It gives positive weight even to an all-zero matrix and therefore does not reproduce Gaussian rank probabilities. Only the specified polynomial moments are being checked. This is also separate from the screen's pseudo-Gaussian seed example.

```python
from fractions import Fraction as Q
from itertools import product
from math import prod

nodes, weights = [-1, 0, 1], {-1: Q(1, 6), 0: Q(2, 3), 1: Q(1, 6)}
for d, m in [(1, 2), (2, 1), (2, 2), (3, 2), (2, 3)]:
    total = mean_trace = mean_trace_square = mean_squared_trace = Q(0)
    for flat in product(nodes, repeat=m*d):
        weight = prod(weights[z] for z in flat)
        A = [flat[k*d:(k+1)*d] for k in range(m)]
        S = [[Q(3, m)*sum(A[k][i]*A[k][j] for k in range(m))
              for j in range(d)] for i in range(d)]
        trace = sum(S[i][i] for i in range(d))
        trace_square = sum(v*v for row in S for v in row)
        total += weight
        mean_trace += weight*trace
        mean_trace_square += weight*trace_square
        mean_squared_trace += weight*trace*trace
    assert total == 1 and mean_trace == d
    assert mean_trace_square == d+Q(d*d+d, m)
    assert mean_squared_trace == d*d+Q(2*d, m)
    variance = (mean_squared_trace-mean_trace**2)/(d*d)
    assert variance == Q(2, d*m)
    print("d, m, normalized second moment, normalized trace variance:",
          d, m, mean_trace_square/d, variance)
```

## Questions for research use

1. Can tr(S²) and (tr S)² be interchanged? **No: they are different moments.**
2. Does mean eigenvalue one certify small eigenvalues? **No: trace alone does not identify nullspaces or conditioning.**
3. What does three-node quadrature verify here? **Specified low-degree polynomial moments, not the whole spectrum law or rank probabilities.**

For Gaussian matrices and finite versus asymptotic results, see [Vershynin, §5.3.1·5.4.3](https://arxiv.org/html/1011.3027). The moments and rational quadrature above are derived directly for the stated model.

## Exercises with explanations

1. What is forced when d>m? **A nullspace.**
2. Does an outside-bulk value violate the theorem? **No: the reference is asymptotic.**
3. Why is S positive semidefinite? **Every quadratic form vᵀSv is a squared norm divided by m.**

Further reading: [Vershynin, High-Dimensional Probability — publisher’s page](https://www.cambridge.org/core/books/highdimensional-probability/748AE7D1E34D1EB98E4D96E1659330FA).

[Vershynin, Introduction to the Non-Asymptotic Analysis of Random Matrices](https://arxiv.org/abs/1011.3027).
