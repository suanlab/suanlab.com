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

## Check a numerical example and its smoother

At ℓ=1, adjacent kernel values are exp(−1/2) and the endpoint value exp(−2). Zero regularization interpolates the three observations to numerical precision, not the entire quadratic. Positive regularization trades fit against norm. Inspect coefficients, predictions and linear-system residual together.

$$
\|f\|_H^2=\alpha^TK\alpha,\qquad S=K(K+\lambda I)^{-1}.
$$

The smoother maps labels to training predictions. Its trace is three at zero regularization and between zero and three otherwise. Blue is synthetic truth, orange regression and dots observations.

## Length scale differs from regularization

Length scale controls which locations interact. Lambda penalizes norm within the selected RKHS. Select them using held-out evidence or an explicit probabilistic model; fitting three points does not guarantee extrapolation or shifted-distribution performance.

Large datasets also incur quadratic Gram storage and cubic direct-factorization cost. Feature and Nyström approximations address computation but introduce a distinct approximation error.

## Read regularization one eigendirection at a time

For any real PSD training Gram matrix K=U diag(s_j)Uᵀ and λ>0, training predictions are Sy with shrinkage factors q_j=s_j/(s_j+λ). Positive eigenvalues are attenuated; nullspace components are discarded. The effective degrees of freedom ∑q_j measures the smoother's response to labels, not a count of statistically independent observations.

Fix the inputs and write y=f+ε, Eε=0, Cov(ε)=σ²I. Expectation here is over new noise vectors at these same inputs. In the eigenbasis f_j=(Uᵀf)_j,

$$
\frac1n E\|Sy-f\|^2
=\frac1n\sum_j(1-q_j)^2f_j^2
+\frac{\sigma^2}{n}\sum_j q_j^2.
$$

Expand Sy−f=(S−I)f+Sε. The cross term has zero expectation, and E‖Sε‖²=σ² tr(S²), proving the identity. It concerns prediction of the noiseless training-input values, not fresh-input population risk; predicting fresh noisy labels adds σ² per input. At λ=0 with a singular Gram, use the projection convention rather than an undefined inverse.

## Compute the tradeoff without a simulation assumption

Take K=[[1,1/2],[1/2,1]], f=(1,1), and σ²=1/4. Its eigenvalues are 3/2 and 1/2. The code averages the four independent noise vectors (±1/2,±1/2). Their covariance is exactly σ²I, so the identity applies without Gaussian noise.

| λ | Squared bias per input | Variance per input | Expected error |
|---|---|---|---|
| 0 | 0 | .25 | .25 |
| 1/4 | .020408 | .147392 | .167800 |
| 1 | .160000 | .058889 | .218889 |
| 4 | .528926 | .010841 | .539766 |

All numbers come from a separate two-input model; the live RBF controls remain unchanged. Larger λ reduces this variance but does not monotonically reduce their sum. A small numerical solve residual proves neither a good λ nor good prediction on shifted inputs.

**Exercise:** Why may a signal component in ker K incur bias even with perfect optimization? **Explanation:** Every kernel training prediction lies in range K, so that component cannot be represented at those inputs by the chosen kernel.

```python
from fractions import Fraction as F

sigma_squared=F(1,4)
# K=[[1,1/2],[1/2,1]], f=(1,1): squared signal is 2 in the top eigendirection.
eigenvalues=[F(3,2),F(1,2)]
for lam in [F(0),F(1,4),F(1),F(4)]:
    shrink=[s/(s+lam) for s in eigenvalues]
    bias=2*(1-shrink[0])**2/2
    variance=sigma_squared*sum(q*q for q in shrink)/2
    # Independent enumeration of four zero-mean noise vectors of covariance sigma^2 I.
    risk=F(0)
    for e1 in [F(-1,2),F(1,2)]:
        for e2 in [F(-1,2),F(1,2)]:
            # Project onto unnormalised (1,1),(1,-1), then shrink and reconstruct.
            plus=shrink[0]*(1+(e1+e2)/2)
            minus=shrink[1]*(e1-e2)/2
            pred=[plus+minus,plus-minus]
            risk+=sum((v-1)**2 for v in pred)/8
    assert risk==bias+variance
    print(str(lam),round(float(bias),6),round(float(variance),6),round(float(risk),6))
```

## Exercises with explanations

1. Does larger lambda always improve truth approximation? **No.** It changes a bias–variance tradeoff.
2. Do negative coefficients invalidate PSD? **No.** Coefficient signs and kernel quadratic forms are different issues.
3. Does a small system residual imply small population risk? **No.** Numerical accuracy is separate from statistical performance.

Further reading: [Rasmussen and Williams, Regression](https://gaussianprocess.org/gpml/chapters/RW2.pdf).
