## Interpolation and test risk are different questions

Fitting every training label does not determine risk on unseen inputs. Use n Gaussian observations, at most twelve features and true coefficient (1,0,...,0). Increasing p selects prefixes of one design, with the same fixed label noise. New noise is not drawn for each model to manufacture a desired curve.

Blue is actual training MSE. Orange is exact conditional MSE under a declared independent Gaussian test law. This chapter uses full MSE rather than half-squared loss. Markers identify the selected p, while the curves contain actual fits for every p from zero through twelve.

The featureless p=0 baseline actually predicts zero. Its training MSE is mean(y²) and test MSE is 1+sigma². At seed seven and sigma=.5, risk drops from baseline 1.25 to about .25 at p=1, rises to about .905 at p=4, then falls to about .628 at p=5. These are actual model computations, not a guarantee of the same shape under every setting.

## Minimum-norm least squares and numerical rank

For X_p=U Sigma Vᵀ, compute the pseudoinverse solution

$$
\widehat w_p=\sum_{\sigma_j>\tau}v_j\frac{u_j^Ty}{\sigma_j},\qquad
\tau=10^{-10}\sigma_{max}.
$$

A one-sided Jacobi SVD actually rotates columns. No hidden ridge is added to normal equations. Singular directions below the declared cutoff are omitted; numerical rank and retained singular condition are reported. Even p≥n does not guarantee interpolation without inspecting rank and tolerance. The retained condition is not a full-rank condition number including discarded zero directions.

## Derive exact conditional test risk

Let independent test input z follow N(0,I₁₂), with label z₁+sigma epsilon. Pad the learned p-dimensional coefficient with zeros. Independence and isotropic covariance give

$$
E[(z^Tw-z_1-\sigma\epsilon)^2\mid X,y]
=\|w-e_1\|^2+\sigma^2.
$$

This evaluates population risk conditional on the selected training realization. It is not an observed average over fabricated test examples. Coefficient bias, variance and conditioning can make it large. One seed need not yield a smooth curve or exhibit two descents.

## Separate a conditional curve from an average over designs

For an exact pseudoinverse write P=X⁺X and y=X*theta+noise, with independent centered training noise of covariance sigma²I. Bias lies in the design nullspace while X⁺noise lies in its row space. Orthogonality removes their cross term even before averaging noise. Under identity test covariance,

$$
E_{noise}[R(\widehat w)\mid X]=\|(I-P)\theta\|^2+\sigma^2\|X^+\|_F^2+\sigma^2.
$$

Thus a tiny retained singular value contributes sigma²/s_j² in expectation, even though a particular realized noise may align weakly with that direction. The last term is independent test noise. This conditional-noise average differs from the live curve, which is conditional on both a particular design and its actual training noise.

For an auxiliary standard Gaussian n-by-p design, exact untruncated least squares, unit-norm well-specified theta and sigma>0, inverse-Wishart moments give training-noise terms sigma²*p/(n−p−1) when p<n−1 and sigma²*n/(p−n−1) when p>n+1. Rotational symmetry gives overparameterized bias 1−n/p. These moments require the stated denominators to be positive. At p=n−1,n,n+1 the expected noise risk is infinite, though an individual full-rank realization is finite almost surely. A finite plot cannot refute an infinite expectation.

The moment formula has a short geometric proof. For an n-by-p Gaussian design with n≥p, a Schur complement expresses a diagonal entry of (XᵀX)^(−1) as the inverse squared length of one column after projecting away the others. That residual has n−p+1 independent Gaussian coordinates. If V has chi-square degrees nu, its gamma density gives E[1/V]=1/(nu−2) for nu>2; the integral diverges for nu≤2. Summing p diagonal entries proves the underparameterized term. Applying the same argument to Xᵀ proves the overparameterized term. Finally rotational invariance and trace(P)=n give E[P]=(n/p)I, proving the bias term. These are distributional expectations with exact inverse algebra, not tolerances of the numerical solver.

![Finite Gaussian-design expected test risk with the dimensions of divergent noise expectation marked, distinct from a single fitted design.](/assets/learning/theory-interpolation-en.svg)

| n=6, sigma=1/2, auxiliary p | Expected total test MSE |
|---|---:|
| 4 | 5/4 |
| 6 | infinite |
| 8 | 2 |

```python
from itertools import product
from fractions import Fraction as F
sigma=F(1,2);values=[]
for signs in product((-1,1),repeat=2):
    w=(1+sigma*signs[0]/2,1+sigma*signs[1],F(0))
    values.append(sum((x-1)**2 for x in w)+sigma*sigma)
expected=sum(values)/4
assert expected==1+sigma*sigma*(F(1,4)+1)+sigma*sigma==F(25,16)
print('fixed diagonal design:',expected)
n=6
for p in (4,6,8):
    total=(sigma*sigma*(1+F(p,n-p-1)) if p<n-1 else sigma*sigma+1-F(n,p)+sigma*sigma*F(n,p-n-1) if p>n+1 else None)
    print('Gaussian design expectation:',p,total)
```

The rational diagonal example checks the conditional-noise decomposition independently. The Gaussian expectation figure has no numerical singular cutoff and extends to p=16; the live nested design stops at twelve features and has cutoff 1e-10. They answer different questions. The [ridgeless regression analysis](https://arxiv.org/abs/1903.08560) provides broader research context; this finite calculation is not a reproduction of its full experiments or asymptotic results.

## Worked example and exercise

As a separate minimal hand calculation, let X=[1,1] and y=1. The minimum-norm solution (.5,.5) has zero training loss. If the true coefficient is (1,0) and test covariance is identity, its noiseless test MSE is .5. The true coefficient also fits training but has test risk zero. Minimum norm is not optimal for every possible true model.

Exercise: add independent test label noise sigma=.2. Find both risks. Solution: the minimum-norm risk becomes .5+.04=.54, while the true coefficient has risk .04. This irreducible floor is common to both. Training interpolation does not eliminate it.

## Observe and verify double descent

Near interpolation, small singular values can amplify noise. Additional features change the minimum-norm solution and may reduce risk. Shape nevertheless depends on feature law, target, noise and regularization. The interface shows only actual finite computations. Record multiple seeds and sizes together with rank, conditioning and solver tolerance before making general claims.

Independent checks use all four Moore–Penrose identities, known hand solutions, normal residuals and preservation of nested columns and fixed noise. One picture neither refutes all earlier theory nor guarantees generalization for every overparameterized model.

References: [the original Belkin et al. paper](https://arxiv.org/abs/1812.11118) and [Understanding Machine Learning author page](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/). Curves and examples are independently implemented.
