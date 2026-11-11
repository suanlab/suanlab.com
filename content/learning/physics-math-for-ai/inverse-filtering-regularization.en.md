## Recovering an input needs more than a forward formula

An inverse problem starts with observations and asks which input produced them. This experiment uses periodic eight-sample convolution, two taps h=[1−r,r], and alternating observation noise. The reference truth combines a low-frequency cosine with a Nyquist component:

$$
x[n]=\cos(2\pi n/8)+\tfrac12(-1)^n,\qquad y=h*_8x+\eta(-1)^n.
$$

The Fourier-domain observation is Y=HX+E. A direct inverse divides by H only if every response bin is nonzero. Small nonzero values amplify noise; exact zeros destroy identifiability of the corresponding component. These are distinct problems.

## An exact zero cannot be inverted

At r=0.5, H[4]=1−2r=0. All values of the truth's alternating component produce the same noiseless observation. The direct inverse is therefore undefined. The interface draws no invented reconstruction curve and gives no reconstruction MSE for that request. Infinity in reciprocal gain describes a failed inverse, not a finite computed estimate.

With zero regularization, the minimum-norm pseudoinverse instead sets the missing coefficient to zero. It recovers the visible cosine but cannot infer the reference alternating amplitude 0.5. Its reconstruction MSE is 0.5²=0.25 even without recoverable error in the cosine. Selecting a minimum-norm solution is additional structure, not evidence of uniquely recovered truth.

## Stabilization trades fit for size

For positive λ the Tikhonov estimate is

$$
\widehat X[k]=\frac{\overline{H[k]}Y[k]}{|H[k]|^2+\lambda}.
$$

With our DFT convention this minimizes Σₙ|(h*_8z)[n]−y[n]|²+λΣₙ|z[n]|²; the common Parseval factor cancels. Positive λ suppresses unstable gains but also shrinks identifiable components. At r=0.5, noise amplitude 0.05 and λ=0.1, MSE is about 0.2555. The missing component still dominates the error. Blue is truth and orange the selected estimate; the table exposes observed blur plus noise separately.

## Near a zero is also dangerous

Set r=0.49, λ=0 and choose direct inversion. The Nyquist gain is 0.02, so observation noise 0.05 becomes input error amplitude 2.5. Its MSE is 2.5²=6.25. A tiny observation residual can coexist with a very poor reconstruction: fitting the noisy data exactly is not the same as recovering its source.

## Derive the optimizer and a uniform gain bound

Under the stated periodic boundary, DFT transforms the quadratic objective into (1/N)Σ_k(|H_k Z_k−Y_k|²+λ|Z_k|²). Minimizing each complex scalar gives the displayed conjugate filter. For λ>0 the Hessian is positive definite, hence the minimizer is unique. At λ=0, a zero H_k leaves that coordinate unobserved and the minimum-norm convention selects Z_k=0.

For a=|H_k| and λ>0, (a−√λ)²≥0 gives

$$
\frac{a}{a^2+\lambda}\le\frac1{2\sqrt\lambda}.
$$

Thus positive λ bounds every reconstruction gain, including near a zero. It does not recover the missing component of the true input.

## Separate bias and noise exactly in the two-mode example

Let d=1−2r and q=|H(π/4)|². The truth's cosine and alternating vector are orthogonal, with mean-square norms 1/2 and one. At positive λ the cosine error amplitude is −λ/(q+λ), and the alternating error amplitude is (dη−λ/2)/(d²+λ). Therefore

$$
\operatorname{MSE}=\frac12\left(\frac{\lambda}{q+\lambda}\right)^2+
\left(\frac{d\eta-\lambda/2}{d^2+\lambda}\right)^2.
$$

For d≠0,λ=0, the direct error is η²/d² and the exact-model observation residual is zero. For d=0,λ=0, use the pseudoinverse convention: MSE=1/4 and residual MSE=η². Do not evaluate the displayed fraction as 0/0.

| λ, r=.49, η=.05 | Reconstruction MSE | Observation residual MSE |
|---|---|---|
| 0 | 6.25000000 | 0 |
| .01 | .14799603 | .00338563 |
| .1 | .24368894 | .00826477 |
| 1 | .39432516 | .12781750 |

The code independently solves the eight-dimensional time-domain normal equations with 80-digit Decimal arithmetic and compares the resulting MSE with this mode formula. Regularization initially improves reconstruction while increasing residual, then excessive shrinkage raises reconstruction error again.

The live error readouts now use these analytic model amplitudes, not subtraction of rounded FFT reconstructions. At r=0,η=10⁻¹⁰⁰, direct MSE is 10⁻²⁰⁰; a roughly 10⁻³² FFT rounding artifact is not that model error. Positive error below binary64 range is shown as — with a numeric-range note. Exact noiseless direct inversion retains zero. The FFT curves and table remain finite-precision reconstructions; their subtraction errors need not equal the analytic model readout bit for bit.

**Exercise:** Does a zero observation residual certify the reference input? **Explanation:** No. The λ=0 noisy inverse fits the observation exactly while amplifying its alternating noise.

```python
from decimal import Decimal as D, localcontext

def solve(A,b):
    a=[list(row)+[v] for row,v in zip(A,b)]
    for j in range(len(a)):
        pivot=max(range(j,len(a)),key=lambda i:abs(a[i][j]))
        a[j],a[pivot]=a[pivot],a[j]
        value=a[j][j]
        a[j]=[v/value for v in a[j]]
        for i in range(len(a)):
            if i!=j:
                scale=a[i][j]
                a[i]=[v-scale*w for v,w in zip(a[i],a[j])]
    return [row[-1] for row in a]

with localcontext() as ctx:
    ctx.prec=80
    r,noise=D('.49'),D('.05')
    q=D(2).sqrt()/2
    truth=[1,q,0,-q,-1,-q,0,q]
    truth=[D(v)+D('.5')*(-1)**n for n,v in enumerate(truth)]
    H=[[((1-r) if i==j else D(0))+(r if i==(j+1)%8 else D(0)) for j in range(8)] for i in range(8)]
    observed=[sum(a*b for a,b in zip(row,truth))+noise*(-1)**i for i,row in enumerate(H)]
    for lam in [D(0),D('.01'),D('.1'),D(1)]:
        A=[[sum(H[k][i]*H[k][j] for k in range(8))+(lam if i==j else 0) for j in range(8)] for i in range(8)]
        rhs=[sum(H[k][i]*observed[k] for k in range(8)) for i in range(8)]
        estimate=solve(A,rhs)
        mse=sum((a-b)**2 for a,b in zip(estimate,truth))/8
        residual=sum((sum(a*b for a,b in zip(row,estimate))-y)**2 for row,y in zip(H,observed))/8
        h4=1-2*r
        h1_squared=(1-r)**2+r*r+2*r*(1-r)*q
        analytic=D('.5')*(lam/(h1_squared+lam))**2+((h4*noise-D('.5')*lam)/(h4*h4+lam))**2
        assert abs(mse-analytic)<D('1e-65')
        print(str(lam),round(mse,8),round(residual,8))
```

## Exercise and research connection

First set r=0 and noise zero. Direct inversion is identity and MSE vanishes. Then restore r=0.5 and compare direct, pseudoinverse and positive-λ estimates. Explain which outputs are undefined, which choose among compatible inputs, and which introduce shrinkage.

Deblurring, sensor reconstruction and learned inverse solvers require explicit boundary conditions, noise assumptions and priors. Regularization can select or stabilize a solution, but it cannot create observational identifiability. Report reconstruction error and observation residual separately when a reference truth is available.

Further reading: [Julius O. Smith III, Spectral Audio Signal Processing](https://www.dsprelated.com/freebooks/sasp/).
