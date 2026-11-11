## Reduce noise without changing the mean direction

A control variate subtracts a correlated quantity with known expectation, then adds that expectation back. The goal is to reduce variance while preserving the desired mean. In finite-sum optimization, evaluating the same component gradient at two nearby points provides such a correlation.

The lab uses fᵢ(x)=hᵢ(x−aᵢ)²/2 with h=(1,2,3,4) and a=(−1,1,2,0). The full gradient is G(x)=2.5x−1.75, so the optimizer is x*=0.7. Individual gradients need not vanish there: the component minimizers disagree.

## Derive the SVRG estimator

Compute the full gradient once at an anchor x̃, then sample a uniform component and form

$$
v_i(x,\widetilde x)=g_i(x)-g_i(\widetilde x)+G(\widetilde x),
\qquad\mathbb E_i[v_i]=G(x).
$$

The two component gradients must use the same index. Subtracting an unrelated component can preserve the mean under extra independence assumptions but changes the covariance and loses the intended cancellation.

For this quadratic family the target-dependent terms cancel exactly:

$$
v_i=h_i(x-\widetilde x)+2.5\widetilde x-1.75,
\qquad\operatorname{Var}(v_i)=1.25(x-\widetilde x)^2.
$$

The factor 1.25 is the population variance of h=(1,2,3,4). The anchor-distance formula is specific to this example, not a universal SVRG variance identity.

## Three illuminating anchors

At x=x̃=0.7 all SVRG estimates are zero. Ordinary SGD gradients are (1.7,−0.6,−3.9,2.8), with mean zero and variance 6.575. This nonzero optimum noise is possible because the component objectives do not interpolate a shared minimizer.

Keep x=0.7 and move the anchor to −1. The SVRG variance becomes 1.25·1.7²=3.6125, still unbiased but noisier than with the matching anchor. An arbitrarily stale anchor can even make variance larger than ordinary SGD variance; variance reduction is not a pointwise guarantee for every possible pair.

Now choose x=x̃=2. SVRG variance is zero, but every estimate equals 3.25. The direction is exact yet nonzero: zero variance is not an optimality certificate.

## Account for work and outer loops

The diagram compares four possible component estimates at one pair of points. It is not a full SVRG training run. Computing G(x̃) costs four component gradients here. An inner update generally evaluates gradients at both x and the anchor unless anchor information is cached. The full-gradient cost must be amortized across several updates, with a specified anchor-refresh schedule.

Convergence rates require assumptions such as smooth component losses and appropriate convexity or nonconvex stationarity targets. A lower one-step variance alone does not specify the step size, total work or final accuracy.

## From a control variate to an epoch contraction bound

Assume every component is convex and L-smooth, and the average F is μ-strongly convex on Euclidean space. Let x* minimize F. The smooth-convex gradient inequality, averaged over components, bounds E‖gᵢ(x)−gᵢ(x*)‖² by 2L(F(x)−F*); the linear terms cancel because ∇F(x*)=0. Splitting the SVRG estimator into current and centered anchor differences gives

$$
\mathbb E\|v_i\|^2\le4L\big(F(x)-F^*+F(\widetilde x)-F^*\big).
$$

This is a second-moment bound, not a claim that every component gradient vanishes at the optimizer. Expand one update's squared distance, use convexity, sum m inner steps, and choose the next anchor as their average. For η<1/(2L), the resulting expected objective contraction factor is

$$
\rho=\frac1{\mu\eta(1-2L\eta)m}+\frac{2L\eta}{1-2L\eta}.
$$

Only ρ<1 guarantees contraction by this argument. In the lab's quadratic family L=4,μ=5/2,η=1/32,m=64 gives ρ=3/5. A stale anchor alone is not a full algorithm; refresh and averaging rules matter.

| Inner steps m, with these constants | Bound ρ |
|---:|---:|
| 16 | 7/5 |
| 32 | 13/15 |
| 64 | 3/5 |

The Python extension runs actual sampled epochs with explicit component-call counts; one path need not satisfy an expectation bound at every epoch. The controls remain a one-estimator comparison. Their variance now uses the centered analytic identity, retaining 1.25×10⁻⁴⁰ for x=0,anchor=10⁻²⁰ even when all displayed estimator values round alike.

## Reproduce the calculation in Python

```python
from fractions import Fraction as Q
from random import Random
L,mu,eta=Q(4),Q(5,2),Q(1,32); m=64
for length in [16,32,64]:
    rho=1/(mu*eta*(1-2*L*eta)*length)+2*L*eta/(1-2*L*eta)
    print('m:',length,'rho:',rho)
h=[1,2,3,4]; a=[-1,1,2,0]; calls=0; rng=Random(7)
def gradient(i,x):
    global calls
    calls+=1
    return h[i]*(x-a[i])
anchor=Q(-1)
for epoch in range(4):
    before=calls; full=sum(gradient(i,anchor) for i in range(4))/4
    x=anchor; iterates=[]
    for _ in range(m):
        iterates.append(x); i=rng.randrange(4)
        x-=eta*(gradient(i,x)-gradient(i,anchor)+full)
    anchor=sum(iterates)/m
    assert calls-before==4+2*m
    print('epoch:',epoch+1,'gap:',float(Q(5,4)*(anchor-Q(7,10))**2),'calls:',calls-before)
```

## Exercise and explained answer

At x=0.7,x̃=−1 and selected index zero, g₁(x)=1.7. The SVRG estimate is 1.7−g₁(−1)+G(−1)=−2.55. With η=0.1 the SGD next point is 0.53 while the SVRG next point is 0.955. Explain why either individual step can move away from the optimizer even though both estimators have mean zero there. Verify all four estimates average to G(x).

Further reading: [Johnson and Zhang, Predictive Variance Reduction](https://papers.nips.cc/paper/2013/hash/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Abstract.html).
