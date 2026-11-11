## Energy as a negative-log probability form

Take three spins sᵢ in {−1,1} connected in a triangle. Coupling J and common field h define energy; temperature T>0 turns it into exp(−E/T) weights. This dimensionless example uses the convention kB=1.

$$
E(s)=-J(s_As_B+s_Bs_C+s_Cs_A)-h(s_A+s_B+s_C),\quad
p(s)=e^{-E(s)/T}/Z.
$$

## Sum all states for the partition function

Enumerate eight states to compute Z=Σs exp(−E(s)/T). At J=h=0 every energy is zero, Z=8 and each probability is 1/8. Z is a weight sum rather than a probability and may exceed one.

Numerically subtract maximum log weight M, then restore it using logZ=M+logΣexp(logw−M). Summing displayed relative weights alone does not recover the original Z.

## Low temperature and symmetry

Positive J favors aligned neighbors. At h=0, flipping every spin preserves energy, so mean total spin is exactly zero. Low temperature concentrates on ground states such as +++ and ---. Zero mean spin does not imply a uniform joint distribution.

## Energy expectations and entropy

Average energy and total spin under normalized probabilities. Averaging log p=−E/T−logZ gives entropy E[E]/T+logZ in nats. Parameter derivatives must retain the temperature factor.

$$
H(p)=-\sum_s p(s)\log p(s)=E[E]/T+\log Z,\quad
\frac{\partial\log Z}{\partial h}=E[\sum_i s_i]/T.
$$

## Connections to larger models

Computing Z can require exponentially many states. Energy-based learning must consider normalization changes as well as observed-state scores. Exact enumeration available for eight states is not automatically affordable for large neural energy models. Sampling and variational approximations bring separate accuracy and mixing conditions.

## Curvature of the normalizer measures fluctuations

With finitely many states, finite J,h,T and T>0, all weights are positive and differentiation commutes with the sum. Write M=Σsᵢ and β=1/T, assuming E itself does not depend on β. Differentiating log Z first produces normalized weights; differentiating again subtracts the product of means, giving covariance:

$$
\partial_h\log Z=\frac{E[M]}T,\qquad
\partial_h^2\log Z=\frac{\operatorname{Var}(M)}{T^2}\ge0,
\qquad
\partial_\beta\log Z=-E[E],\quad
\partial_\beta^2\log Z=\operatorname{Var}(E).
$$

At fixed J,h, this also yields ∂T E[E]=Var(E)/T², the heat capacity in this dimensionless convention. Mixed parameter derivatives give covariances of the corresponding sufficient statistics with the appropriate temperature factors. Curvature tells us which state fluctuations a parameter probes.

## Verify the identities in the triangle

At J=1,h=0, the two aligned states have energy −3 and the other six have energy one. Thus $Z=2e^{3/T}+6e^{-1/T}$, and the aligned mass is $g=1/(1+3e^{-4/T})$. The energy gap is four, so Var(E)=16g(1−g). Spin-flip symmetry makes E[M]=0 even when Var(M) is large.

| T | Aligned mass g | Heat capacity Var(E)/T² | Field curvature Var(M)/T² |
|---|---|---|---|
| .5 | .998995 | .064279 | 35.967828 |
| 1 | .947915 | .789955 | 8.583320 |
| 2 | .711235 | .821520 | 1.672469 |

The code compares eight-state sums, the two-level closed form and centered finite differences. Finite differences are numerical checks with tolerances, rather than the proof of the derivative identities.

## Two averages in energy-based learning

Fix T,J and learn only h. If the data's mean total spin is M̄, the per-sample log-likelihood gradient is (M̄−E_model[M])/T. Raising the observed state's field score alone omits the model average contributed by log Z. Negative log-likelihood curvature is Var_model(M)/T²≥0, making this one-parameter problem convex. A nonlinear neural energy parameterization need not retain this convexity.

For eight finite states and T>0, log Z is smooth. A sharp change in the table does not establish a nonanalytic phase transition in a thermodynamic limit.

**Exercise:** Does zero mean spin at h=0 imply zero field curvature? **Explanation:** No. Concentration near ±3 can give zero mean and large variance and curvature.

```python
from math import exp, log, cosh
from itertools import product

def moments(T,h=0):
    states = list(product([-1,1],repeat=3))
    energy = [-sum(s[i]*s[(i+1)%3] for i in range(3))-h*sum(s) for s in states]
    weights = [exp(-e/T) for e in energy]
    Z = sum(weights)
    p = [w/Z for w in weights]
    mean_e = sum(q*e for q,e in zip(p,energy))
    var_e = sum(q*(e-mean_e)**2 for q,e in zip(p,energy))
    mean_m = sum(q*sum(s) for q,s in zip(p,states))
    var_m = sum(q*(sum(s)-mean_m)**2 for q,s in zip(p,states))
    return log(Z),mean_e,var_e,mean_m,var_m

for T in [.5,1,2]:
    logZ,mean_e,var_e,mean_m,var_m = moments(T)
    aligned = 1/(1+3*exp(-4/T))
    assert abs(logZ-log(2*exp(3/T)+6*exp(-1/T))) < 1e-12
    assert abs(var_e-16*aligned*(1-aligned)) < 1e-12
    eps = 1e-4
    plus,minus = moments(T,eps),moments(T,-eps)
    curvature = (plus[0]-2*logZ+minus[0])/eps**2
    assert abs(curvature-var_m/T**2) < 1e-5
    temperature_slope = (moments(T+eps)[1]-moments(T-eps)[1])/(2*eps)
    assert abs(temperature_slope-var_e/T**2) < 1e-6
    print('T, aligned mass, heat capacity, field curvature:',
          *(round(v,6) for v in [T,aligned,var_e/T**2,var_m/T**2]))
```


## Exercises with explanations

1. How is the removed maximum restored in logZ? **Add it back.**
2. What is mean total spin at h=0? **Zero by flip symmetry.**
3. Does low temperature always select one unique state? **No: ground states can be multiple.**

Further reading: [Stanford CS228, Probabilistic Graphical Models](https://ermongroup.github.io/cs228-notes/representation/directed/).
