## Ask three different robustness questions

A single word, robustness, can hide several mathematical questions. How strongly does an infinitesimal amount of contamination affect an estimator? How far does a fixed nonzero amount move its population target? How much replacement is needed to make a finite-sample estimate diverge? These are local influence, finite contamination bias and replacement breakdown. They are related, but none can simply replace the others.

This unit uses an explicitly stated population model. It complements the previous deterministic replacement experiment rather than pretending the two experiments generate the same observations. The clean distribution is Uniform[−1,1], its location is zero, and its scale is known.

## Define a population mixture

Put a point mass at a contaminating position z and mix it with the clean distribution:

$$
F_\epsilon=(1-\epsilon)U[-1,1]+\epsilon\delta_z.
$$

The controls restrict epsilon to at most0.49, so the original distribution retains population weight greater than one half. This does not guarantee that a particular finite iid sample contains a clean majority: a population mixing probability is not the realized fraction of corrupted records.

For a location functional T, define contamination displacement as T(F_epsilon)−T(F_0). Since every clean location here is zero, the displayed contaminated locations equal their biases relative to the clean target. This is not sampling error around a contaminated target and is not a confidence interval.

## Calculate the mean and median exactly

The mixture mean is epsilon z. A tiny positive fraction can therefore produce arbitrarily large bias as z increases without bound. The bounded slider illustrates this sensitivity but cannot prove the limiting claim by itself.

For the median, impose both median CDF inequalities, including the jump at z. The result for epsilon below one half is

$$
T_{\rm med}(F_\epsilon)=\operatorname{clip}\left(z,-\frac{\epsilon}{1-\epsilon},\frac{\epsilon}{1-\epsilon}\right).
$$

If z lies within these bounds, the jump straddles probability one half and the median equals z. Outside the bounds the median lies in the continuous clean part. Ignoring the jump or treating every median as a smooth inverse CDF would miss the central case.

## Define influence as a derivative at zero

The influence function at a reference distribution is

$$
IF(z;T,F)=\lim_{\epsilon\downarrow0}\frac{T((1-\epsilon)F+\epsilon\delta_z)-T(F)}{\epsilon},
$$

when that derivative exists. Under the present reference, the mean influence is z and the median influence is sign(z), with zero at z=0. The median formula uses the positive clean density at its median: in a general continuous reference it is sign(z−m)/(2f(m)). At an atom or a vanishing density the usual differentiability assumptions need separate examination.

A bounded influence function limits this first derivative over contaminating positions. It does not say finite contamination has zero effect, that all finite samples are stable, or that the same estimator solves a different contamination problem.

## Derive the Huber normalization

For known scale1, the Huber population location solves the expectation of its clipped residual score equal to zero. Differentiate the score equation along the contamination path at the clean location. The derivative of the location appears multiplied by the expectation of the score derivative, giving

$$
IF_H(z)=\frac{\operatorname{clip}(z,-c,c)}{\mathbb P_U(|X|<c)}
=\frac{\operatorname{clip}(z,-c,c)}{\min(c,1)}.
$$

The denominator matters. With c=0.5 a very distant point has clipped score0.5 but influence1, not0.5. Dividing by the reference sensitivity of the equation converts a score perturbation into a location perturbation. Merely plotting the clipped score and naming it influence would give the wrong scale.

## Integrate the finite Huber equation

Let rho_c be the Huber loss from the previous unit. Because its derivative is the clipped score, integrate over the clean uniform distribution to obtain

$$
S_U(\theta)=\frac{\rho_c(1-\theta)-\rho_c(1+\theta)}{2}.
$$

The contaminated score is

$$
(1-\epsilon)S_U(\theta)+\epsilon\operatorname{clip}(z-\theta,-c,c)=0.
$$

The code evaluates this analytic integral and solves its monotone equation numerically. It does not approximate the clean distribution by a random sample. With the clean weight greater than one half, the root is unique within[−1−c,1+c]; at either exterior side, the clean clipped contribution dominates any contamination contribution. Inside that bracket, the clean uniform score is strictly decreasing wherever a root can occur.

## Work the small-contamination guide

Set epsilon=0.01,z=100,c=0.5. The mean bias is1. The median bias is1/99. The Huber solution also equals1/99 because its root remains in the central region where the clean score is−c theta and the contaminating score is c.

Both median and Huber have influence1 at this z, so their first-order bias approximation is0.01. That is close to1/99, but it is not exactly the same. Taking epsilon toward zero makes the difference lower order than epsilon; it does not erase it for an arbitrary finite epsilon.

## Increase contamination and change the threshold

At epsilon=0.25 with the same z and c, median and Huber biases are1/3 while their first-order approximations are1/4. The mean bias is25. Bounded influence has reduced sensitivity, yet a nonzero finite bias remains.

Now set c=2. The Huber influence at z=100 is2 and its first-order bias approximation is1/2. The exact finite bias is2/3: all clean residuals remain inside the quadratic region, giving−(1−epsilon)theta+epsilon c=0. Verify this residual-region condition rather than using the simplified equation for every possible c and z.

## Derive reference-model efficiency coefficients

Under suitable regularity, the influence variance is the coefficient of an estimator's asymptotic variance divided by sample size. For this particular uniform reference, direct integration gives

$$
\operatorname{Var}(IF_{\rm mean})=1/3,\qquad
\operatorname{Var}(IF_{\rm med})=1,
$$

and

$$
\operatorname{Var}(IF_H)=\begin{cases}1-2c/3,&0<c\le1,\\1/3,&c\ge1.\end{cases}
$$

For c below1, integrate the squared clipped score inside and outside[−c,c], then divide by c squared. For c at least1, the score is linear across the clean support. These are reference-specific coefficients, not a universal ranking of methods. In particular, a threshold chosen for Gaussian efficiency should not be imported as a universal constant.

## Relate local influence to global breakdown carefully

The previous finite sample had mean breakdown1/21 and median breakdown11/21. A bounded local influence at a smooth reference does not itself prove either finite-sample breakdown value. Conversely, high breakdown does not guarantee a small local variance or small contamination bias at every distribution.

Known scale is another substantive condition. Estimating scale from corrupted data changes the functional and its influence. In regression, a point may have extreme covariates as well as a large residual; bounding only a residual score does not automatically control leverage. These limitations identify the next mathematical questions instead of invalidating the present calculation.

## Research extension: Turn score conditioning into a finite bias and solver bound

An influence derivative is local. To control a finite shift, inspect how strongly the clean score changes over the whole relevant interval. Let $S_P(0)=0$ and assume $-S'_P(\theta)\ge\kappa>0$ for $|\theta|\le r$, with a globally nonincreasing clean score. The contaminating score $S_Q(\theta)=\mathbb E_Q\operatorname{clip}(X-\theta,-c,c)$ is also nonincreasing and has absolute magnitude at most $c$. If

$$
(1-\varepsilon)\kappa r>\varepsilon c,
$$

the mixture score has opposite signs at the endpoints. It is strictly decreasing inside, so its unique root lies inside the interval. At that root the contamination can balance at most $\varepsilon c$ of force. The clean score's slope bound then proves

$$
|\theta_*|\le\frac{\varepsilon c}{(1-\varepsilon)\kappa}.
$$

The denominator is a finite-region conditioning assumption, not merely the score derivative at the clean root. A bounded score alone does not control bias when this denominator becomes small. Likewise, for an approximate root $\widetilde\theta$ in the same interval with mixture score residual at most $e_s$,

$$
|\widetilde\theta-\theta_*|\le\frac{e_s}{(1-\varepsilon)\kappa}.
$$

A small residual becomes an informative location-error bound only after establishing the slope and interval conditions.

For the current uniform reference, take $c=r=1/2$. On this interval $S_P(\theta)=-\theta/2$, hence $\kappa=1/2$. With distant positive contamination $z=10$ and $\varepsilon<1/3$, the bound is attained by $\theta_* =\varepsilon/(1-\varepsilon)$. Its difference from the first-order approximation is exactly $\varepsilon^2/(1-\varepsilon)$:

| Contamination | Exact bias | First-order bias | Remainder |
|---|---:|---:|---:|
| 1/100 | 1/99 | 1/100 | 1/9900 |
| 1/10 | 1/9 | 1/10 | 1/90 |
| 1/4 | 1/3 | 1/4 | 1/12 |

The code evaluates the uniform score by its integrated Huber loss, verifies the root and checks the residual-to-error relation. The theorem addresses bias and numerical conditioning under stated assumptions, not a confidence interval or a new finite-sample breakdown guarantee.

```python
from fractions import Fraction as F
c=r=kappa=F(1,2); z=F(10)
def clip(x):return max(-c,min(c,x))
def loss(x):return x*x/2 if abs(x)<=c else c*(abs(x)-c/2)
def clean(theta):return (loss(1-theta)-loss(1+theta))/2
for epsilon in [F(1,100),F(1,10),F(1,4)]:
    assert (1-epsilon)*kappa*r>epsilon*c
    theta=epsilon/(1-epsilon)
    score=lambda t:(1-epsilon)*clean(t)+epsilon*clip(z-t)
    assert abs(theta)<r and clean(theta)==-kappa*theta and score(theta)==0
    bound=epsilon*c/((1-epsilon)*kappa)
    assert theta==bound
    trial=theta+F(1,100)
    assert abs(trial)<r
    assert abs(trial-theta)==abs(score(trial))/((1-epsilon)*kappa)
    remainder=theta-epsilon
    assert remainder==epsilon**2/(1-epsilon)
    print(epsilon,theta,epsilon,remainder)
```

## Exercises and explained solutions

1. At epsilon=0.25 and z=100, why is the median bias not0.25? Solution: the clean distribution has weight0.75, so moving its CDF by0.25 requires displacement0.25/0.75=1/3.
2. At the same epsilon with z=0.1, what is the median? Solution:0.1 lies inside the clipping interval, so the contamination jump pins the median there.
3. With c=0.25 and a distant z, what is Huber influence? Solution:0.25/0.25=1, not0.25.
4. Does equal clean asymptotic variance imply equal contaminated bias? Solution: no. At c=2 the Huber coefficient equals the mean's1/3, but their distant-contamination biases differ.
5. Is the graph an optimization trajectory? Solution: no. Its horizontal coordinate is mixture probability; z and c are held fixed while population functionals are recomputed.
6. Can epsilon=0.49 certify a clean majority in21 sampled records? Solution: no. Population weight and a random realized count are distinct.

## Connect the elementary model to research

[Chen, Gao and Ren](https://arxiv.org/html/1511.04144v2) explicitly formulate robust inference under population contamination. General results specify parameter classes, losses and sampling assumptions. Our uniform location family provides direct derivations and interpretable counterexamples, not their full minimax theory. The next original topic studies adversarial perturbation, which changes the mathematical threat model again.
