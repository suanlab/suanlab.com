## Pull an output event back

To find the law of Y=g(X), ask which input outcomes satisfy g(X)≤y. This event-based method is more fundamental than an inverse-function formula and still works for many-to-one or constant maps. The lab compares Y=aX and Y=aX² with integer scale a. X is uniform on [0,1] in linear mode and [−1,1] in square mode. The plot shows the output CDF.

$$
F_Y(y)=P(g(X)\le y).
$$

## Why the Jacobian is absolute

For a differentiable one-to-one transform, preserving mass in a small interval introduces the absolute derivative of the inverse. If X is uniform on [0,1] and Y=2X, the support doubles in width and density halves. Negative a reverses orientation but cannot make density negative. In several dimensions, the corresponding factor is the absolute determinant of the inverse Jacobian.

$$
f_Y(y)=f_X(g^{-1}(y))\left|\frac{d}{dy}g^{-1}(y)\right|.
$$

## Squaring has two preimages

For Y=X² and 0<y<1, both positive and negative square roots map to y. Each branch has input density one half and inverse derivative magnitude 1/(2√y). Adding them gives 1/(2√y). At y=0.25, density is one and the CDF is one half. Keeping only one inverse branch loses half the total mass.

$$
f_Y(y)=\sum_{x:g(x)=y}\frac{f_X(x)}{|g^{\prime}(x)|}=\frac1{2\sqrt y},\quad F_Y(y)=\sqrt y\quad(0<y<1).
$$

## Boundaries and constant transforms

The square-mode density diverges near zero but remains integrable. At the boundary the interface gives no finite density readout and flags the unbounded limit. Choosing a density value at one point does not change probabilities. At a=0, the output is always zero: point mass one and no Lebesgue density. Dividing by a zero Jacobian is invalid. Density calculations in invertible generative models rely on these assumptions; the formula does not apply unchanged to every neural network.

$$
a=0\implies P(Y=0)=1,\qquad a\ne0:\;f_{aX}(y)=\frac1{|a|}f_X(y/a).
$$

## ReLU turns continuous input into a mixed law

Let X be uniform on [−1,1] and set Y=max(0,X). Every negative input maps to zero, an interval of input probability one half. Positive inputs retain their positions. Pulling back {Y≤y} gives a CDF with both a jump and a sloping part:

$$
F_Y(y)=\begin{cases}0&y<0,\\(1+y)/2&0\le y<1,\\1&y\ge1.\end{cases}
$$

| y | CDF at y | Point mass at y |
|---|---|---|
| −1/2 | 0 | 0 |
| 0 | 1/2 | 1/2 |
| 1/4 | 5/8 | 0 |
| 1 | 1 | 0 |

The law is one half of a unit atom at zero plus one half of Uniform(0,1). Its continuous part has density 1/2 on (0,1), integrating to only 1/2; the missing half is the atom. The Dirac measure δ0 is a unit point mass, not an ordinary Lebesgue density. In measure notation,

$$
P_Y(dy)=\tfrac12\delta_0(dy)+\tfrac12\mathbf1_{(0,1)}(y)\,dy.
$$

For example, P(Y≤1/4)=5/8, while P(0<Y≤1/4)=1/8. Differentiating the sloping part alone loses the jump. The one-to-one Jacobian formula cannot account for a whole interval collapsed onto a point. Direct integration gives mean 1/4, second moment 1/6 and variance 5/48.

## A two-dimensional Jacobian also needs the transformed support

Take independent Uniform(0,1) inputs X1,X2 and transform them to U=X1 and V=exp(X1)X2. The inverse and its determinant are explicit:

$$
(x_1,x_2)=(u,ve^{-u}),\qquad
Dg^{-1}(u,v)=\begin{pmatrix}1&0\\-ve^{-u}&e^{-u}\end{pmatrix}.
$$

The support is curved in these coordinates: 0<u<1 and 0<v<exp(u). Multiplying the original unit density by the inverse determinant gives

$$
f_{U,V}(u,v)=e^{-u}\mathbf1_{\{0<u<1,\;0<v<e^u\}}.
$$

At u=log 2, the conditional width in v is two. The joint density at v=1 is 1/2, but at v=3 it is zero. Integrating over v gives e⁻ᵘeᵘ=1; integrating over 0<u<1 then gives total mass one. Keeping the old rectangle 0<v<1 would instead retain only 1−e⁻¹ of the mass. A correct log determinant with the wrong support still defines the wrong law. In invertible generative models, verify both inverse mapping and support rather than treating the determinant as a complete density specification.

## Run the mixed and invertible examples separately

The code uses exact fractions for the mixed law and floating-point exponentials for the explicit triangular map. It checks interval masses, inverse Jacobians and the conditional-width normalization. These extensions are separate from this page's existing scaled-linear and squared-variable CDF controls; they do not add a ReLU distribution or a general flow engine to those controls.

```python
from fractions import Fraction as Q
from math import exp, log, isclose

def cdf(y):
    return Q(0) if y < 0 else (1 + min(y, Q(1))) / 2

def interval_mass(a, b):
    atom = Q(1, 2) if a < 0 <= b else Q(0)
    length = max(Q(0), min(b, Q(1)) - max(a, Q(0)))
    return atom + length / 2

for y in [Q(-1, 2), Q(0), Q(1, 4), Q(1)]:
    assert cdf(y) == interval_mass(Q(-2), y)
    print("y, CDF, point mass:", y, cdf(y), Q(1, 2) if y == 0 else 0)
assert cdf(Q(1, 4)) == Q(5, 8)
assert interval_mass(Q(0), Q(1, 4)) == Q(1, 8)
mean, second = Q(1, 4), Q(1, 6)
assert second - mean**2 == Q(5, 48)

def density(u, v):
    return exp(-u) if 0 < u < 1 and 0 < v < exp(u) else 0.0

for x1 in [.25, .5, .75]:
    for x2 in [.25, .75]:
        u, v = x1, exp(x1)*x2
        assert isclose(v*exp(-u), x2, abs_tol=1e-14)
        forward = ((1.0, 0.0), (v, exp(u)))
        inverse = ((1.0, 0.0), (-v*exp(-u), exp(-u)))
        for i in range(2):
            for j in range(2):
                value = sum(inverse[i][k]*forward[k][j] for k in range(2))
                assert isclose(value, float(i == j), abs_tol=1e-14)
        assert isclose(density(u, v)*exp(u), 1.0, abs_tol=1e-14)
u = log(2)
assert isclose(density(u, 1), .5, abs_tol=1e-14)
assert density(u, 3) == 0.0
print("mixed-law mean and variance:", mean, second-mean**2)
print("triangular density:", density(u, 1), density(u, 3))
```

## Further questions and research use

1. Why is there no atom at Y=1? **Its preimage is the single endpoint X=1, which has input probability zero.**
2. Can a noninvertible layer always use the usual flow likelihood formula? **No: collapsed regions, multiple branches and the reference measure must be accounted for.**
3. Why can the joint density at u=log 2 be 1/2 while U has density one? **Integrating over its v-support of width two restores the marginal density.**

For the event-first method and inverse-branch assumptions, see [Pishro-Nik, §4.1.3](https://www.probabilitycourse.com/chapter4/4_1_3_functions_continuous_var.php). The ReLU and triangular-map examples above are derived directly. Compare the distinction between mass and density in [the CDF unit](../mass-density-cdf/).

## Exercises with explanations

1. What is the linear density for a=−2? **One half inside (−2,0).**
2. Does squaring have one inverse branch? **No: two at positive interior values.**
3. Does an unbounded density imply an atom at zero? **No: nonzero-scale point probability remains zero.**

Further reading: [Pishro-Nik, Introduction to Probability, Statistics, and Random Processes](https://www.probabilitycourse.com/).
