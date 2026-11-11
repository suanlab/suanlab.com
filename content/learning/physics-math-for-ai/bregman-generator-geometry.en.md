## Measure a convex function's gap above its tangent

For a differentiable convex generator φ, define

$$
D_\phi(x,y)=\phi(x)-\phi(y)-\phi'(y)(x-y).
$$

Convexity makes this nonnegative. Strict convexity makes it zero only at equal arguments. The second argument y fixes the tangent's base point; the first x is where the vertical gap is evaluated. Blue plots φ and orange its tangent, so the two marked heights at x differ by Dφ(x,y).

A divergence is not automatically a metric distance. It can be asymmetric and can fail the triangle inequality. Even half squared Euclidean distance is not itself a distance satisfying that inequality.

## Compare three generators

The lab restricts positive inputs to [0.1,3]. Its generators give

$$
\begin{aligned}
\phi(x)=x^2/2 &: D=(x-y)^2/2,\\
\phi(x)=x\log x-x &: D=x\log(x/y)-x+y,\\
\phi(x)=-\log x &: D=x/y-1-\log(x/y).
\end{aligned}
$$

The entropy divergence becomes KL when summed over normalized probability vectors, since the linear mass terms cancel. A scalar positive argument alone is not a normalized distribution. The log generator emphasizes relative ratios rather than absolute differences.

At x=2,y=1, entropy forward divergence is 2log2−1≈0.386294 and reverse is 1−log2≈0.306853. The tangent base changes when arguments are swapped. Quadratic divergence instead gives 0.5 in both directions.

## Local curvature and numerical cancellation

For a small displacement h, Dφ(y+h,y)=φ″(y)h²/2+o(h²). Curvature is one for the quadratic, 1/y for entropy and 1/y² for the log generator. The same absolute displacement therefore has different local cost at different positive base points.

Directly subtracting φ values and their tangent can erase a tiny positive gap through cancellation. The implementation uses stable logarithmic Taylor remainders near equality. It does not hide a negative result by clipping it to zero or confuse a numerical zero with a proof that unequal arguments have zero divergence.

## A useful three-point identity

Expanding the definitions gives

$$
D(x,z)-D(x,y)-D(y,z)
=(\phi'(y)-\phi'(z))(x-y).
$$

The lab computes both sides independently. This gradient-difference term explains why Euclidean-looking triangle or Pythagorean reasoning cannot simply be reused. Additional orthogonality or optimality conditions are needed for particular projection inequalities.

## Recover a Pythagorean inequality with the correct orientation

Suppose a minimizer z of Dφ(v,x) over a convex set C exists in the differentiability domain. First-order optimality gives ⟨∇φ(z)−∇φ(x),u−z⟩≥0 for every u∈C. Insert this into the three-point identity:

$$
D_\phi(u,x)-D_\phi(u,z)-D_\phi(z,x)
=\langle\nabla\phi(z)-\nabla\phi(x),u-z\rangle\ge0.
$$

This is a projection inequality, with the projected variable in the first argument. It is not a triangle inequality for arbitrary points. Convexity of C and the appropriate minimization orientation do real work; reversing KL or using a nonconvex set does not preserve this proof.

For positive vectors use φ(v)=Σ(vᵢlog vᵢ−vᵢ), and project x onto the probability simplex. The solution is z=x/S, S=Σxᵢ. The gradient difference is the constant vector −log S, whose pairing with u−z vanishes because both have mass one. Thus equality holds for every interior simplex u. The unnormalized divergence D(z,x)=S−1−log S includes mass terms; dropping them would be incorrect.

| x | z | D(z,x) |
|---|---|---|
| (2,1) | (2/3,1/3) | 2−log3 |
| (1,2) | (1/3,2/3) | 2−log3 |
| (3,1) | (3/4,1/4) | 3−log4 |

```python
from decimal import Decimal as D, localcontext
with localcontext() as ctx:
    ctx.prec = 80
    def divergence(u, v):
        return sum(a*(a/b).ln()-a+b for a, b in zip(u, v))
    for x in ((D(2), D(1)), (D(1), D(2)), (D(3), D(1))):
        mass = sum(x)
        z = tuple(a/mass for a in x)
        for first in (D(1)/4, D(1)/2, D(3)/4):
            u = (first, 1-first)
            residual = divergence(u, x)-divergence(u, z)-divergence(z, x)
            assert abs(residual) < D('1e-75')
        assert abs(divergence(z, x)-(mass-1-mass.ln())) < D('1e-75')
        print(tuple(x), tuple(z), divergence(z, x))
```

This vector projection extends the scalar tangent-gap illustration without changing its controls. Decimal arithmetic checks nine instances of the equality. It is a numerical check of a separately proved identity, not a proof from a near-zero residual.

## Exercise and explained answer

Use quadratic mode with x=3,y=2,z=1. D(3,1)=2 but D(3,2)+D(2,1)=1, directly violating the triangle inequality. Then use entropy mode and swap x=2,y=1 to verify the unequal gaps above. Finally set x=y: both divergences vanish, while the three-point identity remains valid for any displayed z.

Mirror descent uses such generator geometry in an update rule. Choosing a generator changes the notion of movement cost and often preserves a useful domain; it does not change the objective into the generator itself.

Further reading: [Nemirovski, Mirror Descent Tutorial](https://www2.isye.gatech.edu/~nemirovs/COLT2012Tut.pdf).
