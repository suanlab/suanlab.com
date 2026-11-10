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

## Exercise and explained answer

Use quadratic mode with x=3,y=2,z=1. D(3,1)=2 but D(3,2)+D(2,1)=1, directly violating the triangle inequality. Then use entropy mode and swap x=2,y=1 to verify the unequal gaps above. Finally set x=y: both divergences vanish, while the three-point identity remains valid for any displayed z.

Mirror descent uses such generator geometry in an update rule. Choosing a generator changes the notion of movement cost and often preserves a useful domain; it does not change the objective into the generator itself.

Further reading: [Nemirovski, Mirror Descent Tutorial](https://www2.isye.gatech.edu/~nemirovs/COLT2012Tut.pdf).
