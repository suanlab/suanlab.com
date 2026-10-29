## Angles become coordinates

Trigonometry connects a direction with coordinate components. On the unit circle, cosθ is the horizontal coordinate and sinθ the vertical coordinate. Scaling radius to r gives p=(r cosθ,r sinθ). The construction also explains why sine and cosine repeat and why an angle cannot be recovered from the origin.

The lab uses dimensionless coordinates and a fixed comparison point q=(1,0). Both plot axes have equal unit length so the circle remains a circle and displayed directions are geometrically meaningful.

## Degrees and radians are different numerical inputs

A full turn is 360 degrees or 2π radians. Radians measure arc length divided by radius, a dimensionless ratio with an angle convention. Therefore

$$
\theta_{\rm rad}=\theta_{\rm deg}\frac{\pi}{180}.
$$

The controls accept degrees and convert before calling numerical trigonometric functions. Supplying 60 directly to a function expecting radians would not calculate the sine of 60 degrees.

Calculus formulas such as derivative of sinθ being cosθ use a radian input. If the variable is a degree count, the chain rule introduces π/180. This is a concrete example of why units and parameterization matter to a derivative.

## Work the reference triangle

At θ=60°=π/3 and r=2, cosθ=1/2 and sinθ=√3/2. Thus p=(1,√3). Its distance from q=(1,0) is √3 because the horizontal difference is zero.

For general points p,q in orthogonal Euclidean coordinates,

$$
\lVert p-q\rVert_2=\sqrt{(p_x-q_x)^2+(p_y-q_y)^2}.
$$

This assumes compatible coordinate units and a Euclidean metric. A longitude-latitude pair or a learned feature space does not automatically have physical distances given by this formula without additional modeling.

## Identities and periodicity

The unit circle gives cos²θ+sin²θ=1. Angle addition follows from composing planar rotations:

$$
\cos(\alpha+\beta)=\cos\alpha\cos\beta-\sin\alpha\sin\beta,
$$

with the corresponding sine sum using plus. Adding 2π preserves the point, so θ=−300° and 60° give the same coordinates. The representation of angle is not unique even when the nonzero point is fixed.

Tangent is sinθ/cosθ only when the denominator is nonzero. A vertical direction is not assigned an arbitrary finite tangent. This repeats the domain discipline used for rational functions.

## Recovering an angle needs conventions

For a nonzero point, atan2 uses both coordinate signs to select a quadrant. Using arctan(y/x) alone loses quadrant information and fails when x=0. Even atan2 selects a principal branch, so its reported angle can jump by a full turn near the branch cut.

At r=0, all θ yield p=(0,0). The inverse direction is undefined. The input control still contains an angle, but the resulting point does not encode it. The lab therefore reports no recovered argument rather than pretending the input can be inferred.

## Read the experiment and connect to models

Blue is the unit reference circle, orange the position vector p. Radius changes the point, not the definition of cosine and sine. The reported distance is to q=(1,0), not to the origin. The origin distance would simply be r.

Rotations, phases and periodic features appear in signal processing, robotics and positional representations. Encoding a phase by sine/cosine avoids a scalar angle jump at a wrap boundary, but adds an algebraic constraint cos²+sin²=1. Predicting two unconstrained values does not automatically keep them on the unit circle.

For a research study, state coordinates, metric and angle convention. Test wrapped angles, zero-length vectors and quadrant boundaries, rather than checking only acute angles.

## Exercises and reference

1. Find p at r=2,θ=90°. **Answer:** (0,2) in exact arithmetic.
2. Convert −180° to radians. **Answer:** −π.
3. Explain why the distance to q differs from the radius in general.
4. Compare θ=179° and −179° as scalar numbers and as unit-circle points. Why can a naive scalar squared error misrepresent proximity?

See [OpenStax Algebra and Trigonometry 2e](https://openstax.org/details/books/algebra-and-trigonometry-2e) for trigonometry and coordinates. The equal-scale lab and stated metric are part of this explicit construction.
