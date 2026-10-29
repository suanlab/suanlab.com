## Control intermediate values

On a convex domain, convexity means

$$
f((1-t)x+ty)\le(1-t)f(x)+tf(y),\quad0\le t\le1.
$$

This quantifies over all pairs and interpolation weights. The domain must contain the connecting segments. A bowl-like picture is intuition, not the definition.

## First-order supporting inequality

For differentiable convex f,

$$
f(y)\ge f(x)+\nabla f(x)^T(y-x).
$$

Apply convexity at x+t(y−x), subtract f(x), divide by t and let t decrease to zero. Conversely, the supporting inequality for all pairs implies convexity. A single successful pair does not.

## An exact quadratic gap

The lab uses f(x)=ax², giving

$$
f(y)-f(x)-f'(x)(y-x)=a(y-x)^2.
$$

For a≥0 every gap is nonnegative. For a<0 distinct points give a negative gap. Yet x=y gives zero even for a concave function. This exposes the difference between checking one probe and certifying a global property.

## Curvature and strong convexity

For sufficiently smooth functions on an open convex domain, PSD Hessians everywhere characterize convexity. The scalar example has f″=2a. For a>0, m=2a gives

$$
f(y)\ge f(x)+f'(x)(y-x)+\frac m2|y-x|^2.
$$

A positive uniform curvature bound differs from merely positive curvature at selected points. The strictly convex function x⁴ is not globally strongly convex with positive m, because its curvature vanishes at zero. Keep convexity, strict convexity and strong convexity separate.

## What optimization conclusions follow?

A zero gradient at a point of a differentiable convex function certifies a global minimum there. It does not establish existence or uniqueness in general: exp(x) on R has no attained minimum, while a constant function has every point as a minimizer.

For neural losses, identify whether convex reasoning concerns a local approximation, one parameter block or a surrogate objective. One PSD Hessian measurement does not establish global convexity.

## Read the experiment and exercises

Blue is the function and orange the tangent at x. The y control selects one comparison point. The universal indicator uses the analytic condition a≥0, not the sampled gap.

1. At a=0 the function is **convex but not strongly convex**.
2. At a=−1,x=y=0, the gap is **zero despite nonconvexity**.
3. A zero gradient in a differentiable convex problem certifies **a global minimum at that point**.

Further reading: [MIT course overview](https://ocw.mit.edu/courses/18-100a-real-analysis-fall-2020/). The official scope is linked as supplementary reading; the small examples and arguments here are independently constructed.
