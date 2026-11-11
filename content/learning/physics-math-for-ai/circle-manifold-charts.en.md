## A curved object can have simple local coordinates

A circle is one-dimensional: near any point, a single real parameter describes nearby points smoothly. Yet no single real chart covers the entire circle. An angle has a seam, and a stereographic coordinate has an excluded point. The failure of one coordinate is not a geometric singularity of the circle.

A smooth manifold has compatible charts on open neighborhoods, together with the usual Hausdorff and second-countability conditions. A chart is a homeomorphism to an open subset of Euclidean space; smoothness is expressed by smooth transition maps on overlaps. The lab uses two explicit charts on the standard unit circle.

## Derive the first stereographic coordinate

Write p(θ)=(cos θ,sin θ). Projection from (−1,0) onto the vertical line x=0 gives

$$
t=\frac{y}{1+x}=\tan(\theta/2),\qquad
x=\frac{1-t^2}{1+t^2},\quad y=\frac{2t}{1+t^2}.
$$

The coordinate is defined on the circle with (−1,0) removed. For every real t the inverse formula gives a point on that open circle neighborhood. Near θ=π, t becomes large; at θ=π it is undefined. The blue circle remains perfectly regular. The open orange marker marks the selected chart's excluded point, and the filled marker marks the current point.

## Cover the missing point with another chart

Projection from (1,0) gives u=y/(1−x), which excludes the opposite point. Its inverse has x=(u²−1)/(u²+1) and y=2u/(1+u²). At the left point u=0 is valid even though t is not.

On the overlap, neither t nor u is zero and

$$
u=\frac1t,\qquad \frac{du}{dt}=-\frac1{t^2}.
$$

Both coordinates are smooth there. A negative transition derivative means the two coordinate orientations differ; it does not indicate a broken manifold. At the upper point t=u=1 and the derivative is −1. The overlap has two components, corresponding to the upper and lower semicircles after removing both poles.

## Derive a metric instead of measuring coordinate size

![A smooth unit circle with its excluded chart point, and a metric coefficient that decreases for large stereographic coordinates.](/assets/learning/geometry-charts-en.svg)

Differentiate the inverse stereographic parameterization. Squaring and adding its two derivatives gives the pulled-back circle metric. On the overlap, the transition derivative converts that metric exactly:

$$
\left\|\frac{dp}{dt}\right\|^2=\frac{4}{(1+t^2)^2},\qquad
\frac{4}{(1+u^2)^2}\left(\frac{du}{dt}\right)^2
=\frac{4}{(1+t^2)^2},\quad u=1/t.
$$

Large coordinate increments can thus represent small physical lengths near the excluded pole. A coordinate magnitude is not a distance. At t=2, u=1/2, the metric coefficients are 4/25 and 64/25; multiplying the latter by the squared transition derivative 1/16 recovers 4/25.

Why can no single chart cover the circle? A homeomorphism would send the compact circle to a nonempty compact open subset of the real line. Such a subset cannot exist: compact subsets are closed and bounded, while the only clopen subsets of the connected real line are empty and the whole line. This argument uses the chart's open-image requirement.

The exact-fraction code checks reconstruction and metric transformation away from t=0. It deliberately never applies the transition at a point outside its overlap.

| t | u | Metric in t |
|---:|---:|---:|
| 1 | 1 | 1 |
| 2 | 1/2 | 4/25 |
| -2 | -1/2 | 4/25 |

## Reproduce the calculation in Python

```python
from fractions import Fraction as F
for t in map(F, [1, 2, -2]):
    u = 1 / t
    x, y = (1-t*t)/(1+t*t), 2*t/(1+t*t)
    gt, gu = 4/(1+t*t)**2, 4/(1+u*u)**2
    assert x*x+y*y == 1
    assert (u*u-1)/(u*u+1) == x and 2*u/(1+u*u) == y
    assert gu/t**4 == gt
    print(t, u, gt)
```

## Exercise and explained answer

Move the point to θ=π and select the first chart. There is no selected coordinate. Switch to the second chart without moving the point: u=0. Now move to θ=0. The first chart works with t=0, while the second fails. This pair covers the circle. Explain why evaluating 1/t at t=0 would misuse the transition map: that point is outside the overlap.

## Coordinates in AI models

Local latent coordinates, rotation parameterizations and manifold representations can suffer chart singularities. A huge coordinate value may reflect a poor chart choice rather than a large physical change. Track chart domains and transition rules when differentiating or comparing representations. The displayed curve is the physical circle in equal-scale x,y axes; coordinate values belong to the readout, so a coordinate blow-up does not flatten or enlarge the object itself.

Further reading: [John M. Lee, Introduction to Smooth Manifolds](https://sites.math.washington.edu/~lee/Books/ISM/).
