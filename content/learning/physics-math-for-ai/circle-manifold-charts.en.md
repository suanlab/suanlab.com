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

## Exercise and explained answer

Move the point to θ=π and select the first chart. There is no selected coordinate. Switch to the second chart without moving the point: u=0. Now move to θ=0. The first chart works with t=0, while the second fails. This pair covers the circle. Explain why evaluating 1/t at t=0 would misuse the transition map: that point is outside the overlap.

## Coordinates in AI models

Local latent coordinates, rotation parameterizations and manifold representations can suffer chart singularities. A huge coordinate value may reflect a poor chart choice rather than a large physical change. Track chart domains and transition rules when differentiating or comparing representations. The displayed curve is the physical circle in equal-scale x,y axes; coordinate values belong to the readout, so a coordinate blow-up does not flatten or enlarge the object itself.

Further reading: [John M. Lee, Introduction to Smooth Manifolds](https://sites.math.washington.edu/~lee/Books/ISM/).
