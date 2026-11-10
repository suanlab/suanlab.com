## A generator specifies continuous motion

A matrix exponential converts an infinitesimal generator into a finite group element. For a constant angular vector ω, let Ω=hat(ω). A translational generator v combines with it into a twist matrix ξ̂=[[Ω,v],[0,0]]. The trajectory T(t)=exp(tξ̂) satisfies Ṫ=ξ̂T and T(0)=I. This convention fixes a constant spatial twist; body-frame conventions must be stated separately.

The angular vector need not be a unit axis. Its norm times t is the rotation angle, and its direction supplies the axis when nonzero. The controls expose all three angular components and all three translational generator components, together with time. Thus the calculation covers arbitrary constant se(3) twists within the stated finite ranges.

## Rodrigues and the coupled translation

Write W=tΩ and θ=||tω||. The exponential has the form

$$
R=I+\frac{\sin\theta}{\theta}W+
\frac{1-\cos\theta}{\theta^2}W^2,
\qquad
p=J(t\omega)\,tv,
$$

$$
J=I+\frac{1-\cos\theta}{\theta^2}W+
\frac{\theta-\sin\theta}{\theta^3}W^2.
$$

The translation also equals ∫₀ᵗexp(sΩ)v ds. The generator's translation direction rotates during motion, so simply appending tv after exponentiating Ω generally gives a different transformation. The blue curve is the xy projection of the exact origin trajectory; orange is that naive straight translation. Full xyz and the orientation appear in the readout and oblique frame drawing.

## Work the quarter-turn motion

Choose ω=(0,0,π/2), v=(1,0,0), t=1. Integration gives p=(sinωz/ωz,(1−cosωz)/ωz,0)=(2/π,2/π,0), approximately (0.637,0.637,0). Naive tv=(1,0,0) differs by about 0.733. This is genuine rotation–translation coupling, not a numerical integration error.

## The zero-angle limit is regular

Direct formulas contain divisions by θ, but their coefficients have finite limits. Near zero we use

$$
\frac{\sin\theta}{\theta}=1-\theta^2/6+\cdots,\quad
\frac{1-\cos\theta}{\theta^2}=1/2-\theta^2/24+\cdots,\quad
\frac{\theta-\sin\theta}{\theta^3}=1/6-\theta^2/120+\cdots.
$$

Consequently R=I and J=I at zero angular generator, giving pure translation p=tv. Series also prevent subtractive cancellation for small nonzero angles. The implementation is checked against an independently summed four-by-four matrix exponential and a separate integration of exp(sΩ)v.

## Exercise and explained answer

Set ω=0 with v=(1,0,0): the curved path becomes the straight unit translation. Next set ω=(0,0,2π) and v=0. The generator is nonzero but its time-one rotation is identity. Thus exponential coordinates are not globally one-to-one. For this same full rotation with v=(1,0,0), the transverse translation averages to zero; it does not become one unit forward.

## Research connection

Exponential updates are useful for pose optimization and continuous-time motion models, but frame conventions, units and multiplication order remain essential. A principal rotation logarithm requires branch choices and has special behavior near angle π. This lab computes exponentials; it does not silently assume a globally unique inverse logarithm.

Further reading: [Lynch and Park, Modern Robotics, Chapter 3](https://modernrobotics.northwestern.edu/chapters/chapter3/).
