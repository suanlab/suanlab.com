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

## Derive the Jacobian and identify its singular directions

![A full-turn twist closes its transverse origin path while naive translation does not; the two transverse Jacobian singular values vanish at two pi.](/assets/learning/symmetry-twist-en.svg)

For Ω=hat(ω), the identity Ω³=−‖ω‖²Ω reduces the exponential power series to a combination of I,Ω,Ω². Integrating that series term by term yields J in the displayed translation formula. This establishes the formula for arbitrary axes, not just a z-axis example.

Let n be the unit rotation axis and split a vector into parallel and perpendicular parts. The translation Jacobian acts as

$$
Jv=v_\parallel+\frac{\sin\theta}{\theta}v_\perp
+\frac{1-\cos\theta}{\theta}(n\times v),\qquad
\sigma(J)=\left\{1,\frac{2|\sin(\theta/2)|}{|\theta|},\frac{2|\sin(\theta/2)|}{|\theta|}\right\}.
$$

At θ=0 use the continuous limit. At a nonzero full turn 2πk, only the parallel direction survives and J has rank one. Losing transverse translation there is a genuine singularity of these exponential coordinates, distinct from choosing a principal rotation-logarithm branch at π. The group itself has not become singular.

The implementation uses longer small-angle series and a half-angle expression for 1−cosθ. Away from zero it evaluates translation in axis components, avoiding cancellation against identity near a full turn. At θ=2π−2⁻⁴⁸, a unit transverse generator has components about −6.0441×10⁻¹⁶ and 1.1477×10⁻³⁰. Both should survive. The code below independently sums sine and cosine at 100 decimal digits from the exact binary64 input; nominal exact quarter/full turns in the lab use its explicit cardinal-angle convention.

| Angle | J's singular values | Interpretation |
|---|---|---|
| 0 | 1,1,1 | Pure-translation limit |
| π | 1,2/π,2/π | Invertible J despite log branch choice |
| 2π | 1,0,0 | Rank-one translation Jacobian |

## Reproduce the calculation in Python

```python
from decimal import Decimal as D, localcontext
from math import pi
with localcontext() as ctx:
    ctx.prec=100
    for value in [1e-4,.001,2*pi-2**-48]:
        x=D.from_float(value); sine=x; term=x
        for k in range(1,200):
            term *= -x*x/D(2*k*(2*k+1)); sine += term
        cosine=D(1); term=D(1)
        for k in range(1,200):
            term *= -x*x/D((2*k-1)*(2*k)); cosine += term
        A=sine/x; B=(1-cosine)/(x*x); C=(x-sine)/(x*x*x)
        print('angle:',value,'A,B,C:',*[float(v) for v in [A,B,C]])
        print('transverse translation:',float(A),float((1-cosine)/x))
```

## Exercise and explained answer

Set ω=0 with v=(1,0,0): the curved path becomes the straight unit translation. Next set ω=(0,0,2π) and v=0. The generator is nonzero but its time-one rotation is identity. Thus exponential coordinates are not globally one-to-one. For this same full rotation with v=(1,0,0), the transverse translation averages to zero; it does not become one unit forward.

## Research connection

Exponential updates are useful for pose optimization and continuous-time motion models, but frame conventions, units and multiplication order remain essential. A principal rotation logarithm requires branch choices and has special behavior near angle π. This lab computes exponentials; it does not silently assume a globally unique inverse logarithm.

Further reading: [Lynch and Park, Modern Robotics, Chapter 3](https://modernrobotics.northwestern.edu/chapters/chapter3/).
