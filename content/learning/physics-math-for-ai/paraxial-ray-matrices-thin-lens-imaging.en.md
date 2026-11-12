## Why can a camera form an image?

A camera does more than collect light. Light leaving one object point in many directions must arrive at nearly one sensor point if the object is to be resolved. A bright detector reading alone does not establish this correspondence. An image is a relationship between object coordinates and detector coordinates, with a tolerance for blur.

Begin with an ideal thin lens and a fan of rays leaving one point. Track the fan first by elementary geometry and then by a matrix. The two calculations should agree. We will distinguish the plane at which the fan actually meets, a displaced sensor plane, a virtual backward intersection and an image at infinity. These distinctions are useful when designing camera models for computer vision, interpreting optical experiments and differentiating a simulator during inverse design.

This is the geometrical part of the topic. The following Gaussian-beam lesson explains why finite-wavelength focusing cannot generally be represented by a zero-width ray intersection. Neither lesson substitutes for a complete treatment of finite pupils, aberrations or electromagnetic boundary conditions.

## Coordinates, units and assumptions

Put the lens at longitudinal coordinate $z=0$ and orient actual propagation toward increasing $z$. A real object point is at $z=-u$, with object distance $u>0$ and transverse height $x_0$. The physical sensor is at $z=s\geq0$. A computed image coordinate $v$ can be positive, negative or unavailable as a finite number.

Use the ray state $(x,\theta)^T$. Height $x$ is measured in metres. The variable $\theta$ is the small paraxial angle, treated as the slope $dx/dz$. A radian is dimensionless, but keeping the label rad helps distinguish angle from height. Exact geometry would use $dx/dz=\tan\theta$; replacing this with $\theta$ is an approximation.

Assume a centred, ideal, infinitesimally thin lens, a common background refractive index, small slopes, and no clipping. The lens has signed focal length $f\neq0$: positive for converging action and negative for diverging action in this convention. Reflection, absorption, chromatic response, finite lens thickness and spherical aberration are not calculated. A large computed slope is evidence that the chosen model needs scrutiny, not a valid prediction from full Maxwell theory.

## A ray is a direction of travel, not a material thread

In a short-wavelength description, rays follow local propagation directions associated with phase fronts. A ray diagram is a model of transport geometry. It is not a picture of a thin luminous wire and it carries no automatic detector intensity unless weights and a measurement model have been supplied.

Our nine-ray fan samples directions uniformly between a central angle minus a half-angle and the central angle plus a half-angle. Equal sampling is a numerical construction, not an asserted angular radiation law of the object. The reported standard deviation of the nine sensor coordinates describes this sampled fan. It is not a calibrated optical point-spread function or a diffraction-limited resolution.

This distinction prevents a common error: a narrow plotted fan does not prove high resolution if the aperture, wavelength and sensor response are absent from the model. Conversely, one sampled ray reaching a point does not prove that all rays from that object point focus there.

## Derive free propagation by similar triangles

Over signed mathematical displacement $d$, a paraxial ray evolves as

$$
x'=x+d\theta,\qquad\theta'=\theta,
\qquad
P(d)=\begin{pmatrix}1&d\\0&1\end{pmatrix}.
$$

For actual forward propagation, $d\geq0$. A negative $d$ can be used to extend a line backward on paper; it does not describe light physically travelling backward through the current experiment. The same algebra can represent both constructions, so the interpretation must accompany the number.

A useful units check is that the upper-right matrix entry has units of length and the lower-left entry has units of inverse length. The diagonal entries are dimensionless. A matrix whose entries are all labelled dimensionless would silently mix incompatible coordinates.

## Derive the thin-lens angular kick

The thin lens changes the ray direction at its plane while leaving the transverse coordinate continuous:

$$
x^+=x^-,\qquad
\theta^+=\theta^- -\frac{x^-}{f},\qquad
L(f)=\begin{pmatrix}1&0\\-1/f&1\end{pmatrix}.
$$

For a positive lens, an initially parallel ray above the axis acquires a downward slope and crosses the axis at distance $f$ in the geometrical model. A negative lens bends that same ray farther upward. A ray at the ideal lens centre has no angular kick in this model.

These signs can be checked without memorizing a matrix. Start with $x>0$ and $\theta^-=0$: a converging lens must make $\theta^+<0$. A sign convention that fails this test has either changed the coordinate direction or implemented the wrong lens action.

## Compose transformations in their physical order

To go from the object plane to a sensor after the lens, first propagate by $u$, then apply the lens, then propagate by $s$:

$$
M(s)=P(s)L(f)P(u)
=\begin{pmatrix}
1-s/f&u+s-us/f\\
-1/f&1-u/f
\end{pmatrix}
=\begin{pmatrix}A&B\\C&D\end{pmatrix}.
$$

The rightmost factor acts first. Interchanging a propagation and a lens generally changes the experiment. The output height is $x_s=A x_0+B\theta_0$. The term multiplying the input angle will determine whether different directions from one object point meet at the sensor.

With the lens absent, use the identity in place of $L$. Then $M=P(u+s)$, so $x_s=x_0+(u+s)\theta_0$. Do not imitate an absent lens with an arbitrary enormous finite focal length if an exact identity transformation is available.

## Obtain the imaging condition instead of quoting it

For fixed object height $x_0$, an ideal image plane makes the output height independent of launch angle. Therefore $B=0$ at $s=v$:

$$
u+v-\frac{uv}{f}=0,
\qquad
\frac1u+\frac1v=\frac1f,
\qquad
v=\frac{fu}{u-f},\quad u\neq f.
$$

At that plane, the transverse magnification is

$$
m=A(v)=1-\frac vf=-\frac vu,
\qquad x_i=m x_0.
$$

A negative magnification reverses the transverse image orientation. It is not negative brightness. The output slope still depends on the input ray, even when output height does not. Focusing a fan to one position does not collapse every phase-space coordinate to the same state.

Some texts order the ray vector as $(n\alpha,y)$, with angle first and height second. Their relevant off-diagonal imaging entry is then named $C$, not our $B$. Matrix letters have meaning only after the coordinate order and refractive-index convention are declared.

## Worked real image: equal object and image distances

Let $f=0.10$ m, $u=0.20$ m and $x_0=1$ mm. Then $v=0.20$ m, $m=-1$ and $x_i=-1$ mm. The complete matrix is

$$
M(v)=\begin{pmatrix}-1&0\\-10&-1\end{pmatrix},
$$

with the lower-left entry understood in m$^{-1}$. A launch angle of $-0.01$, $0$ or $0.01$ rad changes the height at which the ray strikes the lens, but all three reach $-1$ mm at the ideal image plane.

Follow the central launch angle explicitly. It reaches the lens at $1$ mm, receives angular kick $-0.001/0.10=-0.01$ rad and travels another $0.20$ m. Its final height is $0.001-0.002=-0.001$ m. This elementary calculation checks the matrix signs, the units and the image inversion at once.

## A displaced sensor is a different question

Keep that object and lens but move the sensor to $s=0.15$ m. Now $A=-0.5$ and $B=0.05$ m, so

$$
x_s=-0.0005+0.05\theta_0\quad\text{in metres}.
$$

For the symmetric sampled fan $\theta_0\in[-0.01,0.01]$ rad, the sensor coordinates range from $-1$ mm to $0$ mm. The span is $1$ mm; the fan centre is at $-0.5$ mm. The image still exists at $v=0.20$ m, but this sensor is not placed there.

Reducing the sampled angular fan reduces its geometrical blur. Real aperture reduction also changes throughput and diffraction. Those effects require an explicit pupil and wave model; they cannot be inferred from the current ray fan alone. A zero half-angle gives nine coincident samples of a single direction, not proof of a sharply imaged extended angular distribution.

## Virtual images and the meaning of a negative distance

Set $u=0.05$ m and retain $f=0.10$ m. Then $v=-0.10$ m and $m=2$. Forward rays leave the lens without meeting at a real plane behind it. Their backward extensions intersect at $x_i=2$ mm and longitudinal coordinate $-0.10$ m. The virtual image is upright and enlarged in this example.

A detector cannot be placed at the backward intersection and expected to record a real focused image from these forward output rays. Another optical system, such as an eye or a camera lens, can respond to the diverging wavefront as if it originated from that location. The interpretation concerns the output field geometry, not backward energy transport.

A negative lens with $f=-0.10$ m and a real object at $u=0.20$ m gives $v=-0.0666667$ m and $m=1/3$. This virtual image is upright and reduced. A virtual image is therefore not synonymous with enlargement.

## An image at infinity is not a very distant sentinel

When $u=f>0$, $B(s)=f$ for every finite sensor distance. Rays leaving one object point have the same output slope $-x_0/f$, so the outgoing fan is parallel. There is no finite image plane for that point in this ideal system.

Return an unavailable finite image coordinate, magnification and image-height calculation. Do not replace infinity by 9999 m or saturate a display at a chosen finite maximum while labelling it the computed image. A finite sensor fan can still be shown and calculated at its own specified coordinate.

Nearby inputs are different. With $f=0.10$ m and $u=0.1001$ m, the finite image is $100.1$ m away. Approaching the focal distance from opposite sides gives large finite real or virtual coordinates. Exact singular input and a nearby nonsingular input must not be merged by a convenient rounding tolerance.

## Conditioning and phase-space conservation

Differentiate the finite image coordinate with respect to object distance:

$$
\frac{dv}{du}=-\frac{f^2}{(u-f)^2}.
$$

Near the focal plane, a small error in object distance produces a very large change in inferred image distance. This is a conditioning issue in the physical inverse problem. Reporting many digits from a floating-point formula does not make the inferred distance experimentally precise.

Both $P$ and $L$ have determinant one, and so does their product. In the same-index paraxial model they preserve oriented area in $(x,\theta)$ phase space. A bundle may narrow in position while spreading in angle. With changing refractive index, a canonical coordinate such as $n\theta$ changes the bookkeeping; do not carry this determinant statement into an undeclared interface model.

The derivative also helps differentiate optical simulators for AI. Optimization can become unstable near singular configurations even if every matrix multiplication is implemented correctly. Inspect sensitivity, units and feasible measurement locations before interpreting a large learned gradient as useful design information.

## Connect the ray condition to stationary wave phase

For positive propagation distances $u$ and $v$, let $h$ be a transverse integration coordinate at an ideal lens. The paraxial path and lens phase, divided by $k$, contain

$$
\Phi(h)=\frac{(h-x_0)^2}{2u}-\frac{h^2}{2f}
+\frac{(x-h)^2}{2v}.
$$

Its derivative is $h(1/u+1/v-1/f)-x_0/u-x/v$. When the imaging condition holds and $x=-v x_0/u$, that derivative vanishes for every $h$ in this ideal quadratic model. The geometrical imaging relationship is thus consistent with the stationary structure of the paraxial wave phase.

This calculation is not a claim that every Fermat path is a global shortest path. Nor does it give the full wave amplitude at a singular ideal point image. A finite pupil, a finite source distribution and diffraction determine the actual observed field. Negative mathematical image distance denotes an extrapolation here, not an instruction to run a physical forward Fresnel integral over a negative experimental path.

## How to inspect a diagram and a simulator

Predict the image sign and rough size before looking at a computed value. Then compare a launch ray, the complete fan and the sensor plane. Use one physical transverse scale across the stages; normalizing each ray segment separately would hide geometric expansion. If longitudinal and transverse screen scales differ for visibility, state that the drawing is stretched and do not read a visual line angle as the actual slope.

Display a negative image coordinate with a distinct backward extension. Keep the real sensor and forward rays at nonnegative positions after the lens. At $u=f$, remove a finite image marker while retaining the valid outgoing parallel fan. A slope diagnostic should use the calculated slopes, not the apparent angle of a stretched diagram.

For numerical review, compare direct updates $x_L=x_0+u\theta_0$, $\theta_+=\theta_0-x_L/f$ and $x_s=x_L+s\theta_+$ with the matrix result. Test positive and negative lenses, absent lens, displaced sensor, exact focal input and nearby floating values. These cases exercise different physical meanings, not merely different colours of the same picture.

## Research extension: finite sensor data can coexist with divergent image uncertainty

The image-distance derivative above is a local sensitivity. It does not prove that a global uncertainty distribution has a finite mean or variance. We now push a simple continuous preparation uncertainty through the exact thin-lens formula and compare it with observations at a specified finite sensor. This supplementary distribution is distinct from the native nine-ray angular fan; no uncertainty slider is added.

### Put uncertainty on the actual object distance


$$
f>0,\quad u=f+d,\quad d\sim\operatorname{Uniform}[\mu-a,\mu+a],\quad 0<\mu<a,\quad f+\mu-a>0:\qquad v=f+\frac{f^2}{d}\quad(d\ne0).
$$

All lengths are in metres. Every object is real and lies before the lens because u>0. Some realizations have u<f and hence virtual images; others have u>f and real images. The exact focal preparation d=0 has probability zero under this continuous model and no finite image coordinate. Its small probability does not control the nearby large coordinates.

### Transform the distribution and retain both branches


$$
w=v-f,\quad d=\frac{f^2}{w}:\qquad p_v(v)=\frac{f^2}{2a(v-f)^2}\quad\text{on }v\le f-\frac{f^2}{a-\mu}\ \text{or }v\ge f+\frac{f^2}{a+\mu},\qquad p_v(v)=0\text{ otherwise}.
$$

Use the input density 1/(2a) and the absolute Jacobian f²/w² separately on the negative and positive d branches. Their probabilities are (a−mu)/(2a) and (a+mu)/(2a). The gap in image coordinates is real in this distribution; joining the branches with a plotted line would invent impossible outputs. A negative image coordinate remains a backward extrapolation, not a detector location reached by forward light.


$$
R\ge\frac{f^2}{a-\mu}:\quad\Pr(|v-f|>R)=\frac{f^2}{aR},\qquad\mathbb E|v-f|=\infty,\quad\mathbb E(v-f)^2=\infty.
$$

The tail follows directly from abs(d)<f²/R. Its 1/R probability decay gives a logarithmically divergent first absolute moment and a divergent second moment. Each signed tail contributes an infinite first moment, so the ordinary mean of v does not exist; a chosen principal value is a different definition. Subtracting a nominal coordinate or a finite constant cannot restore a finite variance. Finite Monte Carlo samples always give finite empirical moments and cannot prove that the population moments exist.

### Measure how a finite output cutoff hides the divergence


$$
T_2(R)=\mathbb E\!\left[(v-f)^2\mathbf1_{\{|v-f|\le R\}}\right]=\frac{f^2R}{a}-\frac{f^4}{2a}\left(\frac1{a-\mu}+\frac1{a+\mu}\right),\quad R\ge\frac{f^2}{a-\mu}.
$$

Integrate f⁴/d² only where abs(d)≥f²/R. T2 is an unnormalized truncated second moment in m², not a variance and not a conditional RMS. With f=0.1 m, mu=0.0005 m and a=0.001 m, the virtual-image probability is 1/4. The table changes only an analysis cutoff; it does not move the sensor or clip the native image-distance calculation.

| cutoff R (m) | omitted probability | truncated second moment T2 (m²) |
|---|---|---|
| 20 | 0.5 | 66.6666667 |
| 40 | 0.25 | 266.666667 |
| 80 | 0.125 | 666.666667 |
| 160 | 0.0625 | 1466.66667 |

```python
from math import isclose

# Distances in metres; a continuous uncertainty model, not the nine-ray fan.
f, mu, a = .1, .0005, .001
assert 0 < mu < a and f+mu-a > 0
for cutoff in [20, 40, 80, 160]:
    h = f*f/cutoff
    assert h <= a-mu or isclose(h, a-mu, rel_tol=1e-14, abs_tol=0)
    tail = h/a
    # Integrate f^4/d^2 on [-a+mu,-h] and [h,a+mu].
    second = f**4/(2*a)*(2/h-1/(a-mu)-1/(a+mu))
    print(cutoff, format(tail, '.9g'), format(second, '.9g'))
```


$$
v_{\rm nominal}=f+\frac{f^2}{\mu}=20.1\ \mathrm m,\qquad\sigma_{v,\rm linear}=\frac{f^2}{\mu^2}\frac a{\sqrt3}\approx23.094\ \mathrm m.
$$

The linear prediction is finite, but the exact distribution has no ordinary standard deviation. The error is not incorrect floating-point division: extending a local derivative across a pole has changed the statistical conclusion. A nominal prediction need not equal a distribution mean.

### Keep the finite sensor as a separate observable


$$
x_z=(1-z/f)x_0+[u(1-z/f)+z]\theta_0,\qquad z\ge0\text{ fixed}.
$$

This is the same sequential propagation as the native ray calculator: xL=x0+u theta0, output slope theta0−xL/f, then xz=xL+z times output slope. Here x0, theta0, f and sensor z are fixed; only u is uncertain. The formula is affine in u and has no image-distance denominator.


$$
\operatorname{Var}(x_z)=\theta_0^2(1-z/f)^2\frac{a^2}{3}.
$$

For the example above, x0=1 mm, theta0=0.01 rad and z=0.2 m give mean sensor height −0.005 mm and standard deviation about 5.7735 micrometres. These are finite uncertainty statistics of a single fixed-angle ray over the object-distance distribution. They are not the native fan RMS, a diffraction point-spread function or a claim of optical resolution. Divergent inferred conjugate-image moments do not imply divergent observations on this fixed sensor.

### State a regime where a local expansion is controlled


$$
d=\mu+e,\quad |e|\le a<|\mu|:\qquad v-v_{\rm linear}=\frac{f^2e^2}{\mu^2(\mu+e)},\quad |v-v_{\rm linear}|\le\frac{f^2a^2}{\mu^2(|\mu|-a)}.
$$

This is a different uncertainty regime, bounded away from the pole. The identity follows by subtracting the exact first-order rational expansion; the denominator has magnitude at least abs(mu)−a. It gives an explicit absolute approximation budget rather than a generic instruction to make the noise small. For inverse-design losses, state whether the target is a conjugate coordinate, a finite sensor reading or a windowed statistic.

![A small uniform object-distance uncertainty straddles the focal condition; its real and virtual image branches have inverse-square tails and a gap, with the plotted image window explicitly cropped.](/assets/learning/waves-ray-image-uncertainty.en.svg)

### Research questions with explanations

**Does zero probability of the exact pole guarantee finite uncertainty?** **Answer:** No. Arbitrarily close preparations produce inverse-square image-coordinate tails. Mean and variance require integrability, not merely exclusion of one point.

**Must a fixed camera sensor inherit this divergent image variance?** **Answer:** No. Its coordinate is affine in u under the stated preparation. Report the correct observable instead of substituting a moving mathematical image plane.

**When is linear uncertainty propagation defensible?** **Answer:** Specify a distribution bounded away from the pole and check the displayed remainder bound against the required precision. A finite simulation or a small nominal error bar alone is insufficient.

## Explained exercises and research extensions

### 1. Verify the real image without a matrix

For $u=0.20$ m, $f=0.10$ m and $x_0=1$ mm, launch a ray at $0.01$ rad. It reaches the lens at $3$ mm. Its output slope is $0.01-0.003/0.10=-0.02$ rad, giving $x(0.20)=0.003-0.004=-1$ mm. Compare with the zero-angle calculation above. Their agreement at one plane is the imaging property.

### 2. Diagnose a wrong multiplication order

A program uses $L(f)P(u+s)$ for the same setup. It predicts the lens kick after the entire free path. This has changed the physical location of the element; it is not an equivalent algebraic simplification. Check the output height dependence on launch angle and show why it cannot reproduce the desired image matrix.

### 3. Compare a virtual image with the sensor reading

Use $u=0.05$ m, $f=0.10$ m and $x_0=1$ mm. Derive $v=-0.10$ m and $x_i=2$ mm. A zero-angle ray leaves the lens with slope $-0.01$ rad. Its backward value at $-0.10$ m is $2$ mm, while its forward value at $s=0.10$ m is $0$. These are different coordinates and different constructions.

### 4. Separate an infinite image from a neighbouring finite one

For $u=f=0.10$ m, show that every launch angle produces output slope $-x_0/f$. Then set $u=0.1001$ m and obtain $v=100.1$ m. The sensitivity magnitude there is $10^6$. An object-distance error of order a micrometre can therefore matter strongly; linear error propagation is local and should not be extended across the singularity.

### 5. Explain why a zero fan width does not prove resolution

At a displaced sensor, set fan half-angle to zero. The numerical coordinate variance vanishes because only one direction has been sampled repeatedly. It says nothing about omitted angular directions, pupil diffraction or sensor integration. Design a meaningful comparison by retaining the preparation while changing the sensor position.

### 6. Connect an optical forward model to an AI loss

Suppose a calibrated dataset supplies object and image heights at several known distances. A loss can compare measured coordinates to the finite image model with explicit units and uncertainty. Exclude or model infinite-image preparations and distinguish physical sensor data from mathematical image coordinates. Add wavelength, pupil and aberration effects only when the measurement requires them. A low training loss on ideal ray coordinates does not validate a learned simulator for diffraction images.

## Sources and scope of further study

[TU Delft, Geometrical Optics](https://interactivetextbooks.tudelft.nl/tn2421/content/Chap2_Geometrical_optics/GeometricalOptics_2022_01Clean.html) provides ray-matrix and lens material with its own coordinate order. [MIT 6.974 course notes](https://ocw.mit.edu/courses/6-974-fundamentals-of-photonics-quantum-electronics-spring-2006/bf3bdeb971819ddc9c68719d28bb2745_classnotes.pdf), sections on rays, lens imaging and Gaussian propagation, connect ray matrices to wave optics. The formulas here were derived with the declared order and signs and checked against sequential geometry and an independent stationary-phase calculation.

Thick lenses, multiple indices, aberrations, finite apertures, vector focusing and calibrated sensor noise need additional models. The next lesson keeps the ideal thin lens but supplies a finite-width Gaussian field and direct Fresnel checks, so that geometric focusing and diffraction can be compared on the same declared assumptions.
