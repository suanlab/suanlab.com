## Changing fields and moving measuring paths

Imagine holding a rectangular wire still while changing the magnetic field. Then imagine moving one edge of a wire in a constant magnetic field. Flux can change in either case, but the terms needed to describe the force on charge differ.

The preceding Maxwell lesson used a fixed measuring path. Here we prescribe the motion of the path separately and resolve the two contributions. The goal is to distinguish field variation, surface variation, and orientation within a changing-flux measurement. Review cross products, line integrals, and the chain rule first.

## Define electromotive force and its units

Despite its name, electromotive force is not a force measured in N. It is the integral of tangential force per unit charge around a circuit and has units V. Here u is the prescribed velocity of the wire element. Compute the E contribution and u×B contribution separately. We assume a thin wire and nonrelativistic motion; the magnetic force from drift along the wire has no tangential contribution to add.

In a general induction experiment, emf is not simply a single scalar potential difference between endpoints. A nonzero closed E integral cannot be represented solely by a path-independent electrostatic potential.

$$
\mathcal E(t)=\oint_{C(t)}\left(\mathbf E+\mathbf u\times\mathbf B\right)\cdot d\boldsymbol\ell,\qquad [\mathcal E]=\mathrm V.
$$

## Fixed and moving path laws

For a fixed open surface S with boundary C, Faraday law connects electric circulation to the time derivative of magnetic flux. A smoothly moving boundary and compatible surface require the motional term. Tie their orientations together by the right-hand rule. Differentiating an integral over a changing surface must account for the changing geometry.

This simple form does not automatically handle circuits that tear or change connectivity. We prescribe smooth rectangular motion so that the geometric assumptions remain explicit.

$$
\begin{aligned}\oint_C\mathbf E\cdot d\boldsymbol\ell&=-\frac{d}{dt}\int_S\mathbf B\cdot d\mathbf S&&\text{fixed }S,\\\oint_{C(t)}(\mathbf E+\mathbf u\times\mathbf B)\cdot d\boldsymbol\ell&=-\frac{d}{dt}\int_{S(t)}\mathbf B\cdot d\mathbf S&&\text{moving }S(t).\end{aligned}
$$

## The chosen fields and required sources

Prescribe a spatially uniform magnetic field along z inside a bounded region, with quadratic time dependence. Choose a circulating electric field consistent with that variation. This is a manufactured field in a prescribed region, not a full-space solution for a finite solenoid. External sources and boundary data are required.

For y>0, Ex follows the sign of b′; for x>0, Ey has the opposite sign. If b′>0, electric arrows circulate clockwise in the xy plane. The positive traversal for normal +z is counterclockwise, so electric circulation is negative.

$$
b(t)=b_0+b_1t+\tfrac12b_2t^2,\quad\mathbf B=(0,0,b(t)),\quad\mathbf E=\left(\tfrac12b'(t)y,-\tfrac12b'(t)x,0\right).
$$

## Is matching Faraday enough?

The electric curl is −b′ along z, while magnetic curl is zero. If b₂ is nonzero, changing E creates displacement current. Zero magnetic curl therefore requires an opposite prescribed conduction current. Omitting it and calling the family a complete source-free Maxwell solution would be wrong. Both field divergences and current divergence vanish, with ρ=0.

How a material supplies this current requires constitutive and driving models. Here we display the required J explicitly and test consistency between fields and sources.

$$
\nabla\times\mathbf E=(0,0,-b'),\quad\mathbf J=-\epsilon\partial_t\mathbf E=\left(-\tfrac12\epsilon b_2 y,\tfrac12\epsilon b_2 x,0\right),\quad\mathbf J+\epsilon\partial_t\mathbf E=0.
$$

## Connect to time-dependent potentials

The earlier potential lesson treated static fields. Here A changes in time, so E includes −∂tA. The particular choice below produces both prescribed fields. Knowing Faraday circulation alone does not uniquely determine local E. A physical curl-free electric component can be added with appropriate boundaries and sources, changing the physical field.

A coordinated gauge transformation of A and V instead preserves E and B and changes their representation. Keep these operations distinct. Global gauge conditions and choices receive further treatment later.

$$
\mathbf A=\left(-\tfrac12by,\tfrac12bx,0\right),\quad V=0,\quad\mathbf B=\nabla\times\mathbf A,\quad\mathbf E=-\nabla V-\partial_t\mathbf A.
$$

## Geometry of the moving rectangle

Traverse vertices (0,0)→(L,0)→(L,H)→(0,H), giving normal +z. Hold height fixed and vary width as L=L₀+ut. The input u is in m/s and is distinct from electric field or carrier drift.

The model supports L₀∈[1,2] m, u∈[−0.25,0.25] m/s, and t∈[−2,2] s, guaranteeing 0.5≤L≤2.5 m throughout the interval. Height satisfies 0≤H≤4 m. Collapsing-width paths lose regularity and are excluded explicitly, rather than silently repaired.

$$
L(t)=L_0+ut,\qquad\Phi_B(t)=b(t)L(t)H.
$$

## Electric integrals along four edges

On the bottom edge y=0, so Ex vanishes. On the right edge, integrate Ey=−b′L/2 along +y. On the top edge, integrate Ex=b′H/2 along −x. On the left edge x=0, so Ey vanishes. The two nonzero edges each contribute −b′LH/2, yielding −b′LH.

Even for a moving path, this calculation integrates E alone along its instantaneous shape. It does not yet include path velocity. Do not label this instantaneous electric integral as the full derivative of moving flux.

$$
\oint_{C(t)}\mathbf E\cdot d\boldsymbol\ell=-b'(t)L(t)H.
$$

## Read the motional contribution on the edges

If the right edge moves along +x, u×B points −y. Traversing that edge along +y gives −ubH. On horizontal edges the motional magnetic force is perpendicular to the path tangent, giving zero dot product; the left edge is stationary.

The model uses the continuous boundary velocity field (ux/L,0,0), representing stretching horizontal edges. Tangential redistribution of boundary points leaves this motional integral unchanged. This is a prescribed measuring contour, not a solution of microscopic rail or wire connections.

$$
\oint_{C(t)}(\mathbf u\times\mathbf B)\cdot d\boldsymbol\ell=-u\,b(t)H,\qquad\mathcal E=-H\big[b'L+bu\big]=-\frac{d\Phi_B}{dt}.
$$

## A worked example with units

Choose b₀=1 µT, b₁=0.5 µT/s, b₂=0.2 µT/s², L₀=1 m, u=0.2 m/s, H=2 m, and t=1 s. Then b=1.6 µT, b′=0.7 µT/s, and L=1.2 m. Flux is 3.84 µWb, electric circulation is −1.68 µV, motional contribution is −0.64 µV, and total emf is −2.32 µV.

Initial flux is 2 µWb, so its change from 0 to 1 s is 1.84 µWb. The time integral of emf is −1.84 µV·s. Multiplying the final instantaneous emf −2.32 µV by one second would not give the exact interval integral because emf varies in time.

## Why can motional emf exist if magnetic force does no work?

For the total velocity v of an individual charge, qv×B is perpendicular to v, so the magnetic force alone has zero instantaneous power on that charge. This does not prevent electrical energy transfer in a moving conductor. An external agent maintaining the conductor motion can supply mechanical energy.

Resolving motion into conductor and relative carrier motion reveals energy exchange between circuit-directed forces and forces associated with conductor motion. Actual current, resistive heating, and driving force require circuit and mechanical conditions. The present model does not calculate those responses.

$$
(q\mathbf v\times\mathbf B)\cdot\mathbf v=0.
$$

## Orientation reversal and degenerate area

Reversing traversal reverses electric and motional integrals, total emf, and associated flux, while physical Bz stays fixed. Mean Cartesian Bz obtained by dividing flux by signed area also stays fixed.

For H=0, area, flux, and integrals in this model are zero, but flux divided by area is undefined. A collapsing width L=0 with nonzero u is different: a one-sided shape rate can remain nonzero even at zero instantaneous area, and regular moving-path assumptions fail.

## Compute small flux changes stably

Flux is a cubic polynomial in time. Separately evaluating two large endpoints can erase a small difference. Cancel its constant term first and factor Δt. For example, with b₀=4 T, b₁=10⁻¹⁶ T/s and fixed unit area, naive subtraction from 0 to 1 s can give zero while the change is 10⁻¹⁶ Wb.

A constant physical B background differs from a potential gauge constant. If u is nonzero, b₀ contributes to flux change and motional emf and must not be discarded. Distinguish a small available value, exact zero, and arithmetic loss of a nonzero intermediate. The model conservatively marks some expressions unavailable without claiming that every alternative evaluation order must also fail.

$$
\begin{aligned}\Phi_B&=H(p_0+p_1t+p_2t^2+p_3t^3),\\p_1&=b_0u+b_1L_0,\quad p_2=b_1u+\tfrac12b_2L_0,\quad p_3=\tfrac12b_2u,\\\Delta\Phi_B&=H\Delta t\left[p_1+p_2(t_0+t_1)+p_3(t_0^2+t_0t_1+t_1^2)\right].\end{aligned}
$$

## AI models and independent verification

For learned sensor systems with moving trajectories, record both predicted fields and measurement geometry. Applying a fixed-path loss to moving-sensor data can misattribute the omitted motional contribution to field error. Check conduction current, displacement current, and boundary conditions separately.

A zero polynomial residual inside the model is not independent proof. Separate Python references integrate actual vector dot products along all four edges, then compare time integrals and five-point flux derivatives. Constant-B moving loops and changing-B fixed loops isolate the omitted mechanisms. Observed discrepancies describe selected numerical cases, not certified bounds.

## Research extension: close the induction loop with mechanics and energy

### Close the circuit and the mechanical problem

A prescribed moving contour determines emf, but it does not determine its own current or speed. Add a passive resistance R in Ω, a positive inductance ℒ in H and a rail mass m in kg; I is in A and u in m/s. Keep the same +z loop orientation: positive current runs up the right rail. For a constant imposed B along z, u×B points down the moving rail, whereas I dℓ×B points right for positive I and B. Thus the same signed coupling K=BH enters emf and force with the signs below. The two power terms cancel when the subsystems are combined. This is why a model trained only on the flux derivative can miss mechanical backreaction.

$$
K=BH,\quad\mathcal E_{\rm mot}=-Ku,\quad F_{\rm mag}=KI,\quad[K]={\rm T\,m}={\rm N/A},\quad[\mathcal E]={\rm V}.
$$

$$
\mathcal L\dot I+RI=-Ku,\qquad m\dot u=F_{\rm ext}+KI,\qquad\mathcal L>0,\ m>0,\ R\ge0.
$$

### State what the fixed inductance assumes

Use a linear, quasistatic circuit whose fixed series inductor dominates the position-dependent rail inductance. Treat ℒ, R, B and H as constant over the allowed motion, ignore radiation, and restrict the trajectory to a valid non-collapsing rail geometry. The imposed field source and its supports remain outside this reduced model. This is a declared energy-consistent approximation, not a complete field solution or a material model. Multiply the mechanical equation by u and the circuit equation by I; the ±KuI terms cancel and give the following energy balance. Holding speed constant requires an external agent supplying the stated power. That power initially charges the inductor as well as heating the resistor.

$$
\frac{d}{dt}\left(\frac12mu^2+\frac12\mathcal LI^2\right)=F_{\rm ext}u-RI^2,\qquad u\ \text{prescribed}:\quad F_{\rm ext}=-KI,\quad F_{\rm ext}u=\mathcal E_{\rm mot}I.
$$

### Compare imposed speed with free motion

For imposed speed and zero initial current, solve the first-order circuit equation by an integrating factor. The R=0 limit is finite at each fixed time and is not a steady state. At R>0, replacing I by −Ku₀/R is a late-time approximation requiring t much larger than ℒ/R. For a freely moving rail set F_ext=0 instead; its speed must respond. When K≠0 and u₀>0, define the dimensionless current j and time τ below. A prime in the next equations denotes differentiation with respect to τ. The sign σ records the orientation of B rather than hiding it in an absolute-value force.

$$
u=u_0,\ I(0)=0:\quad I(t)=-\frac{Ku_0}{R}(1-e^{-Rt/\mathcal L})\ (R>0),\qquad I(t)=-\frac{Ku_0}{\mathcal L}t\ (R=0).
$$

$$
\omega_0=\frac{|K|}{\sqrt{m\mathcal L}},\quad\zeta=\frac{R}{2\mathcal L\omega_0},\quad\tau=\omega_0t,\quad v=\frac{u}{u_0},\quad j=\sqrt{\frac{\mathcal L}{m}}\frac{I}{u_0},\quad\sigma=\operatorname{sign}K,\quad v'=\sigma j,\quad j'=-\sigma v-2\zeta j.
$$

### Keep lossless, critical and overdamped responses

Eliminate j to obtain the damped oscillator. With I(0)=0 the initial acceleration is zero, yet current starts changing immediately. For 0<ζ<1 the roots are a complex pair, at ζ=1 they coincide, and for ζ>1 they are distinct negative real roots. The displayed solution covers all three; ζ=0 gives undamped exchange. If K=0 these scaled variables are undefined: directly use constant free u and I(t)=I(0)exp(−Rt/ℒ). Larger R does not mean arbitrarily fast braking: the slow overdamped root approaches −1/(2ζ) in scaled time. The two exponentials can be evaluated directly in the overdamped formula to avoid multiplying a large cosh by a tiny exponential.

$$
v''+2\zeta v'+v=0,\qquad\lambda_{\pm}=-\zeta\pm\sqrt{\zeta^2-1},\qquad v(0)=1,\quad j(0)=v'(0)=0.
$$

$$
v=e^{-\zeta\tau}(C_\zeta+\zeta S_\zeta),\quad j=-\sigma e^{-\zeta\tau}S_\zeta,\qquad (C_\zeta,S_\zeta)=\begin{cases}(\cos(\beta\tau),\sin(\beta\tau)/\beta),&0\le\zeta<1,\ \beta=\sqrt{1-\zeta^2},\\(1,\tau),&\zeta=1,\\(\cosh(d\tau),\sinh(d\tau)/d),&\zeta>1,\ d=\sqrt{\zeta^2-1}.\end{cases}
$$

### Follow where the energy goes

The normalized kinetic, inductive and cumulative heat contributions add to one for the declared free initial state. At R=0, v=cos τ and j=−σ sin τ: energy returns from the inductor to motion, and the rail can reverse. “Magnetic braking is always monotone” is therefore false for this ideal lossless loop. For R>0 and K≠0 all eigenvalues have negative real part, so v and j tend to zero and the initial energy eventually becomes heat, as long as the model remains valid. The energy balance by itself establishes passivity; the spectrum establishes asymptotic decay. The table at τ=1 uses σ=+1. Its final column is total remaining energy divided by initial energy, not just kinetic energy.

$$
\mathscr W=\frac12(v^2+j^2),\quad\mathscr W'=-2\zeta j^2,\quad \frac{U_{\rm kin}}{U_0}=v^2,\quad\frac{U_{\rm ind}}{U_0}=j^2,\quad\frac{Q_{\rm heat}}{U_0}=1-v^2-j^2,\quad U_0=\frac12mu_0^2.
$$

### Check the approximation before changing the inductance

A moving physical loop can have ℒ(x), so simply substituting a variable inductance into ℒ I_dot is incomplete. Differentiating the self-flux adds ℒ′uI to the circuit equation. Magnetic coenergy ½ℒ(x)I² gives the additional force +½ℒ′I² at fixed current. Adding the two revised power equations and differentiating ½ℒ(x)I² restores the same total balance. Keeping only one correction would create or lose unaccounted power. This last identity is a consistency check, not a solved nonlinear trajectory. The constant-ℒ results above remain explicitly conditional on negligible geometric variation.

$$
\mathcal L=\mathcal L(x)>0:\quad \mathcal L\dot I+\mathcal L'(x)uI+RI=-Ku,\qquad m\dot u=F_{\rm ext}+KI+\frac12\mathcal L'(x)I^2,\quad \dot x=u.
$$

| zeta | v at tau=1 | j at tau=1 | remaining energy / U0 |
| --- | --- | --- | --- |
| 0.00000000 | 0.54030231 | -0.84147098 | 1.00000000 |
| 0.50000000 | 0.65970015 | -0.53350720 | 0.71983422 |
| 1.00000000 | 0.73575888 | -0.36787944 | 0.67667642 |
| 2.00000000 | 0.82226342 | -0.21390913 | 0.72187425 |

```python
from math import exp, sqrt, sin, cos
for zeta in [0.0, 0.5, 1.0, 2.0]:
    tau = 1.0
    if zeta < 1:
        d = sqrt(1-zeta*zeta)
        S = sin(d*tau)/d
        v = exp(-zeta*tau)*(cos(d*tau)+zeta*S)
        j = -exp(-zeta*tau)*S
    elif zeta == 1:
        v, j = exp(-tau)*(1+tau), -tau*exp(-tau)
    else:
        d = sqrt(zeta*zeta-1)
        slow, fast = exp((-zeta+d)*tau), exp((-zeta-d)*tau)
        S = (slow-fast)/(2*d)
        v, j = (slow+fast)/2+zeta*S, -S
    print(f"{zeta:.8f} {v:.8f} {j:.8f} {v*v+j*j:.8f}")
```

[MIT 8.02 Chapter 10, §10.9.3](https://ocw.mit.edu/courses/8-02-physics-ii-electricity-and-magnetism-spring-2007/ce1720fd4b21def8c2189ff4779f27f7_cha10faraday_law.pdf) provides the motional-emf, rail-force and external-power background. [Haus and Melcher §11.7](https://web.mit.edu/6.013_book/www/chapter11/11.7.html) develops magnetic coenergy and mechanical force. The coupled constant-parameter solutions and variable-inductance consistency identity here are derived for our stated reduced model. The existing prescribed-loop lab checks the same instantaneous motional sign; it does not solve this added circuit or mechanical trajectory.

![Electromechanical phase paths and the energy budget](/assets/learning/electromagnetism-induction-mechanical-backreaction.en.svg)

### Research questions with explanations

**Does a moving-loop emf alone determine its velocity?** **Answer:** No. Closing the circuit and mechanical equations requires resistance, inductance, mass, initial data and any external driving force.

**Must a lossless free rail always slow down?** **Answer:** No. At R=0 the ideal coupled system exchanges kinetic and inductive energy periodically and may reverse. Net energy stays constant.

**Can position-dependent inductance be inserted without changing force?** **Answer:** Generally no. Self-flux differentiation adds ℒ′uI, while linear magnetic coenergy adds +½ℒ′I² to the force. Both enter the power balance.

## Exercises with explanations

Choose the normal first, then calculate electric and motional terms separately.

### Exercise 1: Constant field and moving loop

For B=2 T, u=0.3 m/s, H=0.8 m and b′=0, find emf for the +z-oriented loop. **Answer:** Electric circulation is zero; motional emf is −0.48 V. A fixed-path electric-only calculation misses it.

### Exercise 2: Hold width fixed

If u=0 and b′>0, which term remains? **Answer:** Motional emf vanishes; electric circulation −b′LH remains and equals −d(bLH)/dt.

### Exercise 3: Reverse traversal

Reverse the worked-example traversal. What happens to total emf and Bz? **Answer:** Emf becomes +2.32 µV, while Bz remains 1.6 µT. The changed sign reflects measurement orientation.

### Exercise 4: A missing source

Can b₂ be nonzero while J=0 and spatial curl B=0 both hold? **Answer:** Generally no. Nonzero ε∂tE requires J=−ε∂tE in this manufactured family to satisfy Ampere–Maxwell.

### Exercise 5: Magnetic background versus gauge

If b₀ changes while b′ stays fixed, is every emf preserved? **Answer:** Fixed-path electric circulation is unchanged, but motional emf changes when u is nonzero. Changing physical B is not a gauge transformation.

### Exercise 6: Instantaneous value versus interval integral

Does final emf −2.32 µV times1 s equal the exact negative flux change? **Answer:** No. Emf varies with time and must be integrated, giving −1.84 µWb. Check that V·s=Wb.

## References and next connections

Compare §§10.1–10.3 of [MIT 8.02 Chapter 10](https://ocw.mit.edu/courses/8-02-physics-ii-electricity-and-magnetism-spring-2007/ce1720fd4b21def8c2189ff4779f27f7_cha10faraday_law.pdf). Next we examine how the source-free evolution laws connect initial E and B and propagate transverse pulses.
