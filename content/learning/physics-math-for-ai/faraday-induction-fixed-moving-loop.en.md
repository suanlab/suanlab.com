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
