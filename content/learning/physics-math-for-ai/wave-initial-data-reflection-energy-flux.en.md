## A wave is an initial-value problem

A snapshot of a string does not determine its future. The same shape can be stationary for an instant, moving left, moving right or accompanied by a separate initial kick. The wave equation needs both displacement and velocity. A boundary supplies additional information when a characteristic reaches an endpoint.

This lesson follows a localized smooth pulse. We calculate displacement, velocity, spatial slope, energy density and signed energy flux. The aim is to distinguish what a moving shape shows from what the material and its energy are doing. The previous periodic-chain lesson had no reflecting endpoint; here a half-line can have a fixed or free end.

## Medium, equation and physical scope

For constant line density $\mu>0$ in kg/m and tension $\tau>0$ in N,

$$
u_{tt}=c^2u_{xx},\qquad c=\sqrt{\tau/\mu}.
$$

Displacement $u$ is in m, velocity $u_t$ in m/s and slope $u_x$ is dimensionless. The model is a homogeneous, undamped, linear string with small transverse slopes. It excludes distributed forcing and time-dependent boundary actuation. Finite-interval multiple reflections, Robin conditions and changing media are also outside this particular experiment.

We use either the whole line $x\in\mathbb R$ or the half-line $x\ge0$. A finite plotting window is an observation window; it does not impose a second physical boundary. Waves can leave that window without reflecting.

## Why two initial fields are necessary

Specify

$$
u(x,0)=f(x),\qquad u_t(x,0)=g(x).
$$

If $g=0$, a localized displacement splits into two waves. If $g=-cf'$, the same displacement travels right. If $g=cf'$, it travels left. These are distinct physical preparations even though their displacement snapshots agree initially.

The extra factor $c$ gives $g$ the correct velocity units. A direction setting based only on which way a graph is animated cannot substitute for this initial condition. We will also add an independent compact velocity bump, so displacement and velocity need not share a traveling-wave relation.

## Derive the d’Alembert formula

The general whole-line solution can be written $u=R(x-ct)+L(x+ct)$. At $t=0$,

$$
R+L=f,\qquad -cR'+cL'=g.
$$

Let $H'=g$. One convenient choice is $R=f/2-H/(2c)$ and $L=f/2+H/(2c)$. An arbitrary constant shifts the two components oppositely and cancels in the physical sum. Thus

$$
u(x,t)=\frac{f(x-ct)+f(x+ct)}2+
\frac1{2c}\int_{x-ct}^{x+ct}g(s)\,ds.
$$

The integral is essential. Omitting it changes initial velocity to zero. At $t=0$ its interval collapses; differentiating its moving endpoints recovers $g(x)$ rather than losing the velocity information.

## Velocity and slope are different measurements

Write $\ell=x-ct$ and $r=x+ct$. Direct differentiation gives

$$
u_t=\frac c2[f'(r)-f'(\ell)]+\frac12[g(r)+g(\ell)],
$$

$$
u_x=\frac12[f'(\ell)+f'(r)]+\frac1{2c}[g(r)-g(\ell)].
$$

The first is actual transverse material velocity; the second describes shape. A pulse traveling right has $u_t=-cu_x$, not a constant material velocity equal to $c$. The speed $c$ is the horizontal propagation speed of information and shape. A marked material point remains at its equilibrium horizontal coordinate and moves transversely.

## A compact profile with controlled smoothness

Use

$$
b(z)=\begin{cases}(1-z^2)^3,&|z|<1,\\0,&|z|\ge1,\end{cases}
\qquad f(x)=A\,b\!\left(\frac{x-x_0}{w}\right).
$$

Here $A$ is a signed displacement amplitude in m, $x_0$ the center and $w>0$ the half-width in m. The profile and its first two derivatives match zero at both support edges. Inside the support,

$$
b'(z)=-6z(1-z^2)^2,\qquad
b''(z)=-6+36z^2-30z^4.
$$

Its full integral is $32/35$. The exact compact support makes finite propagation testable: a zero outside the dependence region is a physical zero, rather than an arbitrary cutoff imposed on a Gaussian tail. The profile is $C^2$; one should not assume arbitrarily high smoothness at its edges.

## Prepare split, traveling and kicked pulses

The experiment uses

$$
g(x)=d\,cf'(x)+B\,b\!\left(\frac{x-x_0}{w}\right),
\qquad d\in\{-1,0,1\}.
$$

$B$ is an independent velocity amplitude in m/s. With $B=0$, $d=-1$ is a right-moving pulse, $d=1$ a left-moving pulse and $d=0$ a displacement split. When $B\ne0$, the direction name identifies the displacement-related part of the preparation, not a guarantee that all energy moves in that direction.

For a half-line, require $x_0>w$ so both initial fields vanish near the endpoint. The calculator leaves an additional0.25 m gap. This ensures the fixed/free compatibility conditions without repairing the data after initialization.

## Finite propagation and a surprising plateau

The value at $(x,t)$ depends on initial information in $[x-ct,x+ct]$. This interval is a dependence region, not a claim that every point inside contributes equally. Outside the initial support enlarged by $ct$, both displacement contributions and the velocity integral vanish.

An instructive exception to “a visible offset means energy is here” occurs with $A=0$, $B\ne0$ on the whole line. Once the integration interval fully contains the velocity bump,

$$
u=\frac{Bw}{2c}\frac{32}{35},\qquad u_t=u_x=0
$$

in the interior plateau. The string has a nonzero static offset there, but no local kinetic or strain energy. Energy moves in the two fronts. For $B=0.2$ m/s, $w=1$ m and $c=1$ m/s, the plateau is about0.09143 m.

## Reflecting endpoints require extending both data

On the half-line, a fixed end satisfies $u(0,t)=0$. Extend $f$ and $g$ oddly across0. A free end satisfies $u_x(0,t)=0$; extend both evenly. Then apply the whole-line formula and restrict the answer to $x\ge0$.

The derivative of an odd function is even, and the derivative of an even function is odd. Consequently, even if $g=cf'$ holds on the positive half-line, it is generally wrong to extend $f$ first and define the global $g$ as $c$ times its derivative. That would give the wrong parity for initial velocity. Extend the independently specified initial fields with the same required parity.

The reflected image is a mathematical device outside the physical domain. Its energy must not be counted as additional physical energy on the half-line.

## Reflection sign and arrival times

For an initially left-moving compact pulse with $B=0$, the half-line solution can be written

$$
u(x,t)=f(x+ct)+\rho f(ct-x),\qquad x\ge0,
$$

where $\rho=-1$ for a fixed end and $\rho=+1$ for a free end. The center reaches the boundary at $x_0/c$; the leading edge arrives earlier, at $(x_0-w)/c$. Reflection is a process over the finite pulse width, not an instantaneous change of the whole profile.

For $A=0.1$ m, $x_0=2$ m, $w=1$ m and $c=1$ m/s, the reflected center is back at $x=2$ at $t=4$ s. Its displacement is−0.1 m at a fixed end and+0.1 m at a free end. Both reflected pulses carry energy rightward. Their displacement signs do not determine the direction of energy flow.

## Derive physical energy and local conservation

The energy per unit equilibrium length and flux are

$$
e=\frac12\mu u_t^2+\frac12\tau u_x^2,
\qquad J=-\tau u_tu_x.
$$

$e$ has units J/m and $J$ has units W. Multiply the PDE by $\mu u_t$ and use $\mu c^2=\tau$:

$$
\partial_t e
=\tau(u_tu_{xx}+u_xu_{xt})
=\partial_x(\tau u_tu_x),
\qquad e_t+J_x=0.
$$

This is a local conservation law. It identifies how energy crosses an interval boundary; it does not merely state that a plotted total stays constant. At a fixed end $u_t(0,t)=0$, and at a free end $u_x(0,t)=0$, so either ideal stationary boundary has zero power flow.

## Characteristic energies and signed flux

Define nonnegative right- and left-moving energy densities

$$
e_R=\frac\mu4(u_t-cu_x)^2,\qquad
e_L=\frac\mu4(u_t+cu_x)^2.
$$

Then $e=e_R+e_L$ and $J=c(e_R-e_L)$. In a pure right-moving wave $J=ce$; in a pure left-moving wave $J=-ce$. In an overlap, the net flux can vanish even when energy is nonzero because two equal directional contributions cancel.

Displacement components $R,L$ have an arbitrary opposite constant offset when a primitive of $g$ is used. Directional energy is defined by derivatives and is unaffected by that choice. A display should not infer energy by squaring the component displacement alone.

## Finite-window balance and independent integration

For a fixed observation interval $[a,b]$,

$$
E_{[a,b]}(t)=\int_a^b e(x,t)\,dx,
$$

$$
E_{[a,b]}(t)-E_{[a,b]}(0)
=\int_0^t[J(a,s)-J(b,s)]\,ds.
$$

Positive $J(a)$ brings energy in from the left; positive $J(b)$ takes it out to the right. Each quantity is a signed integral. The calculator integrates spatial energy and boundary power separately. It does not define the flux integral as the endpoint energy difference and thereby force the residual to zero.

For this compact polynomial family, support edges split space and time into polynomial pieces. Energy and flux on each piece have degree at most12. Eight-point Gauss–Legendre quadrature integrates through degree15, so it is exact in real arithmetic after all breakpoints are included. Floating-point roundoff remains; a tiny residual is reported rather than hidden. Plotting samples are separate from these integrals.

## Worked energy-flow experiment and limits

Start with the right-moving pulse $A=0.1$, $B=0$, $w=1$, $x_0=2$, $\mu=\tau=1$. Observe $[0,4]$. Initially the entire support $[1,3]$ lies inside. At $t=4$, the support is $[5,7]$, so window energy is0. The signed right-boundary integral equals the initial energy; left-boundary transfer is0. The whole-line energy remains constant.

For a pure traveling pulse, $E=\tau A^2\int_{-1}^1[b'(z)]^2dz/w$. Thus narrowing a fixed-amplitude pulse increases its energy and slope. The initial maximum slope is $96|A|/(25\sqrt5\,w)$. This is an initial indicator only: an added velocity field and later overlap can change slopes. The supported amplitudes, widths and times are numerical teaching bounds, not a proof that every setting stays in the physical small-slope regime.

## Research extension: a passive endpoint can store and release wave energy

Fixed and free stationary ends reflect without storing boundary energy. Now consider a different, explicitly supplementary model: a massless endpoint attached to a spring and a viscous damper. The string remains linear, but the boundary has memory through the spring displacement. This does not add Robin or damping controls to the native fixed/free calculator.

### Derive the force sign at the left endpoint



$$
u_{tt}=c^2u_{xx},\quad x\ge0,\quad Z=\frac\tau c=\mu c>0,\quad Y(t)=u(0,t),\qquad
\tau u_x(0,t)=K_bY(t)+bY^{\prime}(t),\quad K_b\ge0,\quad b\ge0.
$$

The string pulls the endpoint with transverse force tau ux; the spring and damper oppose displacement and velocity. Their force balance gives the displayed sign. Z and b have units N s/m and Kb has units N/m. Here scalar b is boundary damping, distinct from the earlier dimensionless compact-profile function. No endpoint mass or active boundary actuator is included.



$$
E_s=\int_0^\infty\left(\frac\mu2u_t^2+\frac\tau2u_x^2\right)dx,\quad E_b=\frac{K_bY^2}{2},\qquad
\frac d{dt}(E_s+E_b)=-b(Y^{\prime})^2\le0.
$$

Assume finite-energy compatible data with no power flux at infinity at the finite times considered. Integrate the existing local conservation law: dEs/dt=J(0)=−tau Yprime ux(0). Substituting the force balance separates spring storage from heat. The string alone may gain energy when the spring releases it; passivity concerns the combined budget.

### Calculate the harmonic reflection and absorbed fraction

With exp(+i Omega t), an incoming left-moving amplitude I multiplies exp(+i Omega x/c) and an outgoing amplitude R multiplies exp(−i Omega x/c). At x=0 insert both displacement and slope into the boundary equation:



$$
r(\Omega)=\frac RI=\frac{-K_b+i\Omega(Z-b)}{K_b+i\Omega(Z+b)},\quad \Omega>0,\qquad
|r|^2=\frac{K_b^2+\Omega^2(Z-b)^2}{K_b^2+\Omega^2(Z+b)^2},\qquad
1-|r|^2=\frac{4\Omega^2Zb}{K_b^2+\Omega^2(Z+b)^2}\in[0,1].
$$

The denominator is nonzero under these positive-frequency passive assumptions. A free end Kb=b=0 gives r=+1; the large-stiffness limit at fixed frequency gives r=−1. Exact broadband matching requires Kb=0 and b=Z. Setting b=Z with a finite positive spring leaves frequency-dependent reflection. DC has no propagating cycle power and should not be assigned a measured absorbed fraction by a 0/0 calculation.



$$
P_{\rm in}=\frac{Z\Omega^2|I|^2}{2},\quad P_{\rm out}=\frac{Z\Omega^2|R|^2}{2},\quad
P_{\rm heat}=\frac{b\Omega^2|I+R|^2}{2},\qquad P_{\rm in}-P_{\rm out}=P_{\rm heat}.
$$

These are time-averaged powers. The spring has zero mean storage rate in a periodic state, but it need not have zero instantaneous storage rate during a pulse.

### Derive the causal endpoint memory from characteristics

Write u(x,t)=F(t+x/c)+G(t−x/c), where F is incoming displacement evaluated at the endpoint and G is reflected displacement. These are not the original spatial initial-velocity field. Start before pulse arrival, with F(0)=Y(0)=0. A compact incoming trace can be prepared as an incoming initial pulse away from the endpoint; it is not an active boundary force.



$$
Y=F+G,\quad (Z+b)Y^{\prime}+K_bY=2ZF^{\prime},\qquad
Y(t)=\frac{2Z}{Z+b}\int_0^t e^{-K_b(t-s)/(Z+b)}F^{\prime}(s)\,ds,\quad G=Y-F.
$$

Use tau ux=Z(Fprime−Gprime) and Yprime=Fprime+Gprime to obtain the first-order equation. Its exponential kernel depends only on past inputs. At Kb=0, integration gives Y=2ZF/(Z+b), reproducing free or matched reflection when the corresponding damping is selected.



$$
P_{\rm in}(t)=Z(F^{\prime})^2,\quad P_{\rm out}(t)=Z(G^{\prime})^2,\qquad
P_{\rm in}-P_{\rm out}=K_bYY^{\prime}+b(Y^{\prime})^2=\frac{dE_b}{dt}+b(Y^{\prime})^2.
$$

The two powers are computed from separate characteristic rates. Their difference is not defined by the stored-energy change. This instantaneous identity allows output power after input has ended, provided previously stored spring energy supplies it.

### Integrate the reflected tail after a compact pulse



$$
K_b>0,\quad F(t)=0\ (t\ge T),\quad \lambda=\frac{K_b}{Z+b}:\qquad
Y(t)=Y(T)e^{-\lambda(t-T)},\quad G(t)=Y(t)\quad(t\ge T).
$$

The reflected field has an exponentially decaying time tail even though the incoming displacement is compact. At the endpoint it releases stored energy; propagation through the string still follows the continuum characteristics at speed c. Integrating the outgoing and heat powers from T to infinity gives



$$
E_{\rm rad,tail}=\int_T^\infty Z(G^{\prime})^2dt=\frac Z{Z+b}E_b(T),\qquad
Q_{\rm tail}=\int_T^\infty b(Y^{\prime})^2dt=\frac b{Z+b}E_b(T),\qquad E_{\rm rad,tail}+Q_{\rm tail}=E_b(T).
$$

For b=0 all remaining spring energy returns to the string. Increasing b changes both the split and the state Y(T) created by the earlier pulse; do not assume the stored energy is identical across experiments.

The table fixes Z=1 N s/m, Kb=1 N/m and the incoming displacement F(t)=0.1[1−(t−2)²]³ m for 1<t<3 in SI seconds, zero otherwise. Thus T=3 s. Values are rounded; the code uses composite Simpson quadrature for the causal convolution, followed by the exact tail formulas.

| b (N s/m) | Y(T) (m) | stored E_b(T) (J) | radiated tail (J) | tail heat (J) |
|---|---|---|---|---|
| 0 | -0.0710926056 | 0.00252707929 | 0.00252707929 | 0 |
| 0.5 | -0.0427660489 | 0.000914467470 | 0.000609644980 | 0.000304822490 |
| 1 | -0.0281144098 | 0.000395210019 | 0.000197605010 | 0.000197605010 |
| 2 | -0.0146481885 | 0.000107284713 | 3.57615711e-5 | 7.15231423e-5 |

```python
from math import exp

# Z=1 N s/m, Kb=1 N/m. Incoming displacement: 0.1*(1-(t-2)^2)^3 m on (1,3) s.
def incoming_rate(t):
    z = t-2
    return -0.6*z*(1-z*z)**2 if abs(z) < 1 else 0

def integrate(f, a, b, n=20000):
    h = (b-a)/n
    return h/3*(f(a)+f(b)+sum(
        (4 if j % 2 else 2)*f(a+j*h) for j in range(1, n)))

for damping in [0, .5, 1, 2]:
    y = 2/(1+damping)*integrate(
        lambda t: exp(-(3-t)/(1+damping))*incoming_rate(t), 1, 3)
    stored = y*y/2
    radiation = stored/(1+damping)
    heat = damping*stored/(1+damping)
    print(damping, *(format(x, '.9g') for x in [y, stored, radiation, heat]))
    assert abs(radiation+heat-stored) < 1e-15
```

![A compact incoming pulse leaves a reflected time tail at a passive spring-damper endpoint; stored energy divides into outgoing radiation and heat.](/assets/learning/waves-front-boundary-memory.en.svg)

### Research questions with explanations

**Is b=Z always a perfectly absorbing endpoint?** **Answer:** Only when Kb=0 in this model. A positive spring contributes a frequency-dependent reactive load and leaves reflection, especially at low frequency.

**Can a passive endpoint emit energy after the incoming pulse ends?** **Answer:** Yes. The spring releases previously stored energy. Include boundary storage and dissipation before interpreting a string-only energy increase as active generation.

**Does the reflected time tail violate finite propagation in the PDE?** **Answer:** No. A local causal boundary ODE produces the tail, and each emitted disturbance travels at c. This is different from the infinite lattice’s spatial support question. Validate a learned boundary model against its own memory and energy budget.

## Exercises with explanations

### Exercise 1: A shape does not set a direction

Two strings have the same $f$. One has $g=0$ and one $g=-cf'$. What differs?

**Solution:** The first splits into two half-displacement waves; the second travels right without that split. The initial velocity field supplies the missing information.

### Exercise 2: Velocity-only plateau

Find the interior displacement for $A=0$, $B=0.2$, $w=1$, $c=1$ once the integration interval contains the whole bump.

**Solution:** It is $Bw(32/35)/(2c)\approx0.09143$ m. In the plateau $u_t=u_x=0$, so local energy is0 even though displacement is nonzero.

### Exercise 3: The parity trap

Why is differentiating the odd extension of $f$ insufficient to extend left-moving data at a fixed end?

**Solution:** That derivative is even, but fixed-end initial velocity must be extended oddly. Extend the original $g=cf'$ from the positive half-line independently, with odd parity.

### Exercise 4: Reflected sign and energy

A fixed end returns a negative displacement pulse. Is its flux negative?

**Solution:** After reflection it travels right, so $J=ce\ge0$. Changing amplitude sign changes both velocity and slope signs and leaves their product, hence energy-flow direction, unchanged.

### Exercise 5: Energy leaving a window

The energy in $[a,b]$ falls by0.03 J, with integrated left inflow0.01 J. What is the integrated right outflow?

**Solution:** $-0.03=0.01-I_b$, so $I_b=0.04$ J. Global conservation permits a local decrease and requires both boundary terms.

### Exercise 6: Quadrature at a support edge

Why split an integration interval at the moving support edges before applying a polynomial quadrature rule?

**Solution:** The integrand is polynomial on each piece, not one polynomial across an edge. The rule's exact-degree guarantee applies to the pieces. Ignoring the edges can produce a plausible but inaccurate conservation residual.

## AI verification questions and primary references

A learned wave surrogate should recover the two independent initial fields, respect finite propagation, distinguish a physical reflecting boundary from a cropped plot, and balance local energy with signed boundary flux. Check a plateau case and a reflection with nonzero independent velocity as well as a visually simple traveling pulse. A low average displacement error alone does not verify these properties.

[Herman's d’Alembert notes, sections1–3.2](https://people.uncw.edu/hermanr/pde1/dAlembert/dAlembert.htm) and [Toronto's half-line wave notes](https://www.math.utoronto.ca/courses/mat394h1/20139/L7.html) provide the initial-value and boundary framework. The compact polynomial family, SI characteristic energies, independently integrated observation-window budget and explicit calculator tests are derived here from the stated model. They do not complete the separate topics of diffraction, general dispersive propagation or all boundary-value problems.
