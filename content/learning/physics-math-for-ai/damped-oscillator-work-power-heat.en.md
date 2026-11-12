## A slowing oscillator has not lost the energy accounting

A spring-mounted cart released from displacement oscillates less and less when a viscous resistance acts. Its mechanical energy decreases. That decrease is not a failure of accounting: the model transfers energy into a dissipated heat quantity. If an external actuator also pushes, input work must be included with its sign.

This lesson solves the actual time-dependent initial-value problem, rather than only drawing an energy-level set. Mass and spring stiffness are positive constants, damping is nonnegative, and the external drive is constant. We distinguish stored mechanical energy, drive work and integrated heat. The model specifies no temperature, heat capacity, thermal reservoir or periodic forcing.

## State which interactions store and transfer energy

Let x be displacement from the spring's undeformed position and v its velocity. With stiffness k, damping coefficient c and constant drive F,

$$
m\ddot x+c\dot x+kx=F,\qquad x(0)=x_0,\quad \dot x(0)=v_0.
$$

The spring's potential energy is V=kx²/2 and kinetic energy is K=mv²/2. We keep the constant drive outside the stored mechanical energy, so its work appears explicitly. One could instead include −Fx in a different potential convention, but then the corresponding work term must be removed to avoid double counting.

The viscous force is −cv. Its coefficient has units kg/s, not the units of spring stiffness N/m. Its power is −cv², nonpositive for every velocity sign. Resistance opposes velocity even when the cart moves left.

## Shift to equilibrium before solving the motion

The static equilibrium is xe=F/k. Set y=x−xe, y0=x0−xe and α=c/(2m). The equation becomes

$$
\ddot y+2\alpha\dot y+\omega_0^2y=0,\qquad \omega_0^2=k/m.
$$

The discriminant c²−4mk identifies the regimes. A constant drive shifts equilibrium without introducing a driving frequency. Calling it a resonant periodic experiment would therefore be incorrect. Changing the drive also changes y0 for fixed physical initial position; initial data must be transformed along with the coordinate shift.

## Underdamped motion retains oscillation

When c²<4mk, let ω=√(ω0²−α²). The initial-value solution is

$$
y(t)=e^{-\alpha t}\left[y_0\cos(\omega t)+\frac{v_0+\alpha y_0}{\omega}\sin(\omega t)\right],
$$

$$
v(t)=e^{-\alpha t}\left[v_0\cos(\omega t)-\frac{\alpha v_0+\omega_0^2y_0}{\omega}\sin(\omega t)\right].
$$

With c>0 the envelope decays. With c=0 the same branch reduces to undamped harmonic motion and dissipated heat is exactly zero. The interface names that limit separately. Mechanical energy may vary under a constant drive even in the zero-damping case because the actuator can perform work.

## Critical damping requires the repeated-root solution

At c²=4mk, the two characteristic roots coincide. Distinct-root expressions cannot be evaluated by dividing by their now-zero difference. Use

$$
y(t)=e^{-\alpha t}[y_0+(v_0+\alpha y_0)t],
$$

$$
v(t)=e^{-\alpha t}[v_0-\alpha(v_0+\alpha y_0)t].
$$

The t multiplying the exponential is essential. Both specified initial conditions are retained. “Critical” identifies a root structure, rather than a blanket claim about the fastest response for arbitrary initial data and arbitrary response metrics.

## Overdamped motion has two real decay rates

For c²>4mk, define roots rslow and rfast. Numerically, the slow root is evaluated without subtracting two nearly equal positive numbers:

$$
r_{\mathrm{fast}}=-\alpha-\sqrt{\alpha^2-\omega_0^2},\qquad
r_{\mathrm{slow}}=-\frac{\omega_0^2}{\alpha+\sqrt{\alpha^2-\omega_0^2}}.
$$

Then y=Cs exp(rslow t)+Cf exp(rfast t), where the two coefficients solve Cs+Cf=y0 and rslow Cs+rfast Cf=v0. Velocity is obtained by differentiating these exponentials. Both roots are negative under the stated positive-mass, positive-stiffness and positive-damping assumptions. Depending on initial data, sums of decays can still cross a position or reverse direction; “overdamped” does not itself specify every trajectory shape.

## Derive the instantaneous power balance

Multiply the original motion law by v:

$$
\frac{d}{dt}\left(\frac12mv^2+\frac12kx^2\right)=Fv-cv^2.
$$

Define mechanical energy E=K+V, drive power Pdrive=Fv and damping power Pdamp=−cv². Their sum is the rate of change of stored mechanical energy. Drive power can be positive or negative. Damping power cannot be positive in this model, while the associated heat generation rate is +cv².

A force value alone does not give power: the current velocity is also needed. An instant at rest has zero power even if the spring and drive forces are nonzero and acceleration is about to change the velocity.

## Integrate drive work and dissipated heat separately

Because the drive is constant, its accumulated work is F[x(t)−x0]. Heat is computed from the independent integral

$$
Q(t)=\int_0^t c\,v(s)^2ds\ge0,
\qquad E(t)-E(0)+Q(t)=W_{\mathrm{drive}}(t).
$$

The implementation does not define Q as whatever makes the residual vanish. It integrates the analytic squared-velocity expression. In the oscillatory branch this uses exponential sine/cosine integrals. At the repeated root it uses moments of an exponential times a quadratic polynomial. In the real-root branch it integrates the three exponential terms in the squared sum.

Short-time exponential moments use a convergent series to avoid subtracting nearly equal numbers. All transcendental formulas are evaluated in floating point. Their residual is a numerical diagnostic, not an exact-rational proof. An independent numerical integration of the differential equation and heat rate checks both motion and heat.

## Read the position, energy and power panels

Every horizontal axis is time in seconds. The position panel shows the actual analytic motion in meters. The energy panel compares E, Q and E0+Wdrive in joules; the sum E+Q should match the latter. The power panel shows drive power, negative damping power and their sum in watts.

The fixed ten-second horizon is an illustration of the same initial-value solution. Query time selects current readouts without changing that horizon. Try zero damping, exact critical damping, strong overdamping, a state already at driven equilibrium and a drive opposing an initially positive velocity. Distinguish a negative work value from a negative heat value: only the former is possible under these assumptions.

## Research extension: build an exponential-decay certificate and a force-error bound

A nonincreasing energy is useful, but a convergence rate needs another argument. At an instant with nonzero displacement and zero velocity, the energy is positive while its derivative is zero. Thus H'=−cv² alone cannot imply H'≤−gamma H with a positive constant for every state. We will add a cross term that lets the restoring force participate in the estimate.

### Choose the energy relative to driven equilibrium

Assume m,k,c>0 and a constant drive F. Shift the coordinate by its equilibrium as in the original solution:

$$
y=x-\frac Fk,\quad H=\frac12mv^2+\frac12ky^2
=E_{\mathrm{mech}}-Fx+\frac{F^2}{2k},\qquad
\dot H=-cv^2.
$$

H measures displacement from equilibrium and vanishes there. It differs from the native stored spring-plus-kinetic energy E_mech, which can increase when the drive does work. This analysis does not change that lab energy convention or the independent heat calculation. Positive damping is required here: the native c=0 branch remains valid, but does not have the contraction proved below.

### Add a term that sees a displaced particle at rest

Choose epsilon with units 1/s and differentiate the following expression:

$$
0<\varepsilon<\frac cm,\qquad
L=H+\varepsilon m yv+\frac{\varepsilon c}{2}y^2,
\qquad
\dot L=-(c-\varepsilon m)v^2-\varepsilon k y^2=-D.
$$

To check cancellation, differentiate epsilon m yv: it contributes epsilon m v²−epsilon c yv−epsilon k y². The derivative of epsilon c y²/2 cancels the mixed term. Both coefficients remaining in D are positive. Unlike H', this derivative is strictly negative at any nonzero state, including a displaced particle at rest.

We must also prove L is positive: a decreasing expression is not automatically a bound on the state. In energy-scaled coordinates:

$$
\mathbf z=\begin{pmatrix}\sqrt{k}\,y\\\sqrt{m}\,v\end{pmatrix},\quad
L=\frac12\mathbf z^TS\mathbf z,\quad
S=\begin{pmatrix}1+\varepsilon c/k&\varepsilon\sqrt{m/k}\\
\varepsilon\sqrt{m/k}&1\end{pmatrix},\quad
\det S=1+\frac{\varepsilon(c-\varepsilon m)}{k}>0.
$$

Both coordinates have units square root of a joule and S is dimensionless. Its first diagonal entry and determinant are positive, so the symmetric matrix is positive definite. Let its positive eigenvalues be lambda_min and lambda_max. Rayleigh-quotient bounds give:

$$
\lambda_{\min}H\le L\le\lambda_{\max}H,\quad
d=\min\left(\varepsilon,\frac cm-\varepsilon\right)>0,
\quad D\ge2dH\ge\frac{2d}{\lambda_{\max}}L,
\quad \gamma=\frac{2d}{\lambda_{\max}}.
$$

D contains the same squared coordinates, with positive weights c/m−epsilon and epsilon. Integrating the differential inequality for L, or differentiating exp(gamma t)L, now proves:

$$
L(t)\le e^{-\gamma t}L(0),\qquad
H(t)\le\frac{\lambda_{\max}}{\lambda_{\min}}e^{-\gamma t}H(0).
$$

The prefactor lambda_max/lambda_min matters. Dropping it would incorrectly assert a strict exponential decrease of H from every initial state, including the instant-at-rest counterexample. These are sufficient bounds, not necessarily the sharp decay rate. Zero initial H gives the stationary equilibrium solution.

### Work the constants for the critical oscillator

A convenient general choice is epsilon=c/(2m). For m=k=1 and c=2, S has entries [[3,1],[1,1]], eigenvalues 2±sqrt(2), and gamma=2−sqrt(2) per second. The energy prefactor is 3+2sqrt(2), about 5.828427.

For the same masses and stiffness, the following choices show the constants of this particular certificate:

| c (kg/s) | epsilon (1/s) | gamma (1/s) | energy prefactor |
|---|---|---|---|
| 1 | 0.5 | 0.552786 | 2.618034 |
| 2 | 1 | 0.585786 | 5.828427 |
| 10 | 5 | 0.194193 | 101.990195 |

The rows are guarantees, not a proof of an optimal damping choice or an ordering of every exact trajectory. At critical damping, F=0,y0=1,v0=0 gives y=(1+t)exp(−t), v=−t exp(−t). Substitution lets us compare an exact energy curve with the bound without numerically integrating the ODE.

### Let a known force budget drive a trajectory error

Compare two exact flows with the same m,k,c, but drives differing by r(t). Their position difference e satisfies:

$$
e=\widehat x-x,\qquad m\ddot e+c\dot e+ke=r(t),\qquad
\dot L_e=-D_e+r(\dot e+\varepsilon e).
$$

Here L_e and D_e use e and its velocity in place of y,v. H_e is a trajectory-error metric in joules; it is not an extra physical energy stored in the apparatus. Initial-state mismatch stays in L_e(0).

Weighted Cauchy–Schwarz bounds (e'+epsilon e)² by D_e times the coefficient C_r. Then Young's inequality bounds the product with r:

$$
C_r=\frac1{c-\varepsilon m}+\frac\varepsilon k,
\quad (\dot e+\varepsilon e)^2\le C_rD_e,
\quad r(\dot e+\varepsilon e)\le\frac12D_e+\frac12C_rr^2,
\quad \dot L_e\le-\eta L_e+\frac12C_rr^2,
\quad \eta=\frac d{\lambda_{\max}}.
$$

For example, square the coefficients 1 and epsilon against weights c−epsilon m and epsilon k; their inverse-weight sum is exactly C_r. C_r has units s/kg, so C_r r² is power. Multiplication by exp(eta t) and integration gives a deterministic input estimate:

$$
L_e(t)\le e^{-\eta t}L_e(0)+\frac{C_r}{2}
\int_0^t e^{-\eta(t-s)}r(s)^2\,ds.
$$

If |r(s)|≤rho throughout the interval, integrate the exponential kernel and project from the error energy to position:

$$
L_e(t)\le e^{-\eta t}L_e(0)
+\frac{C_r\rho^2}{2\eta}(1-e^{-\eta t}),\qquad
|e(t)|\le\sqrt{\frac{2L_e(t)}{k\lambda_{\min}}}.
$$

The transient decays and the guarantee approaches a finite upper bound. A varying bounded input need not converge to a constant error. The estimate requires a justified continuous-time force budget; force samples alone do not supply it. For computed discrete trajectories, discretization error is a separate contribution.

### Compare the bound with a known constant-error solution

Take m=k=1,c=2,r=0.1 N and identical zero initial states. The exact position error is 0.1[1−(1+t)exp(−t)] m, tending to 0.1 m. For this certificate eta=1/(2+sqrt(2)), C_r=2 s/kg, and the asymptotic position upper bound is about 0.341421 m. The gap quantifies conservatism, rather than a failure of the inequality.

![For a critical oscillator the exact energy lies below an exponential bound with a prefactor; a constant force mismatch produces a bounded position error below a conservative input bound.](/assets/learning/classical-damped-certificate.en.svg)

The upper panel uses a logarithmic energy-ratio axis for the unforced displaced oscillator. The lower panel is position error in metres for the separate constant-mismatch example, with an exact 0.1 m limit. Both time axes use seconds. These calculations are standalone; the existing lab continues to display its declared physical energy, work, heat and powers.

### Reproduce the algebraic constants and exact examples

The code checks the positive quadratic bounds, derivative estimates and critical closed solutions. Floating-point state samples illustrate the all-state algebraic proof; they do not establish it by sampling.

```python
from math import sqrt, hypot, exp, expm1

def certificate(m,k,c):
    assert m>0 and k>0 and c>0
    epsilon = c/(2*m)
    s11 = 1+epsilon*c/k
    off = epsilon*sqrt(m/k)
    largest = (s11+1+hypot(s11-1,2*off))/2
    determinant = 1+epsilon*(c-epsilon*m)/k
    smallest = determinant/largest  # avoids subtracting close eigenvalues
    d = min(epsilon,c/m-epsilon)
    gamma, eta = 2*d/largest,d/largest
    Cr = 1/(c-epsilon*m)+epsilon/k
    return epsilon,smallest,largest,gamma,eta,Cr

for m,k,c in [(1.,1.,1.),(1.,1.,2.),(1.,1.,10.),(4.,3.,2.)]:
    epsilon,lo,hi,gamma,eta,Cr = certificate(m,k,c)
    for y,v in [(1.,0.),(.3,-.5),(-2.,3.)]:
        H = (m*v*v+k*y*y)/2
        L = H+epsilon*m*y*v+epsilon*c*y*y/2
        D = (c-epsilon*m)*v*v+epsilon*k*y*y
        assert lo*H-1e-12 <= L <= hi*H+1e-12
        assert D+1e-12 >= gamma*L
        for r in [-.2,0.,.3]:
            Ldot = -D+r*(v+epsilon*y)
            assert Ldot <= -eta*L+Cr*r*r/2+1e-12
    print("m,k,c,gamma,prefactor:",m,k,c,
          round(gamma,6),round(hi/lo,6))

# Independent critical closed solutions; m=k=1,c=2.
epsilon,lo,hi,gamma,eta,Cr = certificate(1.,1.,2.)
rho = .1  # constant force mismatch in N, identical zero initial states
for t in [0.,.1,.5,1.,2.,5.,10.]:
    y,v = (1+t)*exp(-t),-t*exp(-t)  # unforced, y0=1,v0=0
    H = (y*y+v*v)/2
    L = H+y*v+y*y
    assert L <= 1.5*exp(-gamma*t)+1e-12
    assert H <= (hi/lo)*exp(-gamma*t)*.5+1e-12
    error = rho*(1-(1+t)*exp(-t))
    error_v = rho*t*exp(-t)
    Le = (error_v**2+error**2)/2+error*error_v+error**2
    bound = Cr*rho*rho/(2*eta)*(-expm1(-eta*t))
    assert Le <= bound+1e-12
    assert abs(error) <= sqrt(2*bound/lo)+1e-12
print("exact position-error limit:",rho)
print("certified position-error limsup:",sqrt(Cr*rho*rho/(eta*lo)))
```

### Worked research questions

1. **Why is H'≤0 insufficient for the displayed exponential rate?** At y=1,v=0, H=k/2>0 and H'=0. A negative multiple of H would already be strictly negative. The auxiliary L supplies the missing displacement term and norm comparison.
2. **Can the same error-floor proof be used at c=0?** Epsilon must lie strictly between 0 and c/m. That interval is empty at zero damping. The undamped branch needs a different input analysis; bounded resonant forcing can accumulate response.
3. **What should a learned-force evaluation report?** Specify the common plant coefficients, initial-state mismatch, the full-interval force budget, the norm/prefactor/rate, and which plotted quantity is physical stored energy versus error metric. Check trajectory predictions as well as the energy budget.

This certificate follows directly from the linear equation and positive quadratic forms. It supplies explicit sufficient rates without changing the model's heat law or claiming empirical validation of its damping coefficient.

## Six exercises with worked solutions

1. **Exercise: undamped limit.** Use m=k=1,c=0,F=0,x0=1,v0=0. **Solution:** x=cos t and v=−sin t, so E=1/2J at every time and Q=Wdrive=0. Position changes while total mechanical energy stays fixed.
2. **Exercise: classify roots.** Keep m=k=1 and choose c=1,2,3. **Solution:** discriminants are−3,0,5, giving underdamped, critical and overdamped motion respectively. The critical case uses the repeated-root formula.
3. **Exercise: critical step drive.** Set m=k=1,c=2,F=1,x0=v0=0. At1s? **Solution:** x=1−2/e and v=1/e. Drive work equals x≈0.264241J; stored energy is about0.102579J.
4. **Exercise: heat for that step.** Integrate the heat rate directly. **Solution:** v=t exp(−t), so Q(1)=2∫0¹t²exp(−2t)dt=(1−5exp(−2))/2≈0.161662J. Adding it to stored energy reproduces the work in exercise3.
5. **Exercise: already at equilibrium.** Use F=k=1,x0=1,v0=0 and any allowed damping. **Solution:** y0=0 and v0=0 give x=1 permanently. Stored spring energy is1/2J, but drive work, heat and both powers are zero. A nonzero drive need not continually inject energy.
6. **Exercise: signed braking input.** Use m=k=c=1,F=−1,x0=0,v0=2 and query0.1s. **Solution:** velocity remains positive initially, so drive power and accumulated drive work are negative. Heat is positive because c v² is positive. The mechanical energy decreases by the magnitude of removed drive work plus generated heat.

## What an energy check can and cannot certify

A learned model can hide trajectory errors by defining an unobserved dissipation term from the residual. Integrating a declared nonnegative heat rate independently is a stronger check. Even then, an accurate balance alone does not identify the correct friction or drive law. Compare measured trajectories, inferred forces and the energy budget under stated assumptions. Real thermal modeling needs additional variables and constitutive relations beyond the cumulative Q used here.

## Background and scope

For the characteristic-root framework, see [MIT OCW, Damped Harmonic Oscillators](https://ocw.mit.edu/courses/18-03sc-differential-equations-fall-2011/pages/unit-ii-second-order-constant-coefficient-linear-equations/damped-harmonic-oscillators/). The constant-drive shift, signed work and independent heat calculations above follow from the declared model. Periodic resonance and statistical thermal reservoirs are developed separately.
