## Different physical systems can share a normalized problem

A heavy spring system and a light one need not move on the same physical clock. Their displacements need not have the same scale either. Nevertheless, appropriately normalized trajectories can solve the same differential equation with the same initial data. This is physical similarity under a declared model, rather than a claim that the two apparatuses are identical.

The parent model is a mass, linear spring and linear damper with no external forcing. Mass and stiffness are positive, damping is nonnegative, and displacement is measured from equilibrium. Its equation is

$$
m x''+b x'+kx=0.
$$

The initial displacement and velocity remain separate pieces of state information. Matching the normalized differential law alone will not make arbitrary initial states follow the same trajectory.

## Choose scales without dividing by a possibly zero initial value

Define the natural rate omega0=sqrt(k/m). Select an independently positive reference displacement X. It can be a declared instrument or design scale and need not equal the initial displacement, which might be zero or negative.

Set normalized time s=omega0*t and normalized displacement u=x/X. Chain-rule differentiation gives x'=X*omega0*u_s and x''=X*omega0 squared*u_ss. Substituting and dividing by kX produces

$$
u_{ss}+2\zeta u_s+u=0,\qquad
\zeta=\frac{b}{2\sqrt{mk}}.
$$

The damping ratio is dimensionless: b has units kg/s, while sqrt(m*k) has the same units. The time coordinate s is also dimensionless. These are derived scale relations, not the result of dropping units from the original numbers.

## Transform both initial data

The normalized initial state is

$$
u(0)=A=\frac{x_0}{X},\qquad
u_s(0)=B=\frac{v_0}{X\omega_0}.
$$

The second equation includes the time-scale factor. Dividing velocity by X alone would leave inverse-time units and would not provide the correct normalized initial derivative. Two systems can have the same damping ratio but different A or B and therefore different normalized responses.

This connects to the earlier distinction between shape and velocity. A normalization is not a source of missing state information. It changes coordinates coherently and helps identify which combinations of physical inputs matter for the declared problem.

## Construct a family of physically similar systems

Let h,r and ell be positive scale factors. Starting from system A, define system B by

$$
m_B=h m_A,\quad k_B=h r^2 k_A,\quad b_B=hr b_A,
\quad X_B=\ell X_A.
$$

Choose initial values xB0=ell*xA0 and vB0=ell*r*vA0. Then omegaB=r*omegaA, the damping ratios agree, and both normalized initial data agree. Consequently the normalized problems coincide. In physical coordinates the relation is

$$
x_B(t)=\ell x_A(rt).
$$

The factors act on different aspects of the system: h scales mass and associated force coefficients, r changes the physical time rate, and ell changes displacement scale. They do not imply that the displacements at the same clock time are equal.

For example, take mA=1,kA=1,bA=1,XA=1,xA0=1,vA0=0. Choose h=3,r=2,ell=2. System B has mass 3, stiffness 12, damping 6, reference displacement 2 and initial displacement 2. Both damping ratios are one half, but the B clock runs twice as fast in the relation above. Plotting against normalized time reveals the common response.

## Use the complete analytic families, including critical damping

Let A and B denote normalized initial displacement and derivative. For zeta below one, define delta=sqrt(1-zeta squared). The response is

$$
u(s)=e^{-\zeta s}\left[A\cos(\delta s)+
\frac{B+\zeta A}{\delta}\sin(\delta s)\right].
$$

At critical damping zeta=1, the response becomes exp(-s) times [A+(B+A)s]. For zeta greater than one, replace the trigonometric functions by cosh and sinh with delta=sqrt(zeta squared-1). These formulas can each be differentiated to verify both the normalized ODE and the two initial conditions.

Near critical damping, expressions such as sin(delta*s)/delta need careful numerical treatment. Their limit is s, so a stable sinc-based evaluation can avoid a small-number division. Classification should also follow the declared parameter values consistently; rational controls can determine the sign of b squared-4mk exactly before numerical square roots are used.

A few sampled points only check the implementation on a grid. The analytic transformation and ODE argument explain why the similarity relation holds for the complete stated linear model.

## Break a group deliberately and inspect what changes

Add extra nonnegative damping to B while keeping its mass, stiffness and transformed initial data. Its damping ratio changes. The normalized law is now different, so the previously guaranteed collapse no longer follows.

If both initial displacement and velocity are zero, however, the response remains zero even after changing damping. Coincident zero curves do not establish that the laws or damping ratios match. This is an information limitation: that unexcited experiment does not reveal every parameter difference. A useful interface should distinguish equality of the defining groups from coincidence of a particular trajectory.

## Separate physical similarity from changing measurement units

All quantities above can be expressed in the same SI units while masses, springs and damping values actually differ. That is the similarity comparison being constructed. A change from meters to centimeters, by contrast, changes numerical coordinates for the same physical displacement and requires consistent changes of every dependent coefficient.

Both operations use coherent scaling, but they answer different experimental questions. State whether one apparatus is being redescribed or several apparatuses are being compared. Confusing these operations can make a normalized plot appear to establish a physical relation that was never tested.

## Check an energy relation in normalized coordinates

The physical mechanical energy is E=(m*v squared+k*x squared)/2. Dividing by kX squared gives normalized energy

$$
\mathcal E=\frac{u_s^2+u^2}{2},\qquad
\frac{d\mathcal E}{ds}=-2\zeta u_s^2\le0.
$$

The derivative follows by multiplying the normalized equation by u_s. For zero damping it is constant; for nonnegative damping it cannot increase under this unforced model. This gives another consistency relation without identifying normalized energy with an unscaled number of joules.

The energy relation is a property of the declared linear spring-damper law. It does not automatically validate a real device with nonlinear stiffness, friction, external forcing or unmodeled energy storage.

## Connect the reduction to learned simulators

A learned surrogate can exploit the normalized equation to share data across different masses and stiffnesses. Its normalized inputs still need damping ratio and initial state, while physical scales are required to reconstruct displacement and time in application units. Omitting the normalized velocity can produce different futures for the same input representation.

A collapse demonstrated inside one linear family also does not establish extrapolation to a new damping ratio or a nonlinear device. Separate the dimensional derivation, the numerical implementation and empirical evidence that the parent model describes the experiment. The reduction is useful precisely when its assumptions and reconstruction map are explicit.

## Research extension: Forced similarity requires matching the input as well

### Extend the scaling calculation to an external force

The laboratory compares unforced oscillators. For a standalone extension, consider a prescribed force f(t) and positive constant mass and stiffness, with nonnegative constant damping. Use the previous physical scaling: mass multiplier h, rate multiplier r, and length multiplier ell are all positive. Differentiate the proposed relation x_B(t)=ell x_A(rt) twice, then substitute the scaled coefficients:

$$
m_B=hm_A,\quad b_B=hrb_A,\quad k_B=hr^2k_A,\quad
x_B(t)=\ell x_A(rt),\qquad
f_B(t)=h\ell r^2 f_A(rt).
$$

Every term on the left of the B equation acquires the same factor h ell r squared. The force on the right must acquire it too. Matching damping ratio and normalized initial conditions cannot compensate for a mismatched input. The initial velocity still scales by ell r, rather than ell alone.

### Make the dimensionless forcing visible

With omega_0=sqrt(k/m), reference length X positive, s=omega_0 t, and u=x/X, the equation becomes

$$
u_{ss}+2\zeta u_s+u=p(s),\qquad
p(s)={f(s/\omega_0)\over kX}.
$$

Take X_B=ell X_A. The force scaling above makes p_B(s)=p_A(s) at the same dimensionless time. For a sinusoidal force, both its amplitude and frequency must transform: F_B=h ell r² F_A and Omega_B=r Omega_A. Keeping the old numerical frequency while changing omega_0 generally changes the ratio nu=Omega/omega_0 and therefore changes the normalized problem.

### Work a frequency-response counterexample

For positive damping, the decaying transient leaves a sinusoidal steady response. Substituting a sine-and-cosine ansatz into the normalized equation gives the displacement amplitude

$$
U={F/(kX)\over\sqrt{(1-\nu^2)^2+(2\zeta\nu)^2}},\qquad
\nu={\Omega\over\omega_0}.
$$

Choose A with m=1, k=4, b=1, X=1, force amplitude F=4, and frequency Omega=1, using coherent SI coordinates. Then omega_0=2, zeta=1/4, and nu=1/2. With h=3,r=2,ell=2, B has m=3,k=48,b=6,X=2. Its correct forcing is amplitude 96 at frequency 2. The normalized amplitude agrees, approximately 1.26491, while its physical displacement amplitude is twice as large. Leaving the B force frequency at one changes its normalized response even though the damping ratios still match.

| Comparison | What must match after normalization | Common missing item |
|---|---|---|
| Unforced initial-value motion | Damping ratio and both initial coordinates | Initial velocity scaling |
| Forced initial-value motion | The same data and the entire p(s) | Force amplitude or time argument |
| Sinusoidal steady response with damping | Damping ratio, force ratio and frequency ratio | Omega/omega_0 |

At zero damping and exact resonance, this finite steady-amplitude formula is not available: a nonzero force proportional to cos(omega_0 t) produces a term proportional to t sin(omega_0 t). Away from resonance an undamped periodic particular solution can exist, but initial transients do not decay. Thus “same normalized amplitude” and “attracting steady response” need their own assumptions.

**Worked question:** Does agreement in a free-decay benchmark justify transferring a learned model to arbitrary driven experiments? **Answer:** No. The input is part of the problem. A useful transfer claim specifies its normalized input class and initial conditions, then checks the force transformation. The code verifies the chain-rule scaling with a manufactured polynomial trajectory using exact fractions, separately checks the sinusoidal amplitude, and never treats these forces as new controls in the unforced widget.

```python
from fractions import Fraction as F
from math import sqrt,isclose
m,k,b=F(1),F(4),F(1)
h,r,ell=F(3),F(2),F(2)
mb,kb,bb=h*m,h*r*r*k,h*r*b
def x(t):return 1+t+t*t+t**3
def dx(t):return 1+2*t+3*t*t
def ddx(t):return 2+6*t
def force(t):return m*ddx(t)+b*dx(t)+k*x(t)
for t in [F(0),F(1,4),F(1,2),F(1),F(2)]:
    lhs=mb*ell*r*r*ddx(r*t)+bb*ell*r*dx(r*t)+kb*ell*x(r*t)
    assert lhs==h*ell*r*r*force(r*t)
def amplitude(m,k,b,X,force,frequency):
    omega=sqrt(float(k/m));zeta=float(b)/(2*sqrt(float(m*k)))
    nu=float(frequency)/omega
    return float(force/(k*X))/sqrt((1-nu*nu)**2+(2*zeta*nu)**2)
ua=amplitude(m,k,b,F(1),F(4),F(1))
ub=amplitude(mb,kb,bb,ell,F(96),F(2))
wrong=amplitude(mb,kb,bb,ell,F(96),F(1))
assert isclose(ua,ub,rel_tol=1e-14) and abs(wrong-ua)>.1
print('matched normalized amplitudes',ua,ub,'unscaled-frequency response',wrong)
```

## Exercises with worked solutions

1. **Exercise:** Why must X be positive and independent of a possibly zero x0? **Solution:** It is a normalization denominator; choosing X=x0 would fail or reverse conventions for zero or negative initial displacement.
2. **Exercise:** What normalized initial velocity accompanies v0? **Solution:** B=v0/(X*omega0), including both displacement and time scales.
3. **Exercise:** Under the constructed rescaling, how does omega change? **Solution:** kB/mB=r squared*kA/mA, so omegaB=r*omegaA.
4. **Exercise:** Does matching zeta alone guarantee equal normalized trajectories? **Solution:** No. Both normalized initial data must also match.
5. **Exercise:** Can zero curves hide different damping ratios? **Solution:** Yes. Zero displacement and velocity give the zero solution for every supported damping ratio.
6. **Exercise:** What does the normalized energy derivative test? **Solution:** It checks dissipation under the stated unforced linear model, not empirical validity of every apparatus.

## Sources and further study

The [official MIT damping handout](https://ocw.mit.edu/courses/18-03sc-differential-equations-fall-2011/7e212064ad281d00e1dac893b1f722a7_MIT18_03SCF11_s13_2text.pdf) provides the characteristic-root regimes and complete solution families. The scaling construction, initial-data transformation and normalized energy identity above are direct derivations. 
