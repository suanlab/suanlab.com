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

## Exercises with worked solutions

1. **Exercise:** Why must X be positive and independent of a possibly zero x0? **Solution:** It is a normalization denominator; choosing X=x0 would fail or reverse conventions for zero or negative initial displacement.
2. **Exercise:** What normalized initial velocity accompanies v0? **Solution:** B=v0/(X*omega0), including both displacement and time scales.
3. **Exercise:** Under the constructed rescaling, how does omega change? **Solution:** kB/mB=r squared*kA/mA, so omegaB=r*omegaA.
4. **Exercise:** Does matching zeta alone guarantee equal normalized trajectories? **Solution:** No. Both normalized initial data must also match.
5. **Exercise:** Can zero curves hide different damping ratios? **Solution:** Yes. Zero displacement and velocity give the zero solution for every supported damping ratio.
6. **Exercise:** What does the normalized energy derivative test? **Solution:** It checks dissipation under the stated unforced linear model, not empirical validity of every apparatus.

## Sources and further study

The [official MIT damping handout](https://ocw.mit.edu/courses/18-03sc-differential-equations-fall-2011/7e212064ad281d00e1dac893b1f722a7_MIT18_03SCF11_s13_2text.pdf) provides the characteristic-root regimes and complete solution families. The scaling construction, initial-data transformation and normalized energy identity above are direct derivations. 
