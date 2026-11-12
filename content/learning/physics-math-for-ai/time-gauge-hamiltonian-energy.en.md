## The same motion can have a different canonical description

A free particle can keep moving at constant speed while its canonical momentum changes. A restoring oscillator can conserve physical mechanical energy while a valid Hamiltonian changes sign. Both statements become concrete when we add a total time derivative to the Lagrangian and consistently transform the momentum variable.

This lesson uses a one-dimensional particle with m>0 and k≥0. Its physical force is always −kx. The parameter introduced below changes the canonical representation, not that force. We will derive the full equations so that apparent energy or force changes can be checked rather than inferred from one term of a formula.

## Start with a natural Lagrangian and a gauge function

Let

$$
L_0(x,v)=\frac m2v^2-\frac k2x^2,
\qquad F(x,t)=\frac\gamma2t x^2.
$$

The function F has action units Js. Here gamma has units kg/s², the same dimension as stiffness. It differs from the coefficient in the earlier time-independent boundary-gauge example, which had units kg/s. Time dependence changes the dimensional accounting.

The total derivative must include both terms:

$$
\frac{dF}{dt}=\frac{\partial F}{\partial t}
+\frac{\partial F}{\partial x}v
=\frac\gamma2x^2+\gamma t x v.
$$

Define Lgamma=L₀+dF/dt. Both added contributions have energy units J. Keeping only the velocity-linear term would produce a different force law, rather than the intended equivalent Lagrangian.

## Explain the boundary condition for equivalence

For fixed endpoint positions and fixed endpoint times,

$$
S_\gamma[x]-S_0[x]=F(x(T),T)-F(x(0),0).
$$

This boundary quantity is the same for all admissible paths with those endpoints. Interior Euler–Lagrange equations are unchanged. Endpoint positions or times that are allowed to vary require a corresponding review of the boundary terms; we are not claiming all such variational problems are identical.

One can also verify the interior cancellation directly. The velocity derivative is mv+gamma t x. Its total derivative is m acceleration+gamma x+gamma t v. The position derivative is −kx+gamma t v+gamma x. Subtracting leaves m acceleration+kx=0.

## Define the new canonical momentum

For this Lagrangian,

$$
P=\frac{\partial L_\gamma}{\partial v}=mv+\gamma t x,
\qquad p_{\mathrm{mech}}=mv.
$$

Both quantities have momentum units kg m/s, but only the second is the Cartesian mechanical momentum. At a given time, the same observed position and velocity lead to different P when gamma changes. Comparisons of learned canonical states therefore require a stated Lagrangian convention.

The velocity Hessian remains m>0. Hence the Legendre map is regular for every x and t, with inverse v=(P−gamma t x)/m. A particle passing through x=0 has a valid canonical state in this Cartesian model. There is no polar coordinate singularity here.

## Eliminate velocity before differentiating H

Perform the Legendre transform P v−Lgamma and substitute the inverse velocity:

$$
H_\gamma(x,P,t)=\frac{(P-\gamma t x)^2}{2m}
+\frac{k-\gamma}{2}x^2.
$$

This is a function of independent x, P and t. The expression mv²/2+kx²/2−gamma x²/2 is its value along the corresponding physical state, but using v as an independent argument would change which partial derivatives we are calculating.

The physical mechanical energy remains

$$
E=\frac m2v^2+\frac k2x^2,
\qquad H_\gamma=E-\frac\gamma2x^2=E-\frac{\partial F}{\partial t}.
$$

Thus the Legendre transform is well defined while its value differs from the conserved mechanical energy. A negative Hgamma is possible without a negative kinetic energy or a change in restoring-force sign.

## Recover the correct Newton equation

Hamilton equations give

$$
\dot x=\frac{\partial H_\gamma}{\partial P}
=\frac{P-\gamma t x}{m}=v,
$$

$$
\dot P=-\frac{\partial H_\gamma}{\partial x}
=\gamma t\frac{P-\gamma t x}{m}-(k-\gamma)x.
$$

Differentiate P=mv+gamma t x along the same motion. Its derivative is m acceleration+gamma x+gamma t v. Equating the two momentum-rate expressions cancels the gauge terms and gives m acceleration=−kx.

In particular, the coefficient k−gamma in H is not the physical spring stiffness by itself. Its contribution must be combined with the momentum shift and its explicit time dependence. The complete canonical equations determine motion.

## Separate a partial time derivative from a total one

At fixed x and P,

$$
\frac{\partial H_\gamma}{\partial t}
=-\gamma x\frac{P-\gamma t x}{m}=-\gamma x v.
$$

Along a solution of Hamilton equations, the position and momentum terms in the total derivative cancel, giving dHgamma/dt=∂Hgamma/∂t. The physical energy meanwhile satisfies dE/dt=v(m acceleration+kx)=0. Equivalently, differentiate Hgamma=E−gamma x²/2 along the physical trajectory to obtain the same −gamma x v.

Explicit time dependence is a property of the function. For nonzero gamma, Hgamma depends explicitly on time in its canonical arguments. A particular resting solution may nevertheless have x v=0 and constant Hgamma. Likewise, a zero derivative at one turning point does not prove that H is constant throughout an orbit.

## Worked example: free motion with changing canonical momentum

Set m=1, k=0, x₀=0, v₀=1 and gamma=2. Then x=t and v=1. Mechanical momentum stays 1 kg m/s, physical force is zero, and physical energy stays 1/2 J. But

$$
P(t)=1+2t^2,\qquad H_\gamma(t)=\frac12-t^2.
$$

At t=1, P=3 kg m/s, Pdot=4 N, Hgamma=−1/2 J and dHgamma/dt=−2 W. These are changes in canonical quantities, not an applied mechanical force or physical energy loss.

At that instant Lgamma=3.5 J and P v−Lgamma=−0.5 J, providing a direct Legendre check. The ordinary action accumulated from zero to one second is 0.5 Js. F at the final endpoint is 1 Js, so the gauged action is 1.5 Js. Instantaneous Lagrangian, Hamiltonian and accumulated action have different roles and units.

## Worked example: a restoring oscillator with sign-changing H

Choose m=1, Ω=π/2, k=π²/4, x₀=1, v₀=0 and gamma=4. Physical motion is x=cos(πt/2), with constant E=π²/8 J. Initially

$$
H_\gamma(0)=\frac{\pi^2}{8}-2<0.
$$

At one second, x=0 and v=−π/2, so Hgamma=π²/8>0. The physical energy is unchanged. Although k−gamma is negative in this representation, the particle still obeys the original restoring equation.

The derivative −gamma x v vanishes at both of those displayed instants, while H changes between them. Read the time curve rather than treating one zero derivative as a conservation certificate.

## Track the phase-action boundary term

The canonical one-form identity is

$$
P\,dx-H_\gamma\,dt
=p_{\mathrm{mech}}\,dx-E\,dt+dF.
$$

Check its two coefficients separately. The dx coefficient changes by gamma t x=∂F/∂x, and the dt coefficient changes by gamma x²/2=∂F/∂t. Each complete term has action units Js. This explains the fixed-endpoint action equivalence and the time-dependent Hamiltonian shift together.

The model checks these coefficient identities and also compares their values along the trajectory. They are direct identities for this example; the full theory of canonical transformations, Poisson brackets and symplectic geometry receives separate treatment later.

## Read the computed histories and their scope

The controls set Ω=πf/4 and k=mΩ². Initial position and velocity codes are divided by two, gamma's code is divided by two, and the query-time code is divided by twenty seconds. Changing mass at fixed f also changes stiffness. Changing gamma leaves the physical trajectory invariant.

The implementation evaluates the analytic free or harmonic trajectory over two seconds and transforms it into canonical variables. It is not a new numerical Hamilton ODE solver. Physical momentum and canonical momentum can be compared on phase plots with position in meters and momentum in kg m/s. Such plots are not spatial trajectories.

Plot physical energy and Hgamma with energy units J; plot dHgamma/dt and its explicit partial derivative with power units W. A fixed-time contour of a time-dependent H need not contain the entire evolving phase trajectory. The code reports floating-point identity residuals and sampled maxima without forcing them to zero or claiming bounds at all real times.

## Research extension: circle winding and global boundary terms

The earlier function $F=\gamma t x^2/2$ is globally defined on the line. On a circle, the local expression $F=\alpha\theta$ needs a branch of angle. Compare the local equations and the action of complete paths before calling a term an irrelevant boundary correction. This circle example is an executable companion, separate from the one-dimensional native controls.

### Constant momentum translation preserves local motion

Let a particle move on a fixed circle of radius $R$, with inertia $I$. Angles are dimensionless, angular velocity has units of inverse time, and $\alpha$ has units of J s.

$$
L_\alpha=\frac I2\dot\theta^2+\alpha\dot\theta,\qquad
I=mR^2>0,\quad \alpha=\text{constant},\qquad I\ddot\theta=0.
$$

The additional term changes canonical momentum but has no explicit time dependence. Legendre transformation gives the physical kinetic energy in the translated momentum coordinate:

$$
p_\alpha=I\dot\theta+\alpha,\qquad
H_\alpha=\frac{(p_\alpha-\alpha)^2}{2I}=E_{\rm physical},\qquad
\oint\alpha\,d\theta=2\pi\alpha.
$$

Unlike the earlier time-dependent $\gamma$ example, here $H_\alpha=E$. The constants $\alpha$ and $\gamma$ have different units and must not be identified. Translation by constant $\alpha$ is a globally defined symplectic map of the cylinder:

$$
\Psi(\theta,p)=(\theta,p+\alpha),\qquad
\Psi^*(d\theta\wedge dp)=d\theta\wedge dp,\qquad
\Psi^*(p\,d\theta)-p\,d\theta=\alpha\,d\theta.
$$

For nonzero $\alpha$, the one-form difference is closed but not exact. Consequently symplectic preservation does not guarantee one global scalar generating function. Locally $\alpha\dot\theta=d(\alpha\theta)/dt$; globally the integral remembers winding.

### Fix physical endpoints and keep the winding class

Hold duration $T>0$ fixed and require identical physical endpoint angles modulo $2\pi$. A lifted path can end at a different integer turn. The constant-speed paths are stationary separately in every winding class:

$$
\theta_n(t)=\theta_0+\frac{2\pi n}{T}t,\qquad
\mathcal A_{\alpha,n}=\frac{I(2\pi n)^2}{2T}+2\pi\alpha n,\quad n\in\mathbb Z.
$$

For a fixed winding $n$, the extra action is a constant and all first variations are unchanged. Across different winding classes it is not the same constant. For the following SI example, define the dimensionless gauge strength $a$ explicitly:

$$
\frac{\mathcal A_{a,n}}{\pi\,{\rm J\,s}}=n^2+2an=(n+a)^2-a^2,\qquad
\frac{E_n}{\rm J}=\frac{n^2}{2},\quad
I=1\,{\rm kg\,m^2},\ T=2\pi\,{\rm s},\ \alpha=a\,{\rm J\,s}.
$$

![Winding paths and their circle gauge actions](/assets/learning/analytical-circle-gauge.en.svg)

| Winding $n$ | $\mathcal A_{0,n}/(\pi\,{\rm J\,s})$ | $\mathcal A_{1,n}/(\pi\,{\rm J\,s})$ | $E_n/\mathrm{J}$ |
|---|---|---|---|
| -2 | 4 | 0 | 2 |
| -1 | 1 | -1 | 1/2 |
| 0 | 0 | 0 | 0 |
| 1 | 1 | 3 | 1/2 |
| 2 | 4 | 8 | 2 |

Only integer winding numbers are admissible. In the figure, connecting lines guide the eye and do not introduce fractional winding paths.

Completing the square proves the minimum over all integer winding numbers: $a=0$ selects $n=0$; $a=1$ selects $n=-1$; $a=1/2$ ties $n=-1,0$. Within each class, Cauchy–Schwarz gives $\int_0^T\dot\theta^2dt\ge(2\pi n)^2/T$, so the constant-speed path attains that class minimum. Combining both arguments proves the global result over all admissible finite-kinetic-action paths. A negative action does not imply negative energy.

The initial-value problem is unchanged: specifying the same physical initial angle and angular velocity gives the same trajectory. The changed minimum concerns a different question, optimization across complete paths with fixed physical endpoints but variable winding. No quantum phase or quantization assumption is needed.

### Execute exact arithmetic and test the interpretation

The cell prints the table using rational arithmetic. Its finite search checks the displayed examples; the inequality and completed square above establish the unrestricted minimum.

```python
from fractions import Fraction as Q

# I=1 kg m^2, T=2*pi s; alpha is measured in J s.
def state(alpha, n):
    alpha = Q(alpha)
    momentum = n + alpha
    energy = Q(n*n, 2)
    action_over_pi = n*n + 2*alpha*n
    assert (momentum-alpha)**2/2 == energy
    assert action_over_pi == (n+alpha)**2-alpha**2
    return momentum, energy, action_over_pi

for n in range(-2, 3):
    print(n, state(0, n)[2], state(1, n)[2], state(1, n)[1])
for alpha, expected in [(Q(0), [0]), (Q(1, 2), [-1, 0]), (Q(1), [-1])]:
    values = {n: state(alpha, n)[2] for n in range(-4, 5)}
    assert [n for n, a in values.items() if a == min(values.values())] == expected
# The completed square proves the result over ALL integers.
# Finite enumeration alone would not establish a global minimum.
```

**Research questions with answers**

- Why does the Euler–Lagrange equation miss the winding correction? **Answer:** Small continuous variations with fixed endpoints stay in one winding class, where the correction is constant.
- Is $n=-1$ a new trajectory selected from identical initial data? **Answer:** No. It minimizes a boundary-value action over classes; it does not replace the solution of a fixed initial-value problem.
- What should an AI action model record? **Answer:** The configuration domain, allowed winding classes, whether its generating function is local or global, and which endpoint data are fixed. Matching local equations alone does not determine all path actions.

## Exercises with explanations

1. Derive the inverse momentum map. **Solution:** Subtract gamma t x from P and divide by positive mass, giving v=(P−gamma t x)/m. The derivative of P with respect to v is m, so the inverse exists everywhere in this model.
2. Evaluate the free example at t=1. **Solution:** Position and velocity are both one, pmech=1, P=3, E=1/2 and Hgamma=−1/2. Hamilton momentum rate is 4 N, while the physical force remains zero.
3. What happens if gamma x²/2 is omitted from Lgamma? **Solution:** The remaining term gamma t x v produces an uncanceled gamma x in the Euler–Lagrange equation. The equation becomes m acceleration+(k+gamma)x=0, so this omission changes the physical model.
4. Is every nonzero-gamma trajectory guaranteed to have changing H? **Solution:** No. A free resting particle has constant x and v=0, so −gamma x v=0. The Hamiltonian function is explicitly time dependent even though its value is constant on that selected solution.
5. Explain the oscillator's change from negative to positive H without energy input. **Solution:** Hgamma=E−gamma x²/2. The conserved physical E is unchanged while the position-dependent subtraction varies; the complete canonical equations still give m acceleration=−kx.
6. Why does changing the boundary action not establish equivalence for arbitrary free endpoints? **Solution:** If endpoint positions or times vary, F's endpoint variation need not vanish. Those boundary conditions must be transformed consistently, rather than using only the unchanged interior Newton equation.

## Research connection and further reading

An AI model trained to identify a Hamiltonian needs an explicit convention for its canonical variables and time dependence. A conservation penalty on the chosen H is appropriate only under the relevant assumptions; in this example physical E is conserved while Hgamma can vary. Compare induced motion and stated physical observables in addition to scalar function values.

David Tong's [The Hamiltonian Formalism](https://davidtong.org/teaching/classical-dynamics/dynhtml/S4) develops Legendre inversion, canonical momentum and the total-versus-explicit time derivative of H. Here the time-gauge example is derived directly and checked using independent symbolic elimination, high-precision differentiation and direct action quadrature. The rational free example also admits exact Fraction-based verification.
