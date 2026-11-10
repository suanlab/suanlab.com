## A repeated push does not immediately produce a repeating motion

Imagine pushing a swing at regular intervals. Its motion depends on the timing of the pushes, but also on where it started and how fast it was already moving. A frequency-response curve describes one particular repeating motion. It does not by itself answer what happens after switching the drive on.

This chapter derives the initial-value response to a sinusoidal force. Prerequisites are a second-order linear differential equation, complex numbers and mechanical energy. First predict whether motion should decay, persist or grow; then compare position, velocity and energy at the same physical time.

## Specify the system and units

Use a mass attached to a linear spring with viscous damping:

$$
m\ddot x+c\dot x+kx=F_0\cos(\Omega t),\qquad x(0)=x_0,\quad\dot x(0)=v_0.
$$

Mass $m$ is in kg, displacement $x$ in m, stiffness $k$ in N/m, damping $c$ in N s/m, and force amplitude $F_0\ge0$ in N. Angular frequency $\Omega\ge0$ is in rad/s, with time in seconds. The model assumes $m,k>0$ and $c\ge0$. Damping is proportional to velocity; Coulomb friction and nonlinear springs require different equations.

The force starts at its positive maximum. Changing the starting phase changes the initial-value experiment. An angular frequency of 1 rad/s is not one cycle per second: its period is $2\pi$ s.

## Separate a particular solution from initial memory

Linearity allows $x=x_p+x_h$. The particular solution satisfies the driven equation. The homogeneous correction satisfies $m\ddot x_h+c\dot x_h+kx_h=0$ and supplies the initial conditions that the particular solution lacks:

$$
x_h(0)=x_0-x_p(0),\qquad \dot x_h(0)=v_0-\dot x_p(0).
$$

The split is a useful representation, not two separate physical masses. At a time when the two contributions nearly cancel, the actual displacement can be small while each contribution is large. Near undamped resonance, evaluating a large particular solution and subtracting a large homogeneous correction can lose floating-point precision.

## Derive the periodic particular response

With the convention $\operatorname{Re}(F_0e^{i\Omega t})$, substitute $x_p=\operatorname{Re}(Xe^{i\Omega t})$:

$$
X=\frac{F_0}{k-m\Omega^2+ic\Omega},\qquad
x_p=A\cos(\Omega t-\phi).
$$

Here $A=F_0/\sqrt{(k-m\Omega^2)^2+c^2\Omega^2}$ and $\phi=\operatorname{atan2}(c\Omega,k-m\Omega^2)$. Consequently $x_p(0)=A\cos\phi$ and $\dot x_p(0)=A\Omega\sin\phi$. Use the two-argument angle: above the natural frequency the lag lies above $\pi/2$, which a one-argument arctangent can miss.

This formula requires a nonzero denominator. For nonzero forcing, its failure at exact undamped resonance is a failure of the bounded periodic ansatz, not a failure of the initial-value equation. If the forcing amplitude is zero, the zero particular solution is always available even when the intrinsic transfer function is singular. The actual motion is then the homogeneous initial-value solution.

## Understand the three damping regimes

Write $\alpha=c/(2m)$ and $\omega_0=\sqrt{k/m}$. When $\alpha<\omega_0$, define $\omega_d=\sqrt{\omega_0^2-\alpha^2}$. For $a=x_h(0)$ and $b=\dot x_h(0)$,

$$
x_h=e^{-\alpha t}\left[a\cos(\omega_dt)+\frac{b+\alpha a}{\omega_d}\sin(\omega_dt)\right].
$$

At critical damping, $\alpha=\omega_0$, the repeated-root expression is $x_h=[a+(b+\alpha a)t]e^{-\alpha t}$. In the overdamped case the two real roots $-\alpha\pm\sqrt{\alpha^2-\omega_0^2}$ give decaying exponentials. The free motion then does not oscillate, but a periodic external force can still produce a periodic particular motion.

For $c>0$ all homogeneous contributions decay. For $c=0$ they persist. Calling every homogeneous contribution a “decaying transient” would therefore be incorrect.

## Beats are interference between two temporal frequencies

For zero initial data and no damping, away from resonance,

$$
x(t)=\frac{F_0}{m}\frac{\cos(\Omega t)-\cos(\omega_0t)}{\omega_0^2-\Omega^2}.
$$

When the frequencies are close, the rapid oscillation has a slow envelope. Use the cosine-difference identity to see both scales. An equivalent expression continuous at resonance is

$$
x(t)=\frac{F_0t}{m(\omega_0+\Omega)}
\sin\!\left(\frac{\omega_0+\Omega}{2}t\right)
\operatorname{sinc}\!\left(\frac{\omega_0-\Omega}{2}t\right),\quad
\operatorname{sinc}(z)=\frac{\sin z}{z},\quad \operatorname{sinc}(0)=1.
$$

These beats concern one driven degree of freedom. Coupled oscillators and their normal modes are a separate topic.

## Exact resonance grows over time

At $c=0$ and $\Omega=\omega_0$, the zero-initial-data solution is

$$
x(t)=\frac{F_0}{2m\omega_0}t\sin(\omega_0t).
$$

Its envelope grows linearly. At each finite time the displacement is finite; there is no bounded repeating response to which it settles. A real spring eventually leaves the linear regime, so indefinite growth is a prediction within this idealized model.

Taking $c\to0$ in a long-time frequency-response amplitude and taking a finite-time initial-value limit are different operations. State the observation time and initial conditions whenever comparing them.

## A representation that crosses damping and resonance boundaries

The accompanying calculation augments the state with the forcing clock:

$$
\frac{d}{dt}\begin{pmatrix}x\\v\\u\\z\end{pmatrix}=
\begin{pmatrix}0&1&0&0\\-k/m&-c/m&F_0/m&0\\0&0&0&-\Omega\\0&0&\Omega&0\end{pmatrix}
\begin{pmatrix}x\\v\\u\\z\end{pmatrix},\qquad
(u(0),z(0))=(1,0).
$$

Thus $u=\cos(\Omega t)$ and $z=\sin(\Omega t)$. A matrix exponential evaluates the finite-time linear flow without dividing by the resonance denominator. Scaling, a Taylor series and squaring approximate that exponential in floating-point arithmetic. This is not exact arithmetic and does not provide a certified error bound.

The supported calculation ranges are $m,k\in[0.25,4]$, $c\in[0,8]$, $F_0\in[0,4]$, $\Omega\in[0,8]$, $x_0,v_0\in[-4,4]$ and $t\in[0,32]$ in the stated SI units. These are experiment bounds, not laws of nature.

## Read position, phase and energy together

The interactive panels show force, displacement and velocity on separate physical axes, followed by the position–velocity trajectory and energy. The force starts at its maximum. Use the query-time control to move the selected point through the same actual initial-value solution. Mark the selected time consistently on every plot. At startup, the displacement is generally not a single sinusoid, so a single steady-state phase lag does not describe the entire trace.

The model returns 257 samples of the linear flow. Connecting them is a drawing convention; these are not 256 time-integration updates. The maximum phase advance per drawing interval is reported. More samples improve visual resolution, rather than changing the underlying physical equation.

## Derive the finite-time energy balance

Multiply the motion equation by $v=\dot x$:

$$
E=\frac12mv^2+\frac12kx^2,\qquad
\dot E=F_0\cos(\Omega t)v-cv^2,\qquad
E(T)-E(0)=W_{\rm drive}-Q_{\rm heat}.
$$

Heat $Q_{\rm heat}=\int_0^Tcv^2dt$ is nonnegative. Drive work can be negative: an imposed force opposing the instantaneous velocity extracts mechanical energy. The calculation independently approximates both integrals with adaptive Simpson quadrature and reports their balance residual. It resolves oscillation and decay scales before subdividing intervals further, separately from the drawing samples. Local subdivision estimates and completion status are reported; these are diagnostics, not certified global error bounds. A startup window average $W/T$ need not equal a periodic steady-state mean.

## Worked example: select a purely periodic solution

Choose $m=k=c=F_0=\Omega=1$ in SI units. The equation is $\ddot x+\dot x+x=\cos t$. The periodic solution is $x_p=\sin t$, because $-\sin t+\cos t+\sin t=\cos t$. Starting at $x_0=0,v_0=1$ eliminates the homogeneous correction.

Over $T=2\pi$ s, input power and heat power are both $\cos^2t$ W. Each integral is $\pi$ J, while the mechanical energy remains $1/2$ J. A nonzero flow of energy can pass through a system whose stored energy is constant.

## Worked example: finite resonant buildup

Set $m=k=F_0=\Omega=1$, $c=0$, $x_0=v_0=0$. At $t=\pi/2$ s, $x=\pi/4$ m and $v=1/2$ m/s. Therefore $E=1/8+\pi^2/32$ J. The finite result is consistent with resonance: resonance predicts growing amplitude, not infinite displacement immediately after switching on.

## Six exercises with explanations

1. **Initial memory:** For the first example but $x_0=v_0=0$, find the homogeneous initial data. **Answer:** $a=0,b=-1$. The particular velocity is already 1 m/s, so the homogeneous part must cancel it initially.
2. **Critical damping:** With $m=k=1$, find critical $c$. **Answer:** $c=2$ N s/m, from $c^2=4mk$. A sinusoidal force still permits a periodic particular response.
3. **Negative work:** Let $F(0)=1$ N and $v_0=-1$ m/s. What is the initial input power? **Answer:** $-1$ W. For $c=1$, heat power is $+1$ W, so mechanical energy initially decreases at 2 W.
4. **DC equilibrium:** Set $\Omega=0$ and $x_0=F_0/k,v_0=0$. **Answer:** The mass stays at the displaced equilibrium. Its stored energy is $F_0^2/(2k)$; there is no finite driving cycle.
5. **A growing envelope:** Differentiate the resonant solution at $m=k=F_0=1$. **Answer:** $v=(\sin t+t\cos t)/2$. Dropping the derivative of the envelope would miss the $\sin t/2$ term.
6. **A zero residual:** Does defining heat as $W-\Delta E$ test energy conservation? **Answer:** It forces a zero residual algebraically. Compute drive work and heat independently before comparing them with the energy change.

## Use the oscillator as an AI validation problem

A learned dynamics model should reproduce held-out initial conditions, the drive phase, damping boundaries and finite-time resonance. Report units and observation horizon. A low displacement loss on one damped trajectory cannot establish that the model learned forcing, phase or energy transfer.

The independent reference generator uses refined SciPy DOP853 solutions for position, velocity, work and heat, with a separate SciPy matrix-exponential check. The educational model and those references share the physical equation, but do not share implementation code. Finite observation and numerical agreement do not establish correctness outside the tested parameter range.

## Sources and the next question

See [MIT 8.03SC lecture 3, pp. 1–5](https://ocw.mit.edu/courses/8-03sc-physics-iii-vibrations-and-waves-fall-2016/d07bffb5150376ad8bc942566f327f1d_MIT8_03SCF16_Lec3.pdf) for forced and homogeneous solutions. The next chapter derives frequency-dependent gain, phase and absorbed power, and explains why their maxima answer different questions.
