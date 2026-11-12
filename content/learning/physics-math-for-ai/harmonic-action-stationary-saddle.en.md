## Does nature always minimize action?

A restoring spring pulls a displaced mass back toward equilibrium. Its force law is stable and its initial-value problem has a unique solution. Nevertheless, the action of a classical trajectory need not be a minimum over every fixed time interval. To understand this apparent puzzle, distinguish stability of physical motion from the curvature of a functional over candidate paths.

We will keep both endpoint positions at zero and compare paths over a duration T. The zero path is always a classical solution. Changing T changes which variations compete with it. This provides a concrete example of a stationary minimum becoming flat and then becoming a saddle, without changing the restoring sign of the spring force.

## Define the physical oscillator and trial family

Let m>0, angular frequency ω₀≥0, and stiffness k=mω₀². The force is −kq, the potential is kq²/2, and the Lagrangian is kinetic minus potential energy:

$$
L(q,\dot q)=\frac m2\dot q^2-\frac k2q^2.
$$

Write s=t/T and prescribe

$$
q(s)=A\sin(\pi s)+B\sin(3\pi s).
$$

Both modes have zero endpoints. A and B are lengths, not probabilities or physical frequencies. Their temporal frequencies are π/T and 3π/T; these generally differ from the oscillator's natural frequency ω₀. Therefore most selected trials are not physical oscillator solutions. The controls prescribe functions and evaluate them; they do not integrate the equations of motion.

## Integrate the action using orthogonal modes

The two sine modes are orthogonal on [0,1], as are their cosine derivatives. Each squared sine or cosine integrates to 1/2. Consequently,

$$
S[q]=\frac{m\pi^2}{4T}
\left[\left(1-\frac{\omega_0^2T^2}{\pi^2}\right)A^2
+\left(9-\frac{\omega_0^2T^2}{\pi^2}\right)B^2\right].
$$

The derivative brings a factor nπ/T into kinetic energy. Potential energy instead accumulates for time T. On a longer interval, a broad displacement can move slowly while spending substantial time away from equilibrium. The negative potential contribution to L can then outweigh the positive kinetic contribution to action. Neither energy contribution is itself negative.

The action has units Js. Its Hessian entries with respect to amplitudes have units Js/m². Calling a negative action “negative energy” confuses an integral of a difference with the instantaneous sum of two nonnegative energies.

## Identify the stationary reference before classifying it

At A=B=0 the selected function is q=0. The first variation vanishes for every fixed-endpoint variation because the full equation m qddot+kq=0 holds. The action's curvature at that reference depends on the dimensionless horizon ω₀T/π.

If ω₀T<π, both displayed coefficients are positive. If ω₀T=π, the first coefficient is zero and the third-mode coefficient remains positive. If π<ω₀T<3π, the first is negative while the third is positive, giving both descending and ascending directions through q=0. This proves a saddle in the displayed family and, by the same two admissible directions, in the full path space.

A nonzero trial with negative action is not itself the stationary saddle. It is a test path lying below the zero reference. Always report the reference's curvature separately from whether the currently selected path satisfies the differential equation.

## Prove the minimum beyond two modes

For a full fixed-endpoint statement, take variations η in H₀¹(0,T): functions with zero endpoint traces and square-integrable weak derivative. Expand in the sine basis, η(t)=Σ cₙ sin(nπt/T). Parseval's identities give

$$
\int_0^T\eta^2\,dt=\frac T2\sum_{n\geq1}c_n^2,
\qquad
\int_0^T\dot\eta^2\,dt=\frac{\pi^2}{2T}\sum_{n\geq1}n^2c_n^2.
$$

Each n²≥1, so the second integral is at least π²/T² times the first. One can first establish this for finite sine sums and extend in the H₀¹ norm. Equality permits only the first sine mode. It follows that action is strictly positive for nonzero η when ω₀T<π, and nonnegative with a flat first-mode family when ω₀T=π.

For longer intervals, one negative mode and one sufficiently high positive mode establish a saddle. Our bounded controls keep ω₀T/π≤2.5, so mode three always supplies a positive direction. This control restriction explains why the displayed second coefficient never changes sign; it is not a claim that all higher horizons behave identically.

## Use exact classification without pretending sine is rational

The model sets ω₀=πf/4 with integer f from zero to four, and T=N/10 seconds with integer N from five to twenty-five. Then

$$
D_1=40^2-(fN)^2,\qquad D_3=120^2-(fN)^2,
$$

$$
S=\pi^2\frac{m}{6400T}(D_1A^2+D_3B^2).
$$

The signs of D₁ and D₃ are exact integer decisions. In particular, equality at fN=40 is not guessed with a floating-point tolerance. Rational controls give a rational coefficient multiplying π²; they do not make the action itself a rational number. Sine evaluations, cumulative partial integrals, and plotted decimal values use floating-point arithmetic and retain diagnostic discrepancies.

At fixed f, changing mass also changes stiffness k=mω₀². It preserves natural frequency while scaling the action and force. A mass sweep here is therefore not an experiment at fixed spring stiffness.

## Test stationarity of the entire selected path

The Euler–Lagrange residual factors as

$$
R(t)=m\ddot q+kq
=-\frac{m\pi^2}{1600T^2}
\left[D_1A\sin(\pi s)+D_3B\sin(3\pi s)\right].
$$

Because the sine modes are independent and D₃>0 in the lab, the full residual vanishes exactly when D₁A=0 and B=0. This is stronger than checking the residual at one cursor position. At the endpoints every trial's sine terms vanish, so an endpoint residual of zero gives no stationarity certificate.

The mean-square residual over the interval is

$$
\frac1T\int_0^T R(t)^2\,dt
=\frac{m^2\pi^4}{2\cdot1600^2T^4}
(D_1^2A^2+D_3^2B^2).
$$

Its square root has force units and checks the whole prescribed path. The formula comes from exact orthogonality, not from a finite set of time samples.

## Understand the flat critical family

At ω₀T=π, every path A sin(πt/T), with B=0, is a classical solution joining zero to zero. Its action is zero for every A. However, its initial velocity is Aπ/T, so different amplitudes correspond to different complete initial conditions.

This is nonuniqueness of a boundary-value problem, not nonuniqueness of an initial-value problem. Given both position and velocity at time zero, the smooth linear oscillator has one solution. Two endpoint photographs can omit precisely the information that distinguishes these trajectories.

For any nonzero critical amplitude, total mechanical energy is the positive constant m(Aπ/T)²/2. Zero action results from cancellation of equal integrated kinetic and potential contributions. It does not say that the mass remains still.

## Vary the path and compare action curves

Use ξ=C sin(πs)+D sin(3πs) and qε=q+εξ. Let h₁=mπ²D₁/(3200T) and h₃=mπ²D₃/(3200T). Then

$$
\delta S=h_1AC+h_3BD,\qquad
\delta^2S=h_1C^2+h_3D^2.
$$

The exact quadratic identity S(qε)=S(q)+εδS+ε²δ²S/2 holds. At the zero reference, first variation is zero in every direction even in the saddle regime. Curvature decides whether an admissible perturbation increases, leaves unchanged, or decreases action.

Read the horizontal axis of this plot as dimensionless epsilon. Read the separate trajectory plots against physical time. The selected path's instantaneous mechanical energy need not be constant when the path is not a solution; its change is not automatically a numerical conservation error.

## Three worked horizons

Choose m=1 kg, f=2, hence ω₀=π/2 s⁻¹. With T=1 s, ω₀T/π=1/2 and the zero path is a strict minimum. The test path A=1 m, B=0 has action 3π²/16 Js and is not a solution.

With T=2 s the first coefficient is zero. The same amplitude now defines a solution with zero action and energy π²/8 J. All amplitudes on this first mode are flat competitors in the fixed-endpoint problem.

With T=2.5 s the ratio is 1.25. The first-mode trial has action −9π²/160 Js, while a third-mode trial has positive action. The zero path is stationary with a saddle action, and the physical restoring force remains unchanged.

## Research extension: count all negative directions and resolve conjugate endpoints

Two displayed modes can establish a saddle without counting every direction that decreases action. To see the full picture, keep the same fixed-zero endpoints but allow all $H_0^1(0,T)$ variations. The result also explains why recovering an initial velocity from two endpoint positions can fail while the initial-value dynamics remain regular.

### Keep action and its second variation distinct

At the stationary zero path, the first variation vanishes. The second variation is twice the exact action difference:

$$
\mathcal Q[\eta]=\delta^2S[\eta]
=m\int_0^T(\dot\eta^2-\omega_0^2\eta^2)dt,\qquad
S[\eta]-S[0]=\frac12\mathcal Q[\eta],\quad
R=\frac{\omega_0T}{\pi}\ge0.
$$

The mass and horizon are positive. For a finite horizon and finite frequency, the sine expansion used earlier diagonalizes this entire quadratic form:

$$
\eta(t)=\sum_{n\ge1}c_n\sin\frac{n\pi t}{T},\qquad
\mathcal Q[\eta]=\sum_{n\ge1}h_nc_n^2,\qquad
h_n=\frac{m\pi^2}{2T}(n^2-R^2).
$$

The amplitudes $c_n$ have length units, and $h_n$ has units kg/s, giving action units to each term. This is not the instantaneous mechanical-energy spectrum. The $H_0^1$ condition ensures convergence of the derivative-weighted series.

### Count the full Morse index and the zero directions

Define the Morse index as the largest dimension of a subspace on which the second variation is negative definite. Define nullity as the dimension of the kernel of its associated bilinear form, not the set of all vectors whose quadratic value happens to be zero. Here:

$$
\operatorname{index}(\mathcal Q)=\#\{n\in\{1,2,3,\ldots\}:n<R\}
=\max(0,\lceil R\rceil-1),\qquad
\operatorname{nullity}(\mathcal Q)=
\begin{cases}1,&R\in\{1,2,3,\ldots\},\\0,&\text{otherwise}.\end{cases}
$$

The negative sine modes span a negative-definite subspace. Any subspace with one more dimension contains a nonzero variation whose coefficients in those negative modes vanish; its quadratic value is nonnegative. This proves the maximum dimension in the full space. Exactly one coefficient vanishes at a positive integer $R$, while infinitely many higher modes remain positive. Cancellation between a positive and negative term can give zero quadratic value without giving a kernel vector.

![Full Fourier curvature signs and exact oscillator Morse index](/assets/learning/analytical-action-index.en.svg)

| $R$ | Negative directions | Kernel dimension | Nature at the zero path |
|---|---|---|---|
| 1/2 | 0 | 0 | Strict minimum |
| 1 | 0 | 1 | Minimum with one zero mode |
| 3/2 | 1 | 0 | Saddle |
| 2 | 1 | 1 | Saddle with one zero mode |
| 5/2 | 2 | 0 | Saddle |
| 3 | 2 | 1 | Saddle with one zero mode |

At $R=5/2$, modes one and two are negative although the native plot shows only modes one and three. Its saddle classification is correct, but that two-mode view does not count the full negative subspace. At $R=2$, mode two is a zero mode and mode one is negative: a zero direction does not make the reference a minimum. Horizons beyond the native range are explored by the supplementary cell, not additional sliders.

### Connect a zero mode to the endpoint map

For $\omega_0>0$, a general initial position $a$ and velocity $v_0$ give:

$$
q(t)=a\cos(\omega_0t)+\frac{v_0}{\omega_0}\sin(\omega_0t),\qquad
b=a\cos(\omega_0T)+\frac{v_0}{\omega_0}\sin(\omega_0T),\quad \omega_0>0.
$$

At an integer horizon, the coefficient carrying initial velocity into the final position vanishes:

$$
\omega_0T=N\pi\ \Longrightarrow\ b=(-1)^Na,\qquad
\eta_N(t)=c_N\sin\frac{N\pi t}{T},\quad
\eta_N(0)=\eta_N(T)=0,\quad h_N=0.
$$

If $b=(-1)^Na$, every initial velocity gives an admissible classical path, with different initial-value data. If this compatibility condition fails, there is no classical path joining the specified positions in that duration. For example, at $R=1$, $a=1$ m requires $b=-1$ m; choosing $b=0$ instead gives no stationary classical path. The original zero-endpoint family is always compatible. At later integer horizons its family still has the negative directions counted above.

### Separate ill-conditioned boundary recovery from initial-value motion

Away from integer horizons, solving for initial velocity gives the exact endpoint sensitivity:

$$
v_0=\frac{\omega_0[b-a\cos(\omega_0T)]}{\sin(\omega_0T)},\qquad
\frac{\partial v_0}{\partial b}=\frac{\omega_0}{\sin(\omega_0T)},\quad
\sin(\omega_0T)\ne0.
$$

Near a conjugate horizon, a small endpoint error can produce a large inferred-velocity error. The gain has units s$^{-1}$. For $\omega_0=\pi$ rad/s and $R=0.99$, an endpoint error of $0.001$ m changes the inferred velocity by about $0.100016$ m/s; at $R=0.5$ it changes by about $0.003142$ m/s. These are deterministic inverse sensitivities, not stochastic error bars.

The complete initial-to-final state map remains invertible:

$$
\begin{pmatrix}q(T)\\\dot q(T)\end{pmatrix}
=\begin{pmatrix}\cos(\omega_0T)&\sin(\omega_0T)/\omega_0\\-\omega_0\sin(\omega_0T)&\cos(\omega_0T)\end{pmatrix}
\begin{pmatrix}a\\v_0\end{pmatrix},\qquad\det=1.
$$

At an integer horizon it is simply $(-1)^N I$. The information loss occurs when only endpoint positions are retained. For zero frequency, use the regular free-particle limit $q(T)=a+v_0T$, whose inverse gain is $1/T$. For AI boundary reconstruction, record observation type, compatibility and conditioning; a stable initial-value law does not guarantee a well-conditioned endpoint inverse.

### Execute exact classifications before numerical gains

Integer critical horizons are detected with rational arithmetic before any floating-point sine division. `None` denotes an undefined endpoint inverse, not a universal assertion that no path exists.

```python
from fractions import Fraction as Q
from math import pi, sin

def signature(ratio):
    r = Q(ratio)
    if r < 0:
        raise ValueError("Nonnegative ratio required")
    ceiling = (r.numerator+r.denominator-1)//r.denominator
    return max(0, ceiling-1), int(r > 0 and r.denominator == 1)

for r in map(Q, ['1/2', '1', '3/2', '2', '5/2', '3']):
    index, nullity = signature(r)
    signs = [Q(n*n)-r*r for n in range(1, 7)]
    assert sum(x < 0 for x in signs) == index
    assert sum(x == 0 for x in signs) == nullity
    print(r, index, nullity)

# Here omega=pi rad/s and T=r seconds; r must be positive.
for r in map(Q, ['1/2', '99/100', '1', '101/100']):
    gain = None if r.denominator == 1 else pi/sin(float(r)*pi)
    print('endpoint inverse gain:', r, gain)
# At an integer r, the endpoint inverse is undefined: compatible data
# allow a family; incompatible data allow no classical solution.
# The Fourier proof, not this finite list, establishes the full index.
```

**Research questions with answers**

- Does a zero quadratic value identify a null mode? **Answer:** No. In a saddle, positive and negative terms can cancel; a null mode lies in the kernel of the bilinear form.
- At $R=2$, is the zero path a flat minimum? **Answer:** No. Mode two is null, but mode one is negative, so it is a degenerate saddle.
- Does a singular endpoint inverse contradict unique initial-value dynamics? **Answer:** No. Specifying $(a,v_0)$ still selects one trajectory; retaining only $(a,b)$ can discard velocity information or impose incompatible data.

## Exercises with explanations

1. At the critical horizon choose A=0 and B=1. Is this path stationary? No. D₃ is positive, so the third-mode residual and action are nonzero.
2. At the saddle horizon inspect only t=0. Can the residual identify an incorrect trial? No. Both sine terms vanish there; use the whole-path residual or its mean square.
3. Set A=B=0 after choosing a long horizon. What changes? The selected path becomes a solution with zero action, but its negative and positive variation directions remain. Zero selected amplitude does not remove a saddle.
4. Double mass at fixed f and T. Action, stiffness and the residual amplitude double. Natural frequency and the curvature classification do not change. This follows from the stated parameterization of k.
5. Set f=0 and choose a nonzero sine trial. The potential vanishes, action is positive, and the trial is not the fixed-zero-endpoint free-particle solution. That solution is q=0.
6. At the critical horizon double A with B=0. Initial velocity doubles, energy quadruples, and action stays zero. The two solutions have different initial velocities, so unique initial-value evolution is respected.

## Research connection and further reading

For learned variational dynamics, an optimizer that minimizes action over an arbitrary horizon can select a trial that is not the stationary physical trajectory. Examine the Euler–Lagrange residual, admissible boundaries and the Hessian separately. Endpoint reconstruction, initial-state prediction and functional optimization answer different questions.

The sine-series argument above supplies our minimum and flat-case proof. For an independent treatment of the oscillator's time threshold, see Marco Moriconi's [Condition for minimal Harmonic Oscillator Action](https://arxiv.org/abs/2104.07475). The strict, flat and saddle distinctions in this lesson follow from the displayed quadratic form, including its equality case. General Lagrange equations and multiple-coordinate systems follow in the next topic.
