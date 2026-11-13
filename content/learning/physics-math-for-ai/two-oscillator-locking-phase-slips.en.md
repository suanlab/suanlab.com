## 1. Separate the meanings of moving together

Two clock hands can move at the same speed without pointing in the same direction. Hands that align once need not keep the same speed afterward. We distinguish phase alignment, equal instantaneous frequency, equal long-time mean frequency, and stable recovery after a disturbance.

Our model assumes constant oscillator amplitudes. Whether a physical system satisfies the amplitude-recovery conditions from the Hopf chapter must be checked separately. These phase equations do not also calculate the birth or disappearance of amplitude.

## 2. Total coupling versus one edge

Time is in seconds, phase in radians, and angular frequencies and K in s⁻¹. Split total coupling K into K/2 on each directed edge.

$$
\dot\theta_1=\omega_1+\frac K2\sin(\theta_2-\theta_1),\qquad
\dot\theta_2=\omega_2+\frac K2\sin(\theta_1-\theta_2).
$$

Define δ=θ₁−θ₂,Δ=ω₁−ω₂ and Φ=(θ₁+θ₂)/2. The two coupling terms cancel from their mean.

$$
\dot\delta=\Delta-K\sin\delta,\qquad
\dot\Phi=\overline\omega=\frac{\omega_1+\omega_2}{2}.
$$

Under a different convention using edge coupling κ, the relative equation contains2κ. Confusing κ with our total K produces a factor-of-two error in the locking boundary. Supply the mean phase separately; relative phase alone does not recover common rotation.

## 3. Specify the initial circular point exactly

Use an integer half-angle pair a,b, excluding only the pair where both are zero.

$$
\cos\delta_0=\frac{b^2-a^2}{a^2+b^2},\qquad
\sin\delta_0=\frac{2ab}{a^2+b^2},\qquad
\tan\frac{\delta_0}{2}=\frac ab.
$$

The last ratio applies only when b≠0. The case b=0 represents δ₀=π. Multiplying both entries by the same nonzero factor preserves the circular point. The implementation chooses b>0 or b=0,a>0 and places the initial angle in(−π,π]. Choosing this coordinate scale does not change the physical preparation.

## 4. Lift the nonlinear equation into linear coordinates

Write the time-dependent auxiliary coordinates as B,A, initialized at b,a. Their ratio A/B obeys a Riccati equation.

$$
\frac{d}{dt}\begin{pmatrix}B\\A\end{pmatrix}
=L\begin{pmatrix}B\\A\end{pmatrix},\qquad
L=\frac12\begin{pmatrix}K&-\Delta\\\Delta&-K\end{pmatrix},\qquad
\dot u=\frac\Delta2(1+u^2)-Ku.
$$

At B=0, u has a pole but the two coordinates do not vanish together. The linear representation therefore retains the solution through a display-boundary crossing. The physical relative phase is twice the continuous direction angle of B+iA. The auxiliary vector length is not a physical oscillator amplitude.

## 5. One discriminant separates three flows

Let D=K²−Δ². Then

$$
L^2=\frac D4I,\qquad
e^{tL}=C(t)I+F(t)L,\qquad
(C,F)=
\begin{cases}
(\cosh(\gamma t/2),\,2\sinh(\gamma t/2)/\gamma),&D>0,\\
(1,t),&D=0,\\
(\cos(qt/2),\,2\sin(qt/2)/q),&D<0,
\end{cases}
\quad\gamma=\sqrt D,\ q=\sqrt{-D}.
$$

Use γ and q only in their applicable branches. At D=0, L²=0 and the exponential is exactly I+tL. Small positive and negative values are not moved into this branch using a tolerance. K=Δ=0 is a special neutral family of all relative phases, distinct from a positive-coupling critical bottleneck.

## 6. A locked trajectory still needs a stability test

For K>|Δ| there are two fixed phases on the circle.

$$
\sin\delta_* =\frac\Delta K,\qquad
\cos\delta_s=+\sqrt{1-(\Delta/K)^2},\quad
\cos\delta_u=-\sqrt{1-(\Delta/K)^2},\qquad
\lambda_s=-\gamma,\quad\lambda_u=+\gamma.
$$

Perturbations shrink near δs and grow near δu. The local recovery time is1/γ. Starting exactly at the unstable fixed phase leaves a noiseless solution there. Observing a constant phase difference is different from observing stable locking.

At a locked phase, both actual angular frequencies equal their mean. The relative phase need not be zero. When natural frequencies differ, a finite phase difference generally supplies the compensating coupling effect.

## 7. The critical bottleneck is not exponential recovery

For Δ=K>0, put δ=π/2+η.

$$
\dot\eta=K(1-\cos\eta)\sim\frac K2\eta^2.
$$

The η<0 side approaches the fixed phase slowly, while a small positive η moves away and produces a large phase excursion. A zero linear eigenvalue is not evidence of exponential recovery from both sides. Reversing detuning reverses the relevant direction.

This is a nonhyperbolic fixed-point bottleneck on a phase circle. Do not identify it with Hopf amplitude onset or an infinite-population transition. When K and |Δ| are close, locking and drifting can look similar in a finite observation window.

## 8. Slips and finite observation windows

For |Δ|>K, relative phase keeps moving in one direction. Time for one complete turn is independent of starting phase.

$$
q=\sqrt{\Delta^2-K^2},\qquad
T_{\rm turn}=\frac{2\pi}{q},\qquad
\lim_{t\to\infty}\frac{\delta(t)-\delta_0}{t}
=\operatorname{sign}(\Delta)q.
$$

Instantaneous speed Δ−Ksinδ varies within a turn. Thus the finite-window mean[δ(t)−δ₀]/t need not equal its long-time limit. The finite-window mean is unavailable at t=0. Near threshold, a turn can take so long that a positive asymptotic rate coexists with zero completed turns over the selected short window.

## 9. Separate the display branch from completed turns

Let δlift be continuous unwrapped phase, initialized in(−π,π]. Define the display angle and two integer indicators as follows.

$$
\delta_{\rm wrap}=\delta_{\rm lift}-2\pi b,\qquad
b=\left\lceil\frac{\delta_{\rm lift}-\pi}{2\pi}\right\rceil,
\qquad
n_{\rm turn}=\operatorname{sign}(\Delta)
\left\lfloor\frac{qt}{2\pi}\right\rfloor\quad(|\Delta|>K).
$$

The index b records the chosen display branch. The signed nturn counts complete net2π turns since the initial relative phase. These are different indicators. Locked, critical and neutral flows do not complete a turn from their initial phase in finite time, so nturn=0. A large transient phase excursion can still fail to complete a turn under this convention.

The interval includes+π and excludes−π. Exactly δlift=−π therefore gives b=−1 and displayed angle+π. Known exact critical boundary points preserve this convention. If interval evidence cannot resolve an integer boundary, provide unresolved bounds rather than round to a count.

## 10. Alignment and frequency entrainment are different observations

Compare the first and second harmonic order parameters of two oscillators.

$$
R_1^2=\frac{1+\cos\delta}{2},\qquad
R_2^2=\cos^2\delta.
$$

At δ=π, R₁=0 and R₂=1. With identical natural frequencies and K=0, this antiphase pair keeps moving at equal speed. With K>0 and identical frequencies, antiphase is an unstable fixed phase. Both cases have equal frequencies but different stability.

Conversely, δ=0 gives R₁=1 at that instant, while its relative speed is Δ if natural frequencies differ. The most aligned picture does not prove frequency locking. Already with two oscillators, one order statistic loses information about the underlying state.

## 11. Two examples you can calculate by hand

First choose ω₁=3,ω₂=0,K=5. Compare the stable pair a=1,b=3 with the unstable pair a=3,b=1.

| Initial state | sinδ | cosδ | R₁² | Linear growth rate [s⁻¹] |
| --- | --- | --- | --- | --- |
| Stable fixed phase | 3/5 | 4/5 | 9/10 | −4 |
| Unstable fixed phase | 3/5 | −4/5 | 1/10 | +4 |

Both actual angular frequencies are1.5rad/s. Stable recovery has local time1/4s. One fixed trajectory alone does not establish its stability.

For the second example choose Δ=K=1,a=2,b=1 and observe for4s. Starting auxiliary coordinates at(B,A)=(1,2) gives B=1−t/2,A=2−t/2. The initial relative phase is2arctan(2).

| t [s] | δlift | Display angle | b | Completed relative turns |
| --- | --- | --- | --- | --- |
| 0 | 2arctan(2) | 2arctan(2) | 0 | 0 |
| 2 | π | π | 0 | 0 |
| 4 | 2π | 0 | 1 | 0 |

The final phases align, but relative speed is1rad/s. A display boundary has been crossed without completing2π from the initial phase. The reversed critical example gives displayed+π and branch−1 at exactly−π.

## 12. Recording an exact flow is not time integration

The production calculation evaluates a real2×2 exponential using directed Taylor sums, an explicit tail bound and scaling/squaring. The critical I+tL and exact fixed phases have separate algebraic branches. Dividing homogeneous coordinates by their norm is a coordinate conversion, not projection of a physical amplitude trajectory onto a circle.

Continuous phase is recovered from local atan changes between exact-flow points. First enforce a speed bound of at most1/2rad of relative phase per subinterval. This partition does not approximate the differential equation by time steps. Display-record density changes neither that partition nor the physical endpoint. Independent enclosures check det(exp(tL))−1 and the circular residual.

The independent reference uses a general220-digit matrix exponential, analytic turn periods and a principal argument to recover the lift. It does not reproduce the production local-angle sum. Selected cases also use independent DOP853 integration of the scalar phase equation; this floating integration-error check is not called a rigorous enclosure.

## Research extension: what a long phase record actually samples

### Slow regions occupy more of the record

A runner who moves slowly through one bend of a track appears in more time-stamped photographs of that bend. The same effect occurs in a drifting relative phase. Uniform observation times do not produce uniformly distributed angles. A peak in an angle histogram can therefore come from deterministic slowing, even when no noise has been specified.

First take Δ>0 and 0≤K<Δ. The relative speed f(δ)=Δ−K sinδ is positive everywhere. Put q=sqrt(Δ²−K²). Separating time and angle gives the duration of one complete turn.

$$
T_{\rm turn}=\int_0^{2\pi}\frac{d\delta}{\Delta-K\sin\delta}
=\frac{2\pi}{q},\qquad q=\sqrt{\Delta^2-K^2}.
$$

To evaluate the integral, use u=tan(δ/2), splitting at the pole δ=π. Completing the square in Δu²−2Ku+Δ gives q in the denominator of the resulting arctangent. The pole is a coordinate boundary, not a stoppage of the physical phase.

The fraction of a long record spent in a short angle interval is its traversal time divided by the turn duration. This gives a normalized time occupation density p. Its constant circular flux is j; j has units of turns per second, while q is an angular speed.

$$
p(\delta)=\frac{1}{T_{\rm turn} f(\delta)}
=\frac{q}{2\pi(\Delta-K\sin\delta)},\qquad
f(\delta)p(\delta)=j=\frac{q}{2\pi},\qquad
-\partial_\delta(fp)=0.
$$

For a deterministic initial phase, the state at a specified time is still one point. The density describes time occupation, or an ensemble initialized with density p. It does not create a noisy trajectory. A long record sampled at a fixed discrete interval can also alias the turn; the formula is for continuous-time occupation, approached by adequately resolved sampling, rather than every possible discrete sampling schedule.

### Calculate averages before interpreting a histogram

Integrating a function against p gives its long-time average. Averaging the phase equation gives Δ−K〈sinδ〉=q. The cosine average vanishes because cosδ/f(δ) is proportional to the derivative of ln f over a full period. Rationalize the sine expression to keep its K=0 limit explicit.

$$
\langle\sin\delta\rangle_p=\frac{\Delta-q}{K}
=\frac{K}{\Delta+q}\quad(K>0),\qquad
\langle\sin\delta\rangle_p=0\quad(K=0),\qquad
\langle\cos\delta\rangle_p=0.
$$

The sine average is positive because the phase spends more time where the positive sine reduces its speed. The zero cosine average does not mean the density is uniform. It also implies that the time average of R₁²=(1+cosδ)/2 equals 1/2 throughout this drifting regime. A single averaged order statistic misses how concentrated the phase occupation becomes.

For an easily measured summary, count the fraction of continuous observation time spent in the half-circle 0<δ<π. The same half-angle integral gives

$$
P_{+}=\int_0^\pi p(\delta)\,d\delta
=\frac12+\frac1\pi\arctan\frac{K}{q}.
$$

At K=0 this is 1/2. As K approaches Δ from below, it tends to 1 because the bottleneck lies at δ=π/2. The endpoints have zero occupation measure here. Threshold behavior does not make this formula applicable to a locked or critical trajectory: there f has zeros and a finite turn duration is unavailable.

### Near threshold, observation time becomes a resource

Differentiate the period at fixed positive Δ. The resulting sensitivity measures how strongly a period estimate changes under a coupling change; it is not an uncertainty interval for measured data.

$$
\frac{\partial T_{\rm turn}}{\partial K}=\frac{2\pi K}{q^3},\qquad
K\frac{\partial\ln T_{\rm turn}}{\partial K}
=\frac{K^2}{\Delta^2-K^2}.
$$

Thus a long apparent pause can coexist with nonzero asymptotic drift. For Δ=1 and K=0.96, q=0.28 and one full turn takes about 22.44 seconds, beyond the reader's 4-second experiment limit. These longer-duration statistics are standalone calculations; changing the number of display records does not extend that limit or physical observation time.

We can also bound the difference between finite-window and asymptotic angular speed. Define a continuously lifted angle coordinate ψ by accumulated traversal time. It increases uniformly, although the original angle does not.

$$
\psi(\delta)=q\int_0^\delta\frac{du}{f(u)},\qquad
\psi(\delta+2\pi)=\psi(\delta)+2\pi,\qquad
\dot\psi=q,\qquad H(\delta)=\psi(\delta)-\delta.
$$

H is periodic. Since ψ is a strictly increasing circle map, its difference from δ has oscillation less than 2π. Subtract the endpoint identities for ψ to get a finite-window statement, using the unwrapped physical phase.

$$
\delta(T)-\delta(0)-qT=H(\delta(0))-H(\delta(T)),\qquad
\left|\frac{\delta(T)-\delta(0)}T-q\right|<\frac{2\pi}{T}\quad(T>0).
$$

This conservative bound concerns deterministic window bias, not random sampling variance. It can be too loose to classify a short record. At an integer number of full turns the endpoint H values coincide and the window mean equals q exactly. At K=0 the speed is constant for every window.

For negative detuning reflect the angle. The period remains positive and the direction of flux reverses.

$$
\widetilde\delta=-\delta,\qquad
\dot{\widetilde\delta}=|\Delta|-K\sin\widetilde\delta,\qquad
p(\delta)=\frac{\sqrt{\Delta^2-K^2}}{2\pi|\Delta-K\sin\delta|}
\quad(\Delta<-K).
$$

There is no normalized density of this positive-flux form at |Δ|=K>0: traversal time diverges at the fixed phase. The case Δ=K=0 is a separate neutral family. It has no turn flux and does not select a unique stationary distribution.

### Reproduce the occupation and period statistics

Set Δ=1 s⁻¹. The Python code computes continuous-time statistics from the analytic integrals. It does not generate independent random observations or simulate an additional stochastic model.

![Continuous-time phase occupation peaks at the deterministic bottleneck while the full-turn period grows near locking](/assets/learning/synchronization-statistics-research-slip-occupation.en.svg)

The upper panel shows normalized time occupation densities for three subthreshold couplings. The peak near π/2 marks slow motion. The lower panel shows turn duration at fixed Δ=1. All curves remain on the drifting side; the singular threshold is excluded.

```python
from math import atan, pi, sqrt

for coupling in [0.0, 0.6, 0.8, 0.96]:
    q = sqrt(1 - coupling**2)
    period = 2 * pi / q
    half = 0.5 + atan(coupling / q) / pi
    sine = coupling / (1 + q)
    print(f"{coupling:.2f} {q:.6f} {period:.6f} {half:.6f} {sine:.6f}")
```

| K | q | Turn duration | Positive-sine half occupation | Mean sine |
| --- | --- | --- | --- | --- |
| 0.00 | 1.000000 | 6.283185 | 0.500000 | 0.000000 |
| 0.60 | 0.800000 | 7.853982 | 0.704833 | 0.333333 |
| 0.80 | 0.600000 | 10.471976 | 0.795167 | 0.500000 |
| 0.96 | 0.280000 | 22.439948 | 0.909666 | 0.750000 |

### Research self-checks

#### 1. Does a peaked phase histogram prove noisy attraction?

**Research explanation.** No. In this deterministic drifting model, time occupation is proportional to reciprocal speed. The peak reflects slow passage rather than a stable fixed phase. State the observation schedule, detuning and whether any noise is actually part of the model before fitting a stochastic interpretation.

#### 2. Does mean R₁²=1/2 identify a uniform phase distribution?

**Research explanation.** No. It is 1/2 for every drifting coupling considered here because the cosine average is zero. The half-circle occupation and sine average distinguish these distributions. Averaging a representation can remove information even from a fully observed deterministic system.

#### 3. Can one replace more physical observation time with denser plotting?

**Research explanation.** No. Resolving the bottleneck improves a numerical estimate of time occupation, but a short record may still cover only part of one cycle. The window bound scales with physical T. Discrete observations synchronized with a turn can repeatedly sample one angle instead of the continuous occupation density.

[The stationary-flow discussion in Acebrón et al., section II.A](https://doi.org/10.1103/RevModPhys.77.137) provides context for separating drifting and locked phase populations. Here the density, finite-window identity and worked values are derived directly for the two-oscillator relative equation, without an infinite-population or noise assumption.

## 13. Six questions with explained solutions

**1. If each directed edge has coupling κ=2, what is the sine coefficient in the relative equation?** Solution:It is2κ=4. Our total coupling is K=4, with locking boundary |Δ|=4.

**2. Find the stable phase and local recovery time for Δ=3,K=5.** Solution:sinδ=3/5,cosδ=4/5 and γ=4. Recovery time is1/4s. The other solution with cosδ=−4/5 is unstable.

**3. Find both integer indicators at t=2 and t=4 in the critical worked example.** Solution:At t=2,δ=π gives b=0,nturn=0. At t=4,δ=2π gives b=1,nturn=0. Completed turns since the initial angle differ from a display-branch index.

**4. Find the long-time mean and turn period for Δ=−6,K=3.** Solution:q=√27, so the mean is−√27rad/s and the positive turn period is2π/√27s. A negative rate specifies direction, not a negative period.

**5. Does R₁=0 require different frequencies?** Solution:No. An identical-frequency antiphase pair can have R₁=0,R₂=1. At K=0 this is neutral; for positive K the antiphase fixed state is unstable.

**6. Can one fixed relative phase identify Δ,K and common rotation?** Solution:It restricts only Δ=Ksinδ, while common rotation cancels from the relative equation. Recovery from other phases and absolute time/phase observations add information. More display records are not new physical experiments.

## 14. Conditions of use and the next collective model

Coefficients, initial data and coupling are inputs of this specified model. We checked the two-oscillator convention against the K/N normalization in sectionII of [Acebrón et al.'s primary review](https://doi.org/10.1103/RevModPhys.77.137). The small-system solution, branch handling and integer conventions here are derived directly. Without specified reservoirs, mobility or noise, phase attraction is not automatically physical heat or entropy production.

For AI research, separate phases, velocities and stability that produce the same order observation, and design additional experiments. This model does not guarantee synchronization of an arbitrary learned network. Next we compare R₁,R₂,frequency differences and initial clusters in a finite population, separately from continuous-distribution infinite-population limits. The numerical foundation, Worker, teaching UI and reader are now registered; the research progress record links their integration evidence.
