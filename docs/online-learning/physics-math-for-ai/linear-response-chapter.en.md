# Linear response: specify the experimental handle

Authoring draft, 2026-11-08. This manuscript is not yet a registered reader unit. Its finite-state example has independent symbolic, static and high-precision checks; the certified time-domain and finite-input numerical model is also prepared. The interactive reader and browser validation remain to be built.

## 1. The question a stationary picture cannot answer

Imagine people moving among three rooms. At stationarity, each room's expected population stays fixed, even though people keep walking. Now turn a handle that changes one door. How much will room 1's population change?

Knowing the original populations and all original transition rates is insufficient. We must also know how the handle changes those rates. Two handles can change the forward/backward rate ratio by the same amount while changing overall traffic differently. Under a circulating background current, their responses can even have opposite signs.

This is an ensemble experiment: population means probabilities across repeated realizations. A single person's jumping trajectory does not equal a smooth response curve. We initialize every experiment in the original stationary distribution and change only the specified handle.

## 2. Fix rows, columns, directions and units

Let a,b,c be the clockwise rates 0→1, 1→2, 2→0, and d,e,f the reverse rates 1→0, 2→1, 0→2. All six rates are strictly positive, with units s⁻¹. A row probability vector evolves as

$$
\dot p=pQ,\qquad Q=\begin{pmatrix}-a-f&a&f\\d&-d-b&b\\c&e&-c-e\end{pmatrix},\qquad Q\mathbf1=0.
$$

The symbol f in the rate list is a scalar rate. We write the measured observable as the column vector o to avoid confusing the two. For room 1 occupancy, o=(0,1,0)ᵀ and the measured mean is po, a dimensionless probability.

The diagonal removes total outgoing probability from each state. This is a continuous-time Markov model; its rates are not probabilities per animation frame. No temperature, heat or physical local detailed balance has been specified.

## 3. Determine the background before perturbing it

The unique stationary row vector is π=τ/Z, where

$$
\tau=(cd+de+bc,\ ae+ac+ef,\ bf+ab+df),\qquad Z=\tau_0+\tau_1+\tau_2.
$$

Check πQ=0 and π1=1. On this single cycle the stationary clockwise current through every edge is

$$
J=\pi_0a-\pi_1d=\frac{abc-def}{Z}.
$$

Zero current is detailed balance for this positive three-state cycle. Equal room populations alone do not imply J=0: clockwise rates 3/s and reverse rates 1/s have uniform π and nonzero circulation.

## 4. Define the handle, not just its ratio

Use a dimensionless weak input h(t). A family of handles on the 0↔1 edge is

$$
q_{01}(h)=a e^{(\eta+1/2)h},\qquad q_{10}(h)=d e^{(\eta-1/2)h}.
$$

Other off-diagonal rates stay fixed; each diagonal is updated to keep its row sum zero. Every finite h gives positive rates. Every η gives the same derivative ∂h log(q01/q10)=1, but ∂h log√(q01q10)=η. Thus η changes the symmetric traffic factor independently of the ratio change.

The rate family is nonlinear in h. Linear response means differentiating at h=0, with η held fixed. It does not mean replacing every finite-amplitude experiment by a linear law. Our input is bounded and weak; inserting a finite Dirac distribution inside these exponential rates is not an impulse prescription.

## 5. Differentiate the probability evolution

Write Q(h)=Q+hQ₁+O(h²). Only rows 0 and 1 of Q₁ are nonzero:

$$
Q_1=\begin{pmatrix}-a(\eta+1/2)&a(\eta+1/2)&0\\d(\eta-1/2)&-d(\eta-1/2)&0\\0&0&0\end{pmatrix}.
$$

To define an approximation precisely, take h(t)=εu(t), with fixed bounded u and ε→0 on a fixed finite interval. Expanding p(t)=π+εδp(t)+O(ε²), initialized with δp(0)=0, gives

$$
\dot{\delta p}=\delta pQ+u(t)\pi Q_1,\qquad
\delta p(t)=\int_0^t u(s)\pi Q_1e^{Q(t-s)}\,ds.
$$

The row s=πQ₁ is a signed probability source per unit time. Its entries sum to zero, so the first-order perturbation preserves total probability. Individual entries need not be positive: δp is a derivative, not a probability vector.

## 6. Read causality and time order from the kernel

For a general weak h, the first-order mean change is the convolution with the retarded response:

$$
\Delta\langle o(t)\rangle=\int_0^t h(s)R_o(t-s)\,ds+O(\varepsilon^2),\qquad R_o(t)=\pi Q_1e^{Qt}o\quad(t>0).
$$

Here the remainder uses the scaling h=εu just specified. The right limit Rₒ(0⁺)=πQ₁o is an initial response rate, not an instantaneous jump in population. A step input produces zero mean change at t=0. For a dimensionless occupancy, R has units s⁻¹ and its time integral is dimensionless.

The perturbation acts first, then the process propagates, then we measure. A perturbation at a future time cannot affect an earlier observation. Swapping matrix order or time labels changes the proposed experiment.

## 7. Turn the response into an unperturbed correlation

Because every πⱼ is positive, define Bⱼ=sⱼ/πⱼ. Conditional evolution from initial state j then gives

$$
R_o(t)=\sum_j\pi_j B_j(e^{Qt}o)_j
=\mathbb E_\pi[B(X_0)o(X_t)],\qquad \mathbb E_\pi B=0.
$$

This uses the unperturbed stationary process, but B depends on the handle through Q₁. An ordinary occupancy autocorrelation without the correct conjugate observable does not generally determine the response. Here B has units s⁻¹; it is not a dimensionless thermometer or an unspecified stationary density score.

## 8. See how circulation exposes the handle

Let F01=π₀a and F10=π₁d be the two directional stationary flows. Direct multiplication gives

$$
s(\eta)-s(0)=\eta J(-1,1,0),\qquad
R_o(0^+)=\tfrac12(F01+F10)+\eta J
$$

for room 1 occupancy. At detailed balance J=0, so this particular kinetic variation leaves the whole first-order source, and therefore every observable's response, unchanged. This statement concerns this handle family, not arbitrary perturbations.

For a driven background the source can change. An observable can still be insensitive: a constant observable has zero response because s1=0. Nonzero circulation permits protocol dependence; it does not require every measurement to detect it.

## 9. Work through a sign-changing example

Choose (a,b,c,d,e,f)=(2,3,4,1,1,1)/s. Then π=(17,11,10)/38, J=23/38/s and πo=11/38. Three handles on this same background give:

| η | Initial kernel Rₒ(0⁺), s⁻¹ | Static susceptibility |
|---|---:|---:|
| −1 | −1/76 | −3/1444 |
| 0 | 45/76 | 135/1444 |
| 1 | 91/76 | 273/1444 |

For η=−1, increasing h decreases both rates on the chosen edge. Forward traffic starts larger, so reducing its rate can initially lower room 1 occupancy despite increasing the forward/backward ratio. The negative kernel is meaningful; clipping it to zero would erase the phenomenon.

These are derivatives at zero input. For a finite step h₀, the first-order prediction is πo+h₀χₒ(t), while the actual occupancy is π exp[Q(h₀)t]o. Positivity belongs to the latter exact Markov evolution; a tangent extrapolated far from zero can leave [0,1] and must not be relabeled as a physical probability.

## 10. Separate a transient step from static response

Define χₒ(t)=∫₀ᵗRₒ(s)ds. The finite positive chain mixes, so its static limit can be found by differentiating the new stationary distribution:

$$
\pi'Q=-\pi Q_1,\qquad \pi'\mathbf1=0,\qquad
\chi_o(\infty)=\pi'o.
$$

At t=0, χ=0 even when the initial kernel is nonzero. A kernel may change sign, so step susceptibility need not be monotone. Also distinguish π′, a derivative with respect to the handle, from the time derivative ṗ.

The identity Q·1=0 makes Q singular. Solve a normalized linear system rather than taking Q⁻¹. An equivalent route is Qg=−(o−(πo)1), πg=0, followed by χₒ(∞)=sg. Centering removes the stationary mode and the normalization fixes the additive constant in g.

## 11. Interpret oscillatory experiments with a phase convention

For a long-running weak sinusoid h(t)=Re[h₀exp(iΩt)], define the retarded susceptibility for Ω≠0 by

$$
\chi_o(\Omega)=\int_0^\infty R_o(t)e^{-i\Omega t}\,dt
=s(i\Omega I-Q)^{-1}o.
$$

The asymptotic mean change is Re[h₀χₒ(Ω)exp(iΩt)]. Its amplitude is |h₀||χ| and its phase relative to the input is argχ. A negative static susceptibility represents an opposite-sign response; a phase plot must not silently replace it by a positive scalar.

Ω is angular frequency in rad/s, not cycles/s. At Ω=0 the full resolvent is singular; use the centered static construction above. The cancellation s1=0 removes the stationary contribution to the physical limit, but does not make Q invertible. A sinusoid switched on at t=0 also contains a transient before the asymptotic response.

## 12. Design comparisons that answer different questions

| Display | Hold fixed | What to compare |
|---|---|---|
| Protocol comparison | Q, π, measured occupancy | η=−1,0,1 and signed kernels |
| Step experiment | η and background | Finite h₀ evolution versus its tangent |
| Static experiment | η and background | Stationary-law derivative versus time integral |
| Frequency experiment | η and background | Real/imaginary parts, amplitude and phase |

Do not use the same vertical axis for occupancy and kernel without labeling their different units. Probability paths, ensemble means and sensitivity coefficients need distinct explanations. The planned interface will keep fixed conceptual comparisons independent of saved free-experiment settings; it is not implemented by this manuscript.

## 13. Exercises with explanations

**1. Does increasing q01/q10 guarantee a positive initial response?** No. Insert η=−1 and the worked flows: (F01+F10)/2−J=45/76−46/76=−1/76/s. The ratio derivative omits changes in total traffic.

**2. Why does a constant observable not respond?** Since Q(h)1=0 for every h, Q₁1=0, and e^{Qt}1=1. Thus s e^{Qt}1=s1=0 at every lag. This is conservation, even when J≠0.

**3. Recover an equilibrium kernel and step.** With all rates 1/s, π=(1,1,1)/3 and s=(−1,1,0)/3s, independent of η. Its zero-sum mode decays at 3/s, giving Rₒ(t)=exp(−3t/s)/(3s) and χₒ(t)=[1−exp(−3t/s)]/9. The static limit is 1/9; the initial step change is zero.

**4. Predict the equilibrium sinusoidal response.** Transform the preceding kernel: χₒ(Ω)=1/[3(3+iΩs)]. Hence χ(0)=1/9 and the phase is −atan(Ωs/3). The negative phase follows from the explicitly chosen exp(iΩt) input convention.

**5. How can a finite-step numerical derivative test the tangent?** Compare [πexp(Q(ε)t)o−πexp(Q(−ε)t)o]/(2ε) with χₒ(t), using the same original π on both sides. Centered truncation error is O(ε²) for fixed t, while finite arithmetic error eventually grows as ε shrinks. Initializing each experiment in π(±ε) would instead measure a different experiment.

**6. What would justify extending this to an AI controller?** Specify the model state, transition law and how the control changes that law; choose an observable, initialization and time scale; verify weak-input convergence. A learned policy's finite intervention, hidden-state process or changing environment need not satisfy this three-state stationary model. The derivation provides a question to test, rather than a universal guarantee.

## 14. What is checked and what comes next

The independent preparation checks six backgrounds and three η values: 72 kernel/correlation comparisons, 72 integrated/block-exponential step comparisons, 72 centered finite-difference comparisons and 36 nonzero-frequency integral/resolvent comparisons at 120 decimal digits. A separate exact symbolic calculation checks 18 static Poisson/stationary-derivative pairs. These checks cover the stated examples and identities; they do not certify every parameter choice, an unbuilt UI, measured data or general nonlinear response.

This chapter's row-vector formulas and handle example are directly derived. General perturbations, causal response and conjugate observables connect to the source recorded in the [authoring plan](linear-response-authoring-plan.md). The [temporal foundation](linear-response-temporal-foundation.md) now implements numerical availability and finite-input comparison. Next development must connect these calculations to visual explanations and guided experiments, integrate both languages, then validate actual reader behavior. Nonlinear response, nonstationary preparation, spectral relations, inference from data and broader research applications remain part of the curriculum.
