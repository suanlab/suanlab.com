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

## Research extension: path scores, information and control sensitivity

### Include waiting times in the control score

The current reader implements independent fixed comparisons and a saved free experiment. This extension adds a standalone calculation of the control information carried by complete paths; it adds no new lab controls. Keep the six positive rates and the exponential handle from the chapter. Apply a constant small parameter h over a fixed T>0, holding eta fixed. Initialize every parameter value in the same original stationary law pi, not in a new pi(h).

Differentiate the full path density, including the final censored wait. Let gij be the derivative of a jump's log rate. The complete-path score is

$$
g_{01}=\eta+\tfrac12,\quad g_{10}=\eta-\tfrac12,\quad g_{ij}=0\ \text{otherwise},\qquad S_T=\left.\partial_h\log w_h(\Gamma)\right|_{h=0}=\sum_{\text{jumps }i\to j}g_{ij}-\int_0^T\sum_{j\ne X_t}q_{X_tj}g_{X_tj}\,dt.
$$

The jump sum rewards rate changes on observed transitions. The integral subtracts changes in the hazard during all exposure time, including time with no jump. This compensator is essential: a record containing no departures still carries control information. The [Rao–Teh path density, Section 2, equation(2)](https://jmlr.org/papers/volume14/rao13a/rao13a.pdf) provides the rate-product and final-survival factors; our row convention is the transpose of its column convention.

### Derive Fisher information from compensated counts

Nij counts actual i→j jumps. Subtracting its integrated conditional intensity gives a counting-process martingale Mij. Different directed jumps cannot occur simultaneously, so their martingales have zero cross quadratic covariation. Under the unperturbed stationary process,

$$
M_{ij}(T)=N_{ij}(T)-\int_0^T\mathbf1_{X_t=i}q_{ij}\,dt,\quad S_T=\sum_{i\ne j}g_{ij}M_{ij}(T),\quad\mathbb E S_T=0,\quad\mathcal I_T=\mathbb E S_T^2=T\sum_{i\ne j}\pi_iq_{ij}g_{ij}^2.
$$

The squared-jump bracket has expected rate pi_i qij gij². Integrating it gives the second moment of the score. h and S are dimensionless; information per second has units s⁻¹. Counts on different edges need not be independent for this argument. Virtual self events in a uniformization representation must not be treated as extra physical departures in this score.

If the initialization depended on h, an initial-law score would also appear. The formula above therefore belongs to the fixed-initialization experiment, not to a different experiment equilibrated separately at each parameter value.

### Relate the path score to the measured response

Differentiate the normalized path expectation. The bounded terminal observable and finite positive rate family allow the derivative at zero to pass through the fixed finite-time expectation:

$$
\chi_o(T)=\left.\partial_h\mathbb E_h[o(X_T)]\right|_0=\mathbb E_0[o(X_T)S_T]=\operatorname{Cov}_0(o(X_T),S_T).
$$

Here chi_o(T) is the constant-step susceptibility already defined in the chapter, not the impulse kernel at lag T. The [Seifert review, Section 8.2.2, equations(184)–(190)](https://arxiv.org/pdf/1205.4176) explains the path-weight and state-correlation response forms. Integrating its path score over the constant control interval gives this experiment's representation.

Cauchy–Schwarz supplies an information bound:

$$
|\chi_o(T)|^2\leq\operatorname{Var}_\pi(o)\,\mathcal I_T,\qquad o=\mathbf1_{X=1}:\quad\operatorname{Var}_\pi(o)=\pi_1(1-\pi_1).
$$

For room 1 occupancy the terminal variance is the unperturbed stationary Bernoulli variance. The response can be small although the full path contains substantial information: a terminal state ignores intervening jumps and waits. This is a parameter-information bound. Substituting entropy production for Fisher information would require additional results and assumptions.

### Compute score moments without simulating trajectories

A useful independent check attaches a real tilt z to S. A short interval contributes exp(z gij) on a jump and exp(−z times its compensator) during waiting. This yields a positive, generally nonconservative tilted matrix:

$$
(K_z)_{ij}=q_{ij}e^{zg_{ij}}\ (i\ne j),\quad(K_z)_{ii}=q_{ii}-z\sum_{j\ne i}q_{ij}g_{ij},\qquad\mathbb E_0[e^{zS_T}o(X_T)]=\pi e^{TK_z}o.
$$

It is not a Markov generator for nonzero z. Its first and second derivatives are

$$
K_0=Q,\quad K_1:=\partial_zK_z|_0=Q_1,\quad(K_2)_{ij}:=(\partial_z^2K_z|_0)_{ij}=q_{ij}g_{ij}^2\ (i\ne j),\quad(K_2)_{ii}=0.
$$

K2 is not Q''(0): the latter also differentiates the escape-rate diagonal, whereas this matrix tilts the already-defined score. Differentiating pi exp(tKz) at z=0 gives moment rows u1_j=E[S_t 1(X_t=j)] and u2_j=E[S_t² 1(X_t=j)]:

$$
\dot u_1=u_1Q+\pi Q_1,\quad\dot u_2=u_2Q+2u_1Q_1+\pi K_2,\quad u_1(0)=u_2(0)=0,\qquad u_1(T)o=\chi_o(T),\quad u_2(T)\mathbf1=\mathcal I_T.
$$

The first equation is exactly the probability sensitivity equation. Multiplying the second by the all-ones column makes the Q and Q1 terms vanish, leaving the Fisher-information rate. This simultaneously checks the score covariance and conservation without inventing sampled trajectories.

### Compare control handles with the same rate-ratio derivative

Let Fplus and Fminus denote the two original directional edge flows. The rate-ratio derivative remains one for every eta, while path information varies:

$$
F_+=\pi_0a,\quad F_-=\pi_1d,\quad H=F_++F_-,\quad J=F_+-F_-,\qquad\frac{\mathcal I_T}{T}=F_+(\eta+\tfrac12)^2+F_-(\eta-\tfrac12)^2,\quad\eta_*=-\frac J{2H},\quad\min_\eta\frac{\mathcal I_T}{T}=\frac{F_+F_-}{H}.
$$

Differentiate the positive quadratic to find eta_star. For the worked background (2,3,4,1,1,1)/s, Fplus=34/38s⁻¹ and Fminus=11/38s⁻¹, so eta_star=−23/90 and the minimum rate is187/855s⁻¹. This minimizes path distinguishability for the specified ratio change; it does not maximize a chosen occupancy response or establish an optimal physical actuator.

![Control-path Fisher information has a minimizing kinetic handle, while static occupancy response follows a different linear trend.](/assets/learning/response-research-control-path-information.en.svg)

The upper panel shows the information-rate quadratic and its minimizing handle. The lower panel shows the original static room 1 susceptibility. A low-information handle and a large-response handle answer different design questions. In equilibrium the first-order source can be independent of eta even while path information changes through traffic.

```python
from fractions import Fraction as F

forward_flow, backward_flow = F(34, 38), F(11, 38)
traffic = forward_flow + backward_flow
current = forward_flow - backward_flow
eta_min = -current / (2 * traffic)
for eta in [F(-1), F(0), eta_min, F(1)]:
    information_rate = forward_flow * (eta + F(1, 2))**2
    information_rate += backward_flow * (eta - F(1, 2))**2
    initial_kernel = traffic / 2 + eta * current
    static_susceptibility = (135 + 138 * eta) / 1444
    print(eta, information_rate, initial_kernel, static_susceptibility)
```

All entries are exact fractions. Columns list eta, Fisher information per second, the initial room 1 response kernel per second and its static susceptibility. The static formula is the normalized stationary derivative derived earlier; the information rate comes from complete paths.

| eta | I/T [1/s] | R(0+) [1/s] | chi(infinity) |
| --- | --- | --- | --- |
| -1 | 7/8 | -1/76 | -3/1444 |
| 0 | 45/152 | 45/76 | 135/1444 |
| -23/90 | 187/855 | 374/855 | 374/5415 |
| 1 | 317/152 | 91/76 | 273/1444 |

### A score estimator needs its own variance analysis

For any fixed baseline beta, the score's zero mean preserves the response. Minimizing the estimator variance over a constant beta gives

$$
\mathbb E[(o(X_T)-\beta)S_T]=\chi_o(T),\qquad \beta_* =\frac{\mathbb E[o(X_T)S_T^2]}{\mathcal I_T}\quad(\mathcal I_T>0),\qquad\operatorname{Var}(\widehat\chi_M)=\frac{\operatorname{Var}((o(X_T)-\beta)S_T)}M.
$$

The final expression assumes M independent complete paths using the displayed baseline estimator. At I_T=0 do not use the baseline ratio. Centering by the known mean pi o is unbiased, but is not generally the variance-minimizing constant because S² reweights terminal outcomes. Estimating a baseline from the very same paths requires its own bias analysis.

Information grows linearly in T while the static susceptibility stays finite in this mixing finite chain. Hence a longer path does not automatically make a terminal-observable score estimator efficient. Data analysis must specify the observed path, any missing waits, the baseline and dependence across samples. The exact response identity alone is not a finite-data precision guarantee for an AI controller.

### Research self-checks

#### R1. Why can a no-jump record have a nonzero score?

Start in state0 and observe no jump up to T for a handle with nonzero g01. Find the score contribution from that event.

**Research explanation.** The jump sum is empty, but the waiting integral gives−a g01 T. The unchanged 0→2 rate contributes zero. A quiet record carries evidence through its survival probability; counting only jumps would discard that evidence. This is a conditional path example, not the unconditional mean score, which is zero.

#### R2. Does minimum information mean maximum static response?

Compare eta=−23/90 and eta=1 in the worked background.

**Research explanation.** The first minimizes the positive information quadratic. The static susceptibility grows linearly with eta here, so the second has a larger occupancy response. Neither criterion alone defines a universal optimum; one must specify an observable, perturbation constraint and estimation or control objective.

#### R3. Why does the tilted second derivative have a different diagonal?

Compare K2 with Q''(0) for the exponential rate handle.

**Research explanation.** K2 counts squared score increments and has zero diagonal. Q''(0) is a second generator derivative and must keep row sums zero, giving diagonal−sum(qij gij²). Confusing them would make the second-moment source conserve mass and falsely predict zero Fisher information.

## 13. Exercises with explanations

**1. Does increasing q01/q10 guarantee a positive initial response?** No. Insert η=−1 and the worked flows: (F01+F10)/2−J=45/76−46/76=−1/76/s. The ratio derivative omits changes in total traffic.

**2. Why does a constant observable not respond?** Since Q(h)1=0 for every h, Q₁1=0, and e^{Qt}1=1. Thus s e^{Qt}1=s1=0 at every lag. This is conservation, even when J≠0.

**3. Recover an equilibrium kernel and step.** With all rates 1/s, π=(1,1,1)/3 and s=(−1,1,0)/3s, independent of η. Its zero-sum mode decays at 3/s, giving Rₒ(t)=exp(−3t/s)/(3s) and χₒ(t)=[1−exp(−3t/s)]/9. The static limit is 1/9; the initial step change is zero.

**4. Predict the equilibrium sinusoidal response.** Transform the preceding kernel: χₒ(Ω)=1/[3(3+iΩs)]. Hence χ(0)=1/9 and the phase is −atan(Ωs/3). The negative phase follows from the explicitly chosen exp(iΩt) input convention.

**5. How can a finite-step numerical derivative test the tangent?** Compare [πexp(Q(ε)t)o−πexp(Q(−ε)t)o]/(2ε) with χₒ(t), using the same original π on both sides. Centered truncation error is O(ε²) for fixed t, while finite arithmetic error eventually grows as ε shrinks. Initializing each experiment in π(±ε) would instead measure a different experiment.

**6. What would justify extending this to an AI controller?** Specify the model state, transition law and how the control changes that law; choose an observable, initialization and time scale; verify weak-input convergence. A learned policy's finite intervention, hidden-state process or changing environment need not satisfy this three-state stationary model. The derivation provides a question to test, rather than a universal guarantee.

## 14. Extend the question into research

This three-state family shows why the background generator alone does not determine the handle derivative. Curves are ensemble means from the original stationary law, not sampled paths. Positive uniformization, directed enclosures and explicit tails distinguish exact zeros, unresolved intervals and unavailable rate upper bounds. A conservation proof of zero differs from a positive quantity too small to display.

For a research application, specify how control changes the actual transition law, then check initialization, observable, time scale and weak-input convergence. Nonlinear/nonstationary response, hidden states, spectral relations and inference from data require further conditions. This example is not a universal response guarantee for arbitrary AI policies or measured physical systems.
