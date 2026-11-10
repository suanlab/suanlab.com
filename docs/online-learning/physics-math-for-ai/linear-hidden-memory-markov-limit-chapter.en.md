# Eliminating a hidden variable: memory, initial layers and a Markov approximation

This manuscript accompanies a numerical foundation under review. It is not yet registered in Interactive Book; browser labs and reader verification are subsequent work.

## 1. One observed value is not the full state

Two coupled boxes can display the same observed value x while their unseen component y has different preparations. Their subsequent observations can differ. Choosing not to measure a variable does not erase its influence. Just as discrete aggregation retained past states, continuous elimination retains an integral over history.

Use one real observed state and one real hidden state. This small system lets us compare elimination, approximation, stability and numerical precision before extending the reasoning. Its hidden initial value is a supplied number rather than a random noise sample.

## 2. Specify units and the model family

States and the time-scale ratio are dimensionless, time is measured in seconds, and a,b,c,d have units of inverse seconds. The two diagonal decay coefficients are positive; the coupling coefficients may have either sign.

$$
\dot x=-ax+by,\qquad \varepsilon\dot y=cx-dy,\qquad a,d,\varepsilon>0,\quad ad-bc\geq0.
$$

The final condition declares our stable or marginal family. Negative trace and nonnegative determinant exclude positive-real-part eigenvalues. Equality can leave a zero mode. We reject unstable positive-growth inputs rather than silently stabilizing them. Stability alone does not specify a physical energy, heat or entropy law.

## 3. Solve the hidden equation first

Temporarily regard the observed trajectory as supplied and apply an integrating factor to the linear hidden equation.

$$
y(t)=e^{-\kappa t}y_0+\frac c\varepsilon\int_0^t e^{-\kappa(t-s)}x(s)\,ds,\qquad \kappa=\frac d\varepsilon.
$$

The first term retains the hidden preparation. The second accumulates the effects of past observations. With the actual observed solution this identity is exact. We have neither set the hidden state to zero nor assumed immediate equilibration. A small decay rate retains history over a long window.

## 4. Three contributions remain in the observed equation

Substitution separates the current state, hidden preparation and memory.

$$
\dot x(t)=-ax(t)+F(t)+\int_0^t K(t-s)x(s)\,ds,\quad F(t)=by_0e^{-\kappa t},\quad K(u)=\frac{bc}{\varepsilon}e^{-\kappa u}.
$$

Because the initial value is specified, the initial forcing is deterministic. Calling it white noise without specifying a probability law defines a different model. The kernel can be negative for opposite-signed couplings; it is not a probability density. Its integral contribution has units of observed velocity. The memory integral starts at zero even when initial forcing does not.

## 5. Quasisteady replacement is a new approximation

If the hidden state responds sufficiently rapidly, consider replacing it by its instantaneous quasisteady value. Denote the resulting comparison trajectory by a subscript M.

$$
\dot x_M=\mu_Mx_M,\qquad \mu_M=-a+\frac{bc}{d},\qquad x_M(t)=x_0e^{\mu_Mt},\quad y_M(t)=\frac cd x_M(t).
$$

Epsilon disappears from this comparison. Changing only epsilon therefore preserves the comparison trajectory while potentially changing the actual one. Exact elimination and replacement of history by a current-state relation are distinct operations. Section 2(a) of the [Gouasmi–Parish–Duraisamy research paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC5627381/) provides the general linear hidden-variable elimination for comparison.

## 6. Initially quasisteady does not mean invariant

Define the deviation of the hidden variable from the current quasisteady relation. Its evolution follows directly from both equations.

$$
\dot w=-\Gamma w-\frac cd\mu_Mx,\qquad \Gamma=\frac d\varepsilon+\frac{bc}{d}.
$$

Even zero initial deviation is subsequently forced when the final term is nonzero. The quasisteady relation is generally not an invariant manifold. Preparing the hidden state carefully can reduce an initial layer without making the entire approximation exact. Moreover, stability of the full system does not ensure positive Gamma when the coupling product is negative.

## 7. Error identities and conditional bounds

Let the observed error be the true trajectory minus its Markov comparison. They share their observed initial value.

$$
e(t)=b\int_0^t e^{\mu_M(t-s)}w(s)\,ds,\qquad |e(t)|\leq |b|\int_0^t|w(s)|\,ds.
$$

The inequality uses the nonpositive comparison rate in this family. It still depends on the actual hidden mismatch; it is not a universal small bound computable from a parameter name alone. If a finite observation window has a known bound on observed amplitude and positive Gamma, variation of constants gives a conditional estimate.

$$
|w(t)|\leq e^{-\Gamma t}|w_0|+\frac{|c\mu_M|X}{d\Gamma}(1-e^{-\Gamma t}).
$$

With fixed coefficients, preparation and observation window, reducing epsilon increases Gamma. Changing coupling strength, preparation or the window at the same time requires a fresh argument. Resolving a short initial layer also requires observation times appropriate to that physical scale. Increasing the number of display records differs from reducing epsilon.

## 8. Compare the full eigenmodes

Separate the central decay from the drift matrix. The remaining matrix squares to a scalar multiple of the identity, allowing a flow formula without eigenvector diagonalization.

$$
M=\begin{pmatrix}-a&b\\c/\varepsilon&-d/\varepsilon\end{pmatrix},\quad \tau=\frac{a+d/\varepsilon}{2},\quad B=M+\tau I,\quad B^2=\eta I,\quad \eta=\frac{(d/\varepsilon-a)^2}{4}+\frac{bc}{\varepsilon}.
$$

Positive eta gives real modes and negative eta gives damped rotation. Its sign is determined from exact rational inputs. The table uses a=2,b=1,c=2,d=3 and initial values x=1,y=0. The comparison rate remains minus four thirds.

| ε | Hidden relaxation time ε/d [s] | Actual slow eigenvalue [s⁻¹] | Markov growth rate [s⁻¹] |
| --- | --- | --- | --- |
| 1 | 1/3 | −1 | −4/3 |
| 1/2 | 1/6 | −4+2√2 | −4/3 |
| 1/16 | 1/48 | −25+√561 | −4/3 |

Inspect whether the actual slow mode approaches the comparison rate as hidden relaxation becomes relatively faster. An isolated hidden relaxation time and the slow time of the coupled system are different quantities.

## 9. Repeated eigenvalues are legitimate inputs

When eta vanishes the centered matrix is nilpotent. It may be zero or a nonzero matrix with square zero.

$$
e^{Mt}=e^{-\tau t}(I+tB)\qquad(\eta=0).
$$

Take unit a,d,epsilon,b, zero c, and initial observed and hidden values zero and one. The true observation is t times exp of minus t; the Markov observation stays zero. The memory kernel vanishes, yet the initial hidden forcing moves the observation. A zero kernel differs from autonomous closure for every hidden preparation. In this family the latter condition is exactly zero coupling b into the observation.

## 10. Check with an independent integral state

The reference calculation adds a convolution state satisfying its own equation: its derivative is observed state minus kappa times that convolution, with zero initial value. An augmented three-dimensional matrix exponential computes the state and convolution together. Production instead uses a two-dimensional flow and a separate analytic convolution formula.

For the first row of the table, the true observation is (2 exp(−t)+exp(−4 t))/3 and the hidden state is 2(exp(−t)−exp(−4 t))/3. Substitute these states, the initial forcing and integral term into the observed equation. Directed intervals for the independently evaluated sides must differ by an interval containing zero. Containing zero does not make every value within an interval exactly zero.

## 11. Unrepresentably small does not mean zero

Very rapid decay can prevent a fixed-precision exponential enclosure from retaining a positive lower bound. Evaluate the analytic sign and log magnitude separately. The logarithm uses a reference of one inverse second to make its argument dimensionless.

$$
\log\frac{|F(t)|}{1\,\mathrm{s}^{-1}}=\log\frac{|by_0|}{1\,\mathrm{s}^{-1}}-\frac d\varepsilon t\qquad(by_0\ne0).
$$

For unit b and hidden initial value, d=8,epsilon 10⁻⁹ and time 4 seconds, the log is minus 32 billion. The forcing remains positive, but ordinary Number display and relative resolution of 512-bit fixed-point arithmetic cannot represent that magnitude. The result retains an unresolved enclosure alongside its log and analytic sign. It is distinguished from structural zero. Interval width from cancellation is retained too.

## 12. Input bounds and physical-time selection

The contract uses 17 owned integer fields with positive denominators and integer magnitudes at most 10⁹. Diagonal decay and epsilon are positive; coefficient magnitudes are at most 8,epsilon at most 4, initial magnitudes at most 4, time between 0 and 4 seconds, and record intervals between 1 and 64. The stability determinant is compared using integer arithmetic rather than rounded differences near a boundary.

The flow is not obtained by reducing an integration step. It uses real decaying modes, damped rotation or an entire-function series near coalescence, with directed 512-bit endpoints and an explicit series tail. Changing record density preserves results at the same physical endpoint. Exact stationary preparations retain that identity. Zero state, zero time, missing couplings and repeated eigenvalues receive separate checks.

## 13. What should an AI reduction validate?

The following distinctions matter when designing observed representations or latent dynamics.

| Condition | Memory kernel | Initial forcing | Valid conclusion |
| --- | --- | --- | --- |
| b=0 | 0 | 0 | Observation is autonomous for every hidden preparation |
| c=0,b≠0 | 0 | Can remain | The observed initial value alone can miss hidden forcing |
| y₀=(c/d)x₀ | Generally remains | Can remain | Initial quasisteady preparation does not ensure invariance |
| ad=bc and exact stationary preparation | Can remain | Can remain | The complete sum reconstructs a stationary solution |

A model trained on one hidden preparation can miss the initial layer of another. Success at one small epsilon does not establish accuracy for different couplings, windows or initialization. Nonlinear projection, probabilistic preparation and conditional expectation, fluctuation–dissipation relations, unstable modes, spatial renormalization and effective couplings remain separate research work. This small stable linear family does not complete that scope.

## 14. Exercises with explanations

**1. Identify the eliminated terms.** Take a=2,b=1,c=2,d=3,epsilon 1 and zero hidden initial value. Solution: initial forcing vanishes, while the lag kernel is 2 exp(−3 u). Zero initial forcing does not remove memory.

**2. Compare two time scales.** For those coefficients, compare epsilon 1 and 1/16. Solution: hidden relaxation times are 1/3 and 1/48 seconds, while both Markov rates are−4/3 inverse seconds. Actual slow eigenvalues differ as shown in the table, so identical comparison trajectories do not imply identical full trajectories.

**3. Hidden preparation without memory.** Set a=d=epsilon=b=1,c=0 and initial values x=0,y=1. Solution: hidden state is exp(−t), observation is t exp(−t), and the Markov observation is zero. Initial forcing produces the observed motion.

**4. A stable oscillatory solution.** Set a=d=epsilon=1,b=2,c=−2 and initial values x=1,y=0. Solution: observed state is exp(−t)cos(2 t), hidden state is−exp(−t)sin(2 t). The Markov comparison is exp(−5 t), a different trajectory. Gamma is−3, so the conditional estimate requiring positive Gamma does not apply.

**5. Marginality and preparation.** With all five positive model coefficients equal to one, compare initial values (1,1) and (1,0). Solution: the first state is constant. The second has observed state (1+exp(−2 t))/2 and hidden state (1−exp(−2 t))/2. Both Markov observations remain one, so hidden preparation influences even the long-time result.

**6. A tiny positive initial term.** For unit b and hidden preparation, d=8,epsilon 10⁻⁹ and time 4 seconds, is the forcing exactly zero? Solution: no. Its analytic sign is positive and its log relative to one inverse second is−32 billion. Inspect the unresolved enclosure with the separately retained log and sign rather than declaring an unavailable display value zero.
