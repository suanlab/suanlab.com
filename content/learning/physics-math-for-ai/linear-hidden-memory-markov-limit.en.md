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

## Research extension: retain the initial layer while correcting the slow mode

### Fast forgetting still leaves a displacement

A rapidly relaxing hidden variable can act briefly but move the observed state before disappearing. Keeping only the slow equation with the original observed initial value misses that displacement. Conversely, shifting the slow initial value alone fails to match the observation at time zero. A composite approximation keeps a short transient and a corrected slow trajectory together.

Use the stable or marginal family already specified. Keep a,b,c,d,x₀,y₀ and a finite observation horizon T fixed while ε tends to zero. This is essential: an initial hidden value growing as 1/ε describes a different limit. Write μ=−a+bc/d and w₀=y₀−(c/d)x₀.

Taking a Laplace transform of the two full equations, with transform variable s in a convergence half-plane, gives

$$
\widehat x(s)=\frac{(\varepsilon s+d)x_0+\varepsilon b y_0}
{(s+a)(\varepsilon s+d)-bc}.
$$

The two poles retain slow and fast dynamics. Dropping every ε term before examining initialization recovers the original Markov comparison but removes the short contribution from y₀. This transform is a derivation tool, not a new assumption that the hidden preparation is random.

### Move one derivative out of the convolution

Integrate the memory integral by parts using the actual differentiable trajectory. The endpoint at s=0 combines with the original hidden forcing. Here the integration variable is u, distinct from the Laplace variable above.

$$
\dot x(t)=\mu x(t)+bw_0e^{-dt/\varepsilon}
-\frac{bc}{d}\int_0^t e^{-d(t-u)/\varepsilon}\dot x(u)\,du.
$$

This identity is exact. Replacing the final short convolution by (ε/d) times the current derivative motivates a first correction to the slow rate, but does not by itself justify ignoring the initial layer. The expansion has to account for both poles and their amplitudes.

For sufficiently small positive ε the two real eigenvalues are distinct. Label the bounded one λₛ and the fast one λf; subscripts distinguish them in formulas. They solve the characteristic equation

$$
\varepsilon\lambda^2+(d+\varepsilon a)\lambda+ad-bc=0,\qquad
\lambda_f=-\frac d\varepsilon-a-\lambda_s.
$$

The discriminant tends to d²>0 as ε tends to zero. Thus this small-ε branch is well defined even when a larger allowed ε produces damped rotation. Substitute a regular series for λₛ into the polynomial. The first correction follows from the terms proportional to ε.

$$
\lambda_s=\mu-\varepsilon\frac{bc}{d^2}\mu+O(\varepsilon^2),\qquad
\lambda_{\rm corr}=\frac{\mu}{1+\varepsilon bc/d^2}.
$$

The rational corrected rate matches those first two terms. Its denominator must remain positive; stability of the full system at an arbitrary ε does not guarantee that. With fixed coefficients it is positive for all sufficiently small ε. This formula is not a replacement for the reader's exact full flow.

### Match the initial value and derivative

Write the exact observation as a sum of its two modes. The full initial derivative is −ax₀+by₀=μx₀+bw₀. Solving the two initial equations gives the slow amplitude and its expansion.

$$
x(t)=A_s e^{\lambda_st}+A_f e^{\lambda_ft},\qquad
A_s=\frac{\dot x(0)-\lambda_fx_0}{\lambda_s-\lambda_f}
=x_0+\varepsilon\frac{bw_0}{d}+O(\varepsilon^2),\qquad A_f=x_0-A_s.
$$

The slow trajectory therefore starts with a small slip εbw₀/d. It describes the state after the fast layer, rather than an instantaneous jump of the exact solution. Restore a compensating fast exponential to construct an approximation valid from time zero.

$$
x_{\rm comp}(t)=\left(x_0+\varepsilon\frac{bw_0}{d}\right)e^{\lambda_{\rm corr}t}
-\varepsilon\frac{bw_0}{d}e^{-dt/\varepsilon},\qquad
x_{\rm comp}(0)=x_0.
$$

Setting w₀=0 removes this leading initial slip but generally leaves the slow-rate correction. Initially quasisteady and exactly invariant remain different conditions. When b=0 the observed equation is already autonomous. When c=0, memory vanishes but a nonzero y₀ can still create a fast forcing and an initial slip.

### Why the composite error has a smaller order

For fixed coefficients the slow eigenvalue and amplitude have ordinary Taylor expansions near ε=0. The fast amplitude is O(ε), while λf+d/ε remains bounded. Once λf≤−d/(2ε), the mean value theorem for exponentials bounds the fast-exponential replacement uniformly in time.

$$
\sup_{t\geq0}|e^{\lambda_ft}-e^{-dt/\varepsilon}|
\leq\frac{2\varepsilon}{ed}\left|\lambda_f+\frac d\varepsilon\right|.
$$

Multiplying this O(ε) discrepancy by the O(ε) fast amplitude gives O(ε²). The slow-rate and amplitude remainders also contribute O(ε²) on a fixed finite horizon. The Markov comparison without these corrections has only the generic first-order guarantee in this limit.

$$
\sup_{0\leq t\leq T}|x(t)-x_{\rm comp}(t)|=O(\varepsilon^2),\qquad
\sup_{0\leq t\leq T}|x(t)-x_0e^{\mu t}|=O(\varepsilon).
$$

The constants depend on the fixed coefficients, preparation and horizon. These statements describe ε→0, not certified error bars for every slider setting. Degenerate leading coefficients can give smaller errors. Letting T or the hidden preparation grow with 1/ε needs a separate analysis; this finite-horizon argument does not establish that larger limit.

### Reproduce a sampled error comparison

Choose a=2,b=1,c=2,d=3,x₀=1,y₀=0 and T=4 seconds. The code evaluates the exact two-mode expression, the original Markov comparison and the composite approximation. It samples both a uniform physical-time grid and a grid scaled to the fast layer. The reported maxima are over these finite grids and are not rigorous suprema.

![Observed initial layer and sampled error decrease for the Markov and corrected composite approximations](/assets/learning/coarse-predictive-research-initial-layer.en.svg)

The upper panel resolves the beginning of the ε=1/4 trajectory. The composite matches x(0) while retaining the slow displacement caused by the hidden preparation. The lower panel compares sampled errors at four decreasing ε values. The asymptotic argument above, rather than a fitted slope alone, explains the different error orders.

```python
from math import exp, sqrt

for eps in [0.25, 0.125, 0.0625, 0.03125]:
    a, b, c, d, x0, y0 = 2.0, 1.0, 2.0, 3.0, 1.0, 0.0
    mu = -a + b*c/d
    w0 = y0 - c*x0/d
    disc = sqrt((d-eps*a)**2 + 4*eps*b*c)
    slow = -2*(a*d-b*c)/(d+eps*a+disc)
    fast = -(d+eps*a)/eps - slow
    amp_slow = (-a*x0+b*y0-fast*x0)/(slow-fast)
    amp_fast = x0 - amp_slow
    corrected_rate = mu/(1+eps*b*c/d**2)
    slip = eps*b*w0/d
    times = sorted(set([4*i/4000 for i in range(4001)] +
                       [eps*u/(10*d) for u in range(401) if eps*u/(10*d) <= 4]))
    markov_error = composite_error = 0.0
    for t in times:
        exact = amp_slow*exp(slow*t) + amp_fast*exp(fast*t)
        markov = x0*exp(mu*t)
        composite = (x0+slip)*exp(corrected_rate*t) - slip*exp(-d*t/eps)
        markov_error = max(markov_error, abs(exact-markov))
        composite_error = max(composite_error, abs(exact-composite))
    print(f"{eps:.5f} {slow:.6f} {markov_error:.8f} {composite_error:.8f} {markov_error/composite_error:.2f}")
```

| ε | Slow eigenvalue | Sampled Markov error | Sampled composite error | Error ratio |
| --- | --- | --- | --- | --- |
| 0.25000 | -1.255437 | 0.03448273 | 0.00637530 | 5.41 |
| 0.12500 | -1.295300 | 0.02056599 | 0.00186050 | 11.05 |
| 0.06250 | -1.314561 | 0.01155806 | 0.00051199 | 22.57 |
| 0.03125 | -1.324010 | 0.00622477 | 0.00013552 | 45.93 |

### Research self-checks

#### 1. Why not start the corrected slow equation at the slipped value alone?

**Research explanation.** That captures a later slow amplitude but misses x(0) by the slip. The compensating fast term restores the supplied initial value and decays across the initial layer. The exact trajectory remains continuous.

#### 2. Does w₀=0 make the original Markov comparison exact?

**Research explanation.** Generally no. It removes the leading initial-slip amplitude but not the difference between λₛ and μ. The full equations can generate a later hidden mismatch. Exact closure or an invariant preparation requires additional conditions.

#### 3. Does an observed second-order error trend prove the bound for every input?

**Research explanation.** No. A finite grid gives sampled errors for its stated coefficients and preparation. The expansion and fast-exponential estimate establish an asymptotic statement with fixed data and finite horizon. Large ε, changing preparation scales or an invalid corrected denominator need separate treatment.

[The linear elimination in Gouasmi, Parish and Duraisamy, section 2](https://arxiv.org/pdf/1611.06277) supplies the memory-equation context. The pole expansion, initial slip, composite approximation and error argument here are direct calculations for this two-variable linear family. They add standalone Python examples rather than a new approximation control in the existing reader.

## 14. Exercises with explanations

**1. Identify the eliminated terms.** Take a=2,b=1,c=2,d=3,epsilon 1 and zero hidden initial value. Solution: initial forcing vanishes, while the lag kernel is 2 exp(−3 u). Zero initial forcing does not remove memory.

**2. Compare two time scales.** For those coefficients, compare epsilon 1 and 1/16. Solution: hidden relaxation times are 1/3 and 1/48 seconds, while both Markov rates are−4/3 inverse seconds. Actual slow eigenvalues differ as shown in the table, so identical comparison trajectories do not imply identical full trajectories.

**3. Hidden preparation without memory.** Set a=d=epsilon=b=1,c=0 and initial values x=0,y=1. Solution: hidden state is exp(−t), observation is t exp(−t), and the Markov observation is zero. Initial forcing produces the observed motion.

**4. A stable oscillatory solution.** Set a=d=epsilon=1,b=2,c=−2 and initial values x=1,y=0. Solution: observed state is exp(−t)cos(2 t), hidden state is−exp(−t)sin(2 t). The Markov comparison is exp(−5 t), a different trajectory. Gamma is−3, so the conditional estimate requiring positive Gamma does not apply.

**5. Marginality and preparation.** With all five positive model coefficients equal to one, compare initial values (1,1) and (1,0). Solution: the first state is constant. The second has observed state (1+exp(−2 t))/2 and hidden state (1−exp(−2 t))/2. Both Markov observations remain one, so hidden preparation influences even the long-time result.

**6. A tiny positive initial term.** For unit b and hidden preparation, d=8,epsilon 10⁻⁹ and time 4 seconds, is the forcing exactly zero? Solution: no. Its analytic sign is positive and its log relative to one inverse second is−32 billion. Inspect the unresolved enclosure with the separately retained log and sign rather than declaring an unavailable display value zero.
