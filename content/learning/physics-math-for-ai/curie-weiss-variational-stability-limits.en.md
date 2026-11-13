## A stationary point is a candidate, not an equilibrium answer

Imagine a bowl with two depressions and a ridge between them. Releasing a ball near either depression may keep it nearby, although only the deeper depression minimizes the height globally. Making a tiny tilt changes which depression is deeper before it visibly changes the shape. A free-energy landscape makes a related distinction between global minima, local minima and intervening maxima. The ball is an intuition for the landscape; its rolling time is not a thermodynamic prediction.

The previous chapter computed the distribution of a finite, completely connected spin system. This chapter establishes its variational limit and resolves every stationary branch. We keep both ordered minima when symmetry makes them equally favorable. We also keep an unstable root instead of allowing a numerical solver to return it as the unique answer.

You should know the binomial counting measure, finite partition functions, derivatives and local Taylor expansions. The preparation has dimensionless ferromagnetic coupling K from 0 to 8 and field b from −4 to 4 in the calculator. These are computational bounds. The mathematical derivations use finite K≥0 and finite b. Neither a temperature in kelvin nor a spatial lattice dimension is supplied by these two inputs.

## Constrain magnetization before computing entropy

Let m be the total spin divided by N. Holding m fixed leaves a fraction (1+m)/2 of up spins and (1−m)/2 of down spins. Their binary entropy per spin is s(m), in nats. Use the continuous endpoint convention 0 ln 0=0. The constrained dimensionless free energy per spin is ψ.

The negative quadratic term rewards alignment; the field term rewards a chosen sign. The negative entropy term rewards the numerous mixed configurations. At m=0 the entropy is ln 2; at either fully aligned endpoint it is zero. Draw these three contributions separately before adding them: the competition becomes easier to see than in a single final curve.

The physical Hamiltonian is E=−JM²/(2N)−hM, with K=J/(kBT) and b=h/(kBT). Thus ψ is a free energy divided by NkBT, not an energy in joules. With J,h fixed, both nonzero dimensionless parameters scale inversely with temperature; if h=0, b remains zero during a temperature scan.

$$
s(m)=-\frac{1+m}{2}\ln\frac{1+m}{2}-\frac{1-m}{2}\ln\frac{1-m}{2},\qquad \psi_{K,b}(m)=-\frac K2m^2-bm-s(m).
$$

## Why the finite counting problem converges to this landscape

For an allowed m=2k/N−1, the number of named configurations is the binomial coefficient. A useful type-counting bound controls it uniformly, including the endpoints. Write A_N for the largest value of −ψ on the allowed grid. Bounding each term of the partition sum above, and keeping its largest term below, traps the finite pressure within ln(N+1)/N of A_N.

To see the lower counting bound, choose the Bernoulli probability k/N. The count k is a mode of this binomial distribution, so its probability is at least 1/(N+1); rearranging gives the bound. The upper bound follows because that same probability cannot exceed one. At endpoints it reduces to the exact count one.

Continuity of ψ on the compact interval and the increasingly dense magnetization grids make A_N converge to the continuous maximum. This proves the limiting pressure, rather than replacing a finite sum by an unqualified saddle-point picture. It does not provide a uniform error bound for arbitrary derivatives at criticality. Dropping the Hamiltonian's constant self interaction shifts ln Z by K/2 and therefore shifts pressure per spin by K/(2N), which vanishes in this limit.

$$
\frac{e^{Ns(m)}}{N+1}\leq\binom Nk\leq e^{Ns(m)},\qquad A_N-\frac{\ln(N+1)}{N}\leq\frac{\ln Z_N}{N}\leq A_N+\frac{\ln(N+1)}{N}.
$$

$$
p(K,b)=\lim_{N\to\infty}\frac{\ln Z_N(K,b)}N=\max_{-1\leq m\leq1}\left[\frac K2m^2+bm+s(m)\right]=-\min_m\psi_{K,b}(m).
$$

## Differentiate once, then ask how many roots exist

Inside the interval, s′(m)=−atanh(m). Stationary points solve f(m)=0, equivalently the familiar self-consistency equation. The latter is useful intuition: the mean spin must agree with the response to its own effective field Km+b. It is not a complete root-finding procedure.

For finite K,b, the derivative tends to negative infinity at the left endpoint and positive infinity at the right endpoint. Neither endpoint minimizes ψ. Every global minimum is therefore an interior stationary point. A numerical bracket can safely remain strictly inside the endpoints: r=1−2^−64 satisfies atanh(r)>32 ln 2>16, while K+|b|≤12 in the supported preparation.

A fixed-point iteration m←tanh(Km+b) may converge to one minimum or fail to reach a root on a repelling branch. Returning a single converged iterate cannot establish that all stationary roots have been found. We classify monotone intervals first and isolate one root on each interval whose endpoint signs require it.

$$
f(m)=\psi_{K,b}^{\prime}(m)=\operatorname{atanh}m-Km-b,\qquad f(m)=0\ \Longleftrightarrow\ m=\tanh(Km+b).
$$

## Turning points and the spinodal are different from coexistence

When K≤1, f is increasing, with a possible isolated zero derivative at K=1,m=0. Its endpoint signs imply exactly one root. When K>1, the curvature vanishes at ±s where s=sqrt(1−1/K). The function f increases on the outer intervals and decreases between them. Define the positive threshold B_sp below.

For |b|<B_sp there are three distinct roots. For |b|>B_sp there is one. At |b|=B_sp a local minimum and the intervening maximum merge into a double stationary root, alongside a separate global minimum. This is a spinodal: loss of local stability. At b=0,K>1, the two minima instead have equal depth and remain separated. This is coexistence.

A horizontal-line visualization helps: plot atanh(m)−Km and slide the line of height b. Intersections change from three to one when the line touches a turning point. Equal depths require comparing integrals of the derivative, not counting intersections. In exact numerical inputs a threshold comparison may remain unresolved at a precision cap; the model must report this instead of guessing a double root.

$$
\psi^{\prime\prime}(m)=\frac{1}{1-m^2}-K,\qquad s=\sqrt{1-\frac1K},\qquad B_{\mathrm{sp}}=Ks-\operatorname{atanh}s\quad(K>1).
$$

## Label local stability and global selection separately

Positive curvature means a strict local minimum; negative curvature means a local maximum. With three roots, the outer two are minima and the middle one is a maximum. Positive field selects the positive outer minimum globally; negative field selects the negative one. The other outer minimum is metastable. At zero field both outer minima are global.

A symmetry argument explains the selection. The non-field part is even, so a positive m always has lower free energy than its reflected negative counterpart when b>0. The positive half interval contains only the positive outer minimum as its interior minimum. This certifies the global choice even when displayed decimal energies are identical.

| Root preparation | Local shape | Global status |
| --- | --- | --- |
| K<1, b=0, m=0 | Positive curvature | Unique minimum |
| K=1, b=0, m=0 | Zero curvature; positive quartic term | Unique critical minimum |
| K>1, b=0, two outer roots | Positive curvature | Both global minima |
| K>1, three roots, b≠0 | Two minima and one maximum | Field-aligned minimum is global |

A maximum is a valid stationary solution, but it is not an equilibrium state. A metastable minimum is locally stable, but it does not minimize the equilibrium free energy.

## At the critical point the quadratic test is insufficient

Expanding the exact entropy about zero gives the series below for |m|<1. At K=1,b=0 the quadratic term vanishes, but the positive quartic term establishes a strict minimum. The stationary equation has a cubic zero there. A zero Hessian alone would not establish either property for a general objective.

For K just above one at zero field, balancing the quadratic and quartic terms gives m_*²≈3(K−1). At K=1 with a small field, the leading equation is m³/3≈b. The sign follows the field, and its magnitude scales as |b|^(1/3). These are asymptotic approximations with specified small parameters, not replacements for the exact model across the full slider range.

The associated exponents belong to this completely connected model and its mean-field critical behavior. They do not establish the exponents of every nearest-neighbor magnet. Spatial correlations and dimension require additional theory. The next Landau chapter also changes the polynomial coefficients deliberately to study other kinds of stability and coexistence.

$$
\psi_{K,b}(m)=-\ln2+\frac{1-K}{2}m^2+\frac{m^4}{12}+\frac{m^6}{30}+\cdots-bm.
$$

$$
m_*(K,0)^2\sim3(K-1)\quad(K\downarrow1),\qquad m_*(1,b)\sim\operatorname{sgn}(b)(3|b|)^{1/3}\quad(b\to0).
$$

## A branch derivative is not always an equilibrium susceptibility

Differentiate f(m,b)=0 along a smooth stationary branch. Away from zero curvature this gives the response below. On a minimum it is positive. On the intervening maximum it is negative; that number describes the unstable mathematical branch, not a negative equilibrium fluctuation variance.

For K<1 at zero field the response is 1/(1−K). At the exact critical point the limiting response is positive infinity; no finite inverse of a zero curvature should be reported. At K>1,b=0 each ordered branch has a finite local derivative, while the equilibrium pressure has a cusp in b and its first derivative jumps between −m_* and +m_*.

For every finite N, equilibrium response instead equals N Var(m) and is nonnegative. In a symmetric two-peak mixture the mean is zero and Var(m) approaches m_*², making this response grow with N. This does not contradict the finite derivative within either pure phase: the finite ensemble also changes the weights of the two populations.

$$
\left.\frac{dm}{db}\right|_{\text{branch}}=\frac1{\psi^{\prime\prime}(m)}=\frac{1-m^2}{1-K(1-m^2)},\qquad \frac{d\langle m\rangle_N}{db}=N\operatorname{Var}_N(m)\geq0.
$$

## The order of limits specifies the experiment

Fix K>1. At b=0, spin-flip symmetry makes the finite expectation exactly zero for every N, so taking its limit still gives zero. Instead hold a positive field while taking N to infinity, then let that field vanish. This selects the positive ordered minimum. Reversing the field selects its negative counterpart.

These preparations answer different questions. A spontaneous magnetization is defined using a symmetry-breaking preparation, not by asserting that the symmetric finite average secretly equals one peak. The two limiting pressures agree at zero field even though derivatives and selected states differ.

An instructive experiment compares four quantities side by side: finite signed mean, finite mean magnitude, limiting global minima and the derivative on a selected branch. For K=2,b=0, the limiting minima lie near ±0.957504. A finite signed mean remains zero. Increasing N sharpens the distinction; it does not authorize replacing the mean with a modal value.

$$
\lim_{N\to\infty}\langle m\rangle_{N,K,0}=0,\qquad \lim_{b\downarrow0}\lim_{N\to\infty}\langle m\rangle_{N,K,b}=m_*(K)>0\quad(K>1).
$$

## Do not substitute an off-shell surrogate landscape

Another familiar expression is the log-cosh potential φ below. For K>0 its stationary equation is the same self-consistency condition. At a stationary point the entropy identity makes φ and ψ equal. Their curvatures there differ by a positive factor. This explains why both can locate the same stable branches under that condition.

Away from stationary points they are different functions. The constrained magnetization probability is governed by ψ, so a plotted constrained landscape must use ψ throughout. At K=0 the distinction is especially clear: φ is independent of m, whereas ψ has the unique minimum m=tanh b. A solver that differentiates φ at K=0 would falsely return an entire continuum of stationary states.

This is a useful research habit beyond statistical mechanics: two objectives may share optimal values or selected stationary solutions without being interchangeable as functions. Check their domains, their derivatives and the meaning of their nonoptimal values before using either as an energy landscape.

$$
\phi_{K,b}(m)=\frac K2m^2-\ln\!\left[2\cosh(Km+b)\right],\qquad \phi^{\prime}(m)=K\left[m-\tanh(Km+b)\right].
$$

$$
\phi(m_*)=\psi(m_*),\qquad \phi^{\prime\prime}(m_*)=K(1-m_*^2)\psi^{\prime\prime}(m_*)\quad\text{at a stationary point, }K>0.
$$

## Barriers compare extrema but do not determine switching times

With three roots, subtract each minimum's ψ from the middle maximum's ψ to obtain its constrained barrier per spin. Both are positive away from coalescence. At a spinodal the metastable barrier vanishes while the global minimum remains elsewhere. At coexistence the two barriers are equal by symmetry.

The dimensionless extensive barrier is N times this per-spin difference when interpreting the large-N constrained landscape. A switching rate still needs a specified stochastic dynamics, noise convention and prefactor. We do not infer seconds, a mixing-time bound or a measured hysteresis loop from these static values.

For an extremely tiny nonzero field, direct subtraction of two nearly equal free energies can erase their splitting. A useful exact bound avoids that failure. For b>0 write the metastable root as −u and the global root as v, where 0<u<v. Reflection gives a lower bound from the global minimum property; monotonicity along the negative outer branch gives the upper bound. This keeps a positive splitting even when both energies round to the same displayed number.

$$
\Delta_i=\psi(m_{\mathrm{max}})-\psi(m_i)>0,\qquad \psi(-u)-\psi(u)=2bu.
$$

$$
2bu\leq\psi(-u)-\psi(v)\leq2bv\qquad(b>0,\ 0<u<v,\ \text{three roots}).
$$

## Numerical evidence must retain small differences

The calculator interprets each K,b as its actual binary64 rational input. For example, Math.log(3) is not the exact real number ln 3. Analytic answers derived at stipulated exact constants remain valuable, but the reference calculation must evaluate the actual supplied input.

Near zero, subtracting atanh(m)−m loses the leading cubic signal. The implementation instead factors that cubic and evaluates a positive series with a directed remainder. The free-energy difference ψ(m)−ψ(0) similarly factors its quartic term. For t=m²≤1/64, use A and B below; after terms j=0 through L−1, their remaining sums are bounded by the displayed geometric tails. Outside this small-m region directed logarithms evaluate the original expressions. The switch changes the numerical method, not the mathematical model.

Roots are isolated on certified monotone brackets. A root interval must become sufficiently narrow and every derived observation must meet its relative enclosure contract. Adaptive arithmetic increases precision when a sign cannot yet be decided. Hitting the cap returns an unresolved result, never an invented zero or a merged root. A nonzero quantity outside the representable display range keeps its sign and bounds with a null numeric display. Infinity at the exact critical point is a separately established mathematical limit.

Independent Python references use direct high-precision Decimal logarithms and recompute at higher precision. This is intentionally different from copying the factored TypeScript series. A small residual alone is insufficient evidence of an accurate root near zero curvature; a certified interval and branch classification are needed too.

$$
A(t)=\sum_{j=0}^{\infty}\frac{t^j}{2j+3},\qquad B(t)=\sum_{j=0}^{\infty}\frac{t^j}{(2j+4)(2j+3)},\qquad t=m^2.
$$

$$
f(m)=(1-K)m-b+m^3A(m^2),\qquad \psi(m)-\psi(0)=\frac{1-K}{2}m^2-bm+m^4B(m^2).
$$

$$
0\leq R_A\leq\frac{t^L}{(2L+3)(1-t)},\qquad 0\leq R_B\leq\frac{t^L}{(2L+4)(2L+3)(1-t)}\quad(0\leq t<1).
$$

## Connect stationary branches to the experiment

The experiment has a selectable **Free energy** or **Derivative** plot and a table of every stationary root. Switch between the two views for the same preparation: a valley in ψ corresponds to an upward crossing of f, while the intervening maximum corresponds to a downward crossing. The worker supplies certified root roles; the sampled line does not decide them.

| Preparation to compare | Predict before changing it | Read in the result |
| --- | --- | --- |
| K=0.5 → K=1 → K=2 at b=0 | Can zero curvature still describe a strict minimum? | Compare one minimum, the quartic critical minimum, and two global minima with a central maximum. |
| Just below the fold → rounded threshold | Will a rounded field preserve three roots? | Read the certified root count, then use Zoom root and the derivative view; unresolved drawing precision does not erase a certified pair. |
| Positive → negative → smallest nonzero field | Can matching displayed energies establish coexistence? | Inspect global roles and the positive free-energy gap; field sign can select a minimum below display resolution. |

Use **Root details** to compare curvature, entropy, ψ(m)−ψ(0) and the branch derivative. The selected detail row and the plot zoom are separate controls. A negative derivative on the maximum is not a negative equilibrium variance. Enter K or b and commit with Enter or blur; changing the precision ceiling may expose an unresolved comparison without changing the physical preparation. Reset restores the first example. To compare finite E[m] with limiting minimizers, open the preceding finite-system chapter and match K,b explicitly; this reader does not combine the finite distribution and limiting landscape in one panel.

## Research extension: valley depth and valley width both affect finite probabilities

Two minima of equal depth need not carry equal probability in a general finite system: a wider valley can contain more states. Curie–Weiss spin-flip symmetry happens to make the two zero-field widths equal. Deriving that cancellation tells us when it is safe to keep only the two depths and when a missing prefactor matters.

### Derive the prefactor using the constrained counting measure

Use the constrained ψ from this chapter throughout. Choose a fixed small neighborhood I_i around each relevant interior minimum m_i, with positive curvature q_i, separated from other stationary points and endpoints. Its restricted partition sum uses the full-M² Hamiltonian. Stirling's formula gives

$$
Z_{N,i}=\sum_{m\in I_i\cap\{-1,-1+2/N,\ldots,1\}}\binom{N}{N(1+m)/2}e^{N(Km^2/2+bm)},\qquad \binom{N}{N(1+m)/2}\sim\sqrt{\frac{2}{\pi N(1-m^2)}}e^{Ns(m)}.
$$

Within a nondegenerate valley expand ψ(m)=ψ(m_i)+q_i(m−m_i)²/2+… . The lattice spacing is 2/N, so the sum contributes a factor N/2 when replaced by an integral. The Gaussian width is of order N^(−1/2), larger than that spacing. Multiplying all three prefactors is essential.

$$
Z_{N,i}\sim\frac N2\sqrt{\frac{2}{\pi N(1-m_i^2)}}e^{-N\psi(m_i)}\int_{\mathbb R}e^{-Nq_i t^2/2}\,dt=\frac{e^{-N\psi(m_i)}}{\sqrt{(1-m_i^2)q_i}}.
$$

The remaining amplitude A_i includes both entropy's counting prefactor and local curvature. Replacing ψ by the off-shell log-cosh surrogate would give an incorrect measure unless the transformation and its own integration factors were derived as well. Dropping the Hamiltonian's self interaction multiplies every restricted Z by exp(−K/2); normalized branch probabilities stay unchanged.

$$
A_i=\frac1{\sqrt{(1-m_i^2)\psi''(m_i)}}=\frac1{\sqrt{1-K(1-m_i^2)}},\qquad W_i\simeq\frac{A_i e^{-N\psi(m_i)}}{\sum_j A_j e^{-N\psi(m_j)}}.
$$

This estimate concerns fixed isolated minima, with an exponentially smaller contribution away from the relevant neighborhoods. Conditional fluctuations within a valley are Gaussian to leading order. Their small variance does not describe the spread between valleys.

$$
\sqrt N(m-m_i)\mid\{m\in I_i\}\Rightarrow\mathcal N(0,q_i^{-1}),\qquad \operatorname{Var}(m\mid I_i)\sim\frac1{Nq_i}.
$$

### A field of order 1/N can change the mixture by order one

Fix K>1 away from criticality, and let ±m\* be the two zero-field minima. Now prepare b_N=v/N with v fixed. Each valley's location moves by only order 1/N, but the extensive field energy changes by order one. The zero-field amplitudes agree by symmetry. Using the envelope derivative dψ(m_i(b))/db=−m_i(b) gives

$$
N\big[\psi_{b_N}(m_-(b_N))-\psi_{b_N}(m_+(b_N))\big]\longrightarrow2vm_*,\qquad \frac{W_+}{W_-}\longrightarrow e^{2vm_*}.
$$

Here m_−(b_N) and m_+(b_N) denote the negative and positive branches. Their magnitudes need not agree at nonzero field, but both approach the zero-field magnitude m\*. A compact notation for the limiting mixture is

$$
W_+(v)=\frac1{1+e^{-2vm_*}},\qquad \lim_{N\to\infty}\langle m\rangle_{N,K,v/N}=m_*\tanh(vm_*).
$$

This interpolates between equal weights at v=0 and an almost selected phase at large |v|. It differs from a field held nonzero as N grows, which selects one global branch. The limit does not say the spins become independent within a finite sample.

The leading finite response follows directly from the mixture variance, without differentiating an uncontrolled asymptotic error term. The between-valley variance remains order one; the conditional Gaussian variance within a valley is only order 1/N.

$$
\frac{\chi_b(N,K,v/N)}N=\operatorname{Var}_{N,K,v/N}(m)\longrightarrow m_*^2\operatorname{sech}^2(vm_*).
$$

![Exact finite magnetization at a shrinking field approaching the two-valley mixture, and the distinct within-valley and between-valley response scales.](/assets/learning/phase-transition-research-laplace-branch-weights.en.svg)

The first panel varies v while b=v/N; the second compares a finite response growing with N to the finite local derivative of a selected branch. N is a preparation size, not elapsed time.

### Compute a finite sequence and inspect what approaches what

At K=2 and v=0.7, m\*≈0.957504. The table lists N, b_N, exact finite mean, positive-side weight, and limiting mixture mean. At m=0 the positive-side convention assigns half the weight to each side. That convention becomes irrelevant in the limit because the central region is exponentially suppressed. The Python example computes exact macrostate sums rather than using the Laplace expression as its numerical oracle. N>128 belongs to this separate example, beyond the finite reader's controls.

| N | b_N | Finite mean | Positive weight | Limit mean |
| --- | --- | --- | --- | --- |
| 32 | 0.021875 | 0.554374 | 0.790606 | 0.560280 |
| 128 | 0.005469 | 0.558951 | 0.792145 | 0.560280 |
| 512 | 0.001367 | 0.559955 | 0.792469 | 0.560280 |
| 2048 | 0.000342 | 0.560199 | 0.792547 | 0.560280 |

```python
from math import exp, fsum, lgamma, log, tanh

k, v = 2.0, 0.7
lo, hi = 0.5, 1.0
for _ in range(100):
    m = (lo + hi)/2
    if tanh(k*m) > m:
        lo = m
    else:
        hi = m
mstar = (lo + hi)/2
limit = mstar*tanh(v*mstar)
for n in (32, 128, 512, 2048):
    b = v/n
    ms = [j/n for j in range(-n, n + 1, 2)]
    logs = [lgamma(n + 1) - lgamma(j + 1) - lgamma(n-j + 1)
            + n*(k*m*m/2 + b*m) for j, m in enumerate(ms)]
    top = max(logs)
    ws = [exp(x-top) for x in logs]
    mean = fsum(w*m for w, m in zip(ws, ms))/fsum(ws)
    plus = (fsum(w for w, m in zip(ws, ms) if m > 0)
            + fsum(w/2 for w, m in zip(ws, ms) if m == 0))/fsum(ws)
    print(n, f'{b:.6f}', f'{mean:.6f}', f'{plus:.6f}', f'{limit:.6f}')
```

### Research checks and limits of the approximation

#### Why can a barely visible field change a finite ensemble?

**Research explanation.** A field v/N multiplies total magnetization M≈±Nm\* in the exponent. Their relative log weight is approximately 2vm\*, even though the two minimum locations barely move. State selection depends on extensive bias as well as local geometry.

#### Can a small conditional variance justify reporting one selected branch?

**Research explanation.** No. The total-variance identity adds the variance of the conditional means. For comparable valley weights that contribution remains order one in m. Reporting only 1/(Nq_i) discards the dominant source of finite equilibrium susceptibility.

#### Where must the Gaussian prefactor be replaced?

**Research explanation.** At K=1,b=0 the curvature vanishes; at a spinodal stationary points merge; an endpoint or a minimum approaching an endpoint also changes the counting approximation. This derivation assumes separated interior minima with curvature bounded away from zero, and is not uniform through those limits. The preceding quartic calculation supplies the appropriate critical alternative.

The entropy landscape is grounded in [Tong, Statistical Physics, chapter 5](https://davidtong.org/pdfs/teaching/statistical-physics/statphys5.pdf). The discrete Stirling–Laplace prefactor and the shrinking-field sequence are derived here for the stated counting convention. They predict static equilibrium weights; neither a switching rate nor a sampler's mixing time has been specified.

## Worked checks before trusting the picture

### 1. Solve the noninteracting limit

Set K=0,b=0.2. Find the stationary root, pressure and branch derivative.

**Explanation.** The equation is atanh(m)=b, hence m=tanh(0.2)≈0.197375. Substituting the binary-entropy identity gives p=ln(2 cosh 0.2)≈0.713015. The derivative is sech²(0.2)≈0.961043. There is exactly one minimum because ψ has strictly positive curvature. These are checks of a whole analytic family, not a fit to a single plot.

### 2. Verify an ordered root at an exact constant

Stipulate K=ln 3 and b=0 in exact real arithmetic. Show that m=1/2 is a stable root and compute its free energy relative to zero.

**Explanation.** atanh(1/2)=ln 3/2, so f(1/2)=0. The curvature is 4/3−ln 3>0. Using s(1/2)=2 ln 2−3 ln 3/4 gives ψ(1/2)−ψ(0)=5 ln 3/8−ln 2<0. Its reflection has the same energy, and both are global minima. A binary64 log input gives a nearby root; demanding an exactly equal rational root would test the wrong preparation.

### 3. Locate a spinodal without searching a plotted mesh

Use K=9/8. Compute s and B_sp, and distinguish b=0 from b=B_sp.

**Explanation.** s=1/3 and atanh(1/3)=ln 2/2. Thus B_sp=3/8−ln 2/2≈0.02842640972002735. At b=0 the two minima coexist and the middle root is unstable. At the exact positive threshold the negative minimum and maximum merge at −1/3. The positive minimum is still distinct. Adjacent binary64 fields around the rounded threshold must be compared with the exact threshold, rather than all being declared tangent by an epsilon.

### 4. Classify the zero-curvature point

At K=1,b=0, does ψ″(0)=0 imply that zero is unstable? What happens under a small positive field?

**Explanation.** The leading nonconstant term is m⁴/12, which is positive on both sides of zero, so zero is a strict critical minimum. A small positive field yields m≈(3b)^(1/3). The response diverges as b tends to zero even though the magnetization tends to zero. At the smallest positive binary64 field the root is about 2.4564×10^−108; its negative free-energy difference is too small for a nonzero binary64 display. Neither fact justifies reporting an exactly zero field response.

### 5. Separate mixture response from branch response

For K=2,b=0, compare the ordered branch derivative with the finite symmetric equilibrium response.

**Explanation.** The positive root is approximately 0.957504, giving curvature about 10.02125 and branch derivative about 0.09979. The finite symmetric mean is exactly zero. As N increases its variance approaches m_*², so N Var(m) grows approximately as 0.9168 N. The two calculations describe within-branch motion and reweighting of both populations, respectively. A negative derivative on the central maximum belongs to neither stable equilibrium response.

### 6. Preserve a bias invisible in rounded energies

Take K=2 and the smallest positive binary64 b. Which minimum is global if both displayed free energies match?

**Explanation.** The positive root is global by field sign. The negative root remains metastable. The gap is bounded between 2b times the two positive root magnitudes and is close to 1.915008 b. Its sign is positive even if subtracting the two rounded energy numbers returns zero. Repeating just above K=1 gives an even smaller positive gap that lies outside the numeric display range; a signed interval remains meaningful.

## Turn the derivation into an experiment and a research question

The interactive view connects a selectable ψ-versus-m or f-versus-m plot with a table of all roots and detailed curvature, role and pressure readouts. Start at K=0,b=0, increase K through one, then add a small positive field. Show both a broad view and a local zoom near a coalescing pair; a coarse fixed mesh can hide two roots. Use explicit labels for exact criticality, unresolved comparisons and values outside the numeric display range.

A second comparison should place the finite magnetization distribution beside the limiting minima. Keep N and the choice of finite or limiting observable visible. Let the learner predict the signed mean and mean magnitude before revealing them. This combined finite-versus-limit view is a proposed teaching interaction. The current registered reader provides the limiting landscape and root table; the finite distribution is available in the preceding chapter. Standalone numerical checks alone do not establish worker cancellation or visual correctness.

For AI research, the direct connections include energy-based objectives, symmetry breaking, multiple fixed points, implicit differentiation and the difference between local convergence and global selection. A stationary equation does not prove that a sampler reaches equilibrium, nor does a small residual establish reliable implicit gradients when curvature is tiny. Ask which assumptions make a landscape meaningful for the actual trained or sampled system.

[David Tong's statistical field theory notes](https://www.damtp.cam.ac.uk/user/tong/sft/sfthtml/S1.html) provide broader context for constrained free energies and mean-field ideas, including cautions about spatial systems. The counting bounds and numerical contract above refer specifically to our completely connected preparation. Next, a sixth-order Landau family will vary coefficients to separate continuous transitions, first-order coexistence and spinodal stability. That family requires its own derivation and verification.
