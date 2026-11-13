## A correlation can travel around either side of a ring

Imagine arrows placed on a circular necklace. Neighboring arrows tend to align, or tend to oppose each other. To compare two sites, you can walk clockwise or counterclockwise. On a small ring, both routes matter. A formula copied from an infinite line misses the route that wraps around the other side.

The previous chapter studied two spins and the Hessian with respect to their local fields. Here we introduce a genuine spatial index, a boundary condition, and a distance. We will solve the zero-field nearest-neighbor Ising ring exactly, inspect small named configurations, and separate a finite-ring correlation from an infinite-chain correlation length. We also examine why a long alternating correlation need not produce a large response to a uniform field.

You should know finite Boltzmann sums, connected covariance, and a geometric series. A first encounter with a two-by-two matrix is enough for the independent transfer-matrix derivation. [Tong's statistical physics notes](https://www.damtp.cam.ac.uk/user/tong/statphys/statmechhtml/S5.html) provide background on periodic Ising chains. The distance conventions, finite input bounds, worked examples and numerical contracts below are specified explicitly for this chapter.

## Specify sites, bonds and the boundary

There are N named classical spins, with N at least three. Site N is connected back to site 1. Each of the N nearest-neighbor bonds is counted once:

$$
E(\sigma)=-J\sum_{i=1}^N\sigma_i\sigma_{i+1},\qquad \sigma_{N+1}=\sigma_1,\qquad K=\frac{J}{k_BT},\qquad p(\sigma)=Z_N^{-1}\exp\!\left(K\sum_{i=1}^N\sigma_i\sigma_{i+1}\right).
$$

All local fields are exactly zero. K is dimensionless, with positive values favoring alignment and negative values favoring opposing neighbors. The numerical model accepts N=3 through 128 and finite binary64 K in [−8,8]. These are computational limits, not a definition of all physically meaningful chains. N=2 is excluded because writing two periodic links then counts the same unordered pair twice; the preceding two-site model used one pair interaction.

An offset r runs from 0 to N−1. Its shortest ring distance is d=min(r,N−r). The two path lengths are r and N−r. Distance is measured in lattice spacings; multiplying by a physical spacing is a separate unit conversion. No time step, animation frame, optimizer iteration or sampler transition is being used as a spatial distance.

## Expand bonds and count even degrees

Set t=tanh K. The elementary identity for a single bond is

$$
e^{K\sigma_i\sigma_{i+1}}=\cosh K\,(1+t\sigma_i\sigma_{i+1}),\qquad Z_N=(2\cosh K)^N(1+t^N).
$$

To see the partition formula, expand the product of N parentheses. Each term chooses a set of edges. Summing a free spin over ±1 kills a term if that spin has odd degree in the chosen set. On a simple ring, only the empty edge set and the complete loop have even degree at every site. The empty set contributes 1 and the full loop contributes t to the power N. Each surviving spin sum contributes a factor two.

This is exact counting, not a high-temperature truncation. There are no missing higher-order loops on this graph. At K=0, the loop term vanishes and Z is exactly 2 to the power N. When K is negative and N is odd, the loop correction is negative, but its magnitude is less than one for every finite K. The positive partition function therefore stays well defined.

## Insert two spins and expose the two paths

Global spin reversal leaves every bond unchanged and reverses each spin. Thus every site mean is zero at finite N. The connected covariance equals the product expectation:

$$
C_N(r)=\mathbb E[\sigma_i\sigma_{i+r}]=\frac{t^r+t^{N-r}}{1+t^N},\qquad C_N(0)=1,\qquad C_N(r)=C_N(N-r).
$$

Inserting the two spins changes the parity rule at the two selected sites: they must now have odd degree, while all other sites have even degree. Precisely two edge sets survive, the two paths connecting the selected sites. Their weights give the two numerator terms. At r=0 the formula is interpreted with t to the power zero equal to one, including at t=0; the answer is a spin square, exactly one.

The picture is useful before doing algebra. For positive K, both paths contribute positively. For negative K, path parity determines the sign. In an even ring the path lengths have the same parity. In an odd ring they have opposite parity and partially cancel. Shortest distance alone does not remove the longer path from the probability law.

## Verify the result using a transfer matrix

A separate matrix route gives an important cross-check. With the spin basis ordered −1,+1, define

$$
T=\begin{pmatrix}e^K&e^{-K}\\e^{-K}&e^K\end{pmatrix},\qquad D=\begin{pmatrix}-1&0\\0&1\end{pmatrix},\qquad Z_N=\operatorname{tr}(T^N),\qquad C_N(r)=\frac{\operatorname{tr}(DT^rDT^{N-r})}{\operatorname{tr}(T^N)}.
$$

Multiplying transfer entries assigns one factor to every bond; taking a trace closes the ring. Inserting D multiplies the selected site's contribution by its spin value. The eigenvalues of T are 2 cosh K and 2 sinh K. Their ratio is t, and D exchanges the corresponding symmetric and antisymmetric directions. Evaluating the trace therefore reproduces the two-path expression.

A negative second eigenvalue for negative K does not mean that any transition weight or probability is negative. Every entry of T is positive. The negative eigenvalue records an alternating spatial mode. Also, this transfer matrix is a device for summing spatial weights; it is not automatically a stochastic transition kernel for a sampler.

## Count the triangle and the square by hand

For N=3, two configurations have all spins equal and bond sum 3. The other six have bond sum −1. This gives

$$
Z_3=2e^{3K}+6e^{-K},\qquad C_3(1)=C_3(2)=\frac{e^{4K}-1}{e^{4K}+3}.
$$

Use the ideal symbolic values K=±log(3)/2, so t=±1/2. The finite answers are:

| Ring | t | C(1) | C(2) | Uniform response per spin |
| --- | --- | --- | --- | --- |
| N=3 | 1/2 | 2/3 | 2/3 | 7/3 |
| N=3 | −1/2 | −2/7 | −2/7 | 3/7 |
| N=4 | 1/2 | 10/17 | 8/17 | 45/17 |
| N=4 | −1/2 | −10/17 | 8/17 | 5/17 |

For the triangle at positive K, the two aligned configurations together have probability 3/4 and the other six together have probability 1/4. At negative K those group probabilities become 1/28 and 27/28. Within each group the named configurations have equal probability. Treating the two groups as equally likely would discard multiplicity.

The two triangular off-site correlations agree because offset 2 is also one nearest-neighbor step around the ring. In the square, the opposite site is at distance two; its two paths are equally long. Entering a rounded binary64 `Math.log(3)/2` specifies a nearby actual K, rather than the ideal symbolic logarithm used in this table.

## Sum spatial covariance to obtain a uniform response

To define a response, temporarily extend the family by a common dimensionless field b multiplying the total spin M. Evaluate the derivative at b=0 with K held fixed:

$$
M=\sum_i\sigma_i,\qquad m=M/N,\qquad \chi_N=\left.\frac{\partial\mathbb E_b[m]}{\partial b}\right|_{b=0}=\frac{\operatorname{Var}(M)}{N}=\sum_{r=0}^{N-1}C_N(r),\qquad \operatorname{Var}(m)=\chi_N/N.
$$

There are N pairs for each offset in the double covariance sum, which explains the normalization. The self term C(0)=1 must be included. Negative cross terms can reduce the total response but cannot make it negative, because the same quantity is a variance divided by N. It is strictly positive at every admitted finite K: all configurations retain positive probability, including some with different M.

Summing the geometric series yields

$$
\chi_N=\frac{1+t}{1-t}\,\frac{1-t^N}{1+t^N}.
$$

This is a response to a common field, not a derivative with respect to K. Differentiating with respect to K probes the bond-sum feature instead. For an energy bias H per spin, the physical derivative with respect to H includes 1/(kBT) at fixed positive T and J. A dimensionless response alone is not a susceptibility in SI units.

## Separate the finite ring from the infinite-chain envelope

Hold an offset r fixed and let N tend to infinity at fixed finite K. Since |t| is less than one, the wrapping path and loop correction vanish:

$$
C_\infty(r)=t^r,\qquad |C_\infty(r)|=e^{-r/\xi},\qquad \xi=-\frac{1}{\log|t|}\quad(0<|t|<1).
$$

For negative K, the envelope decays while the sign alternates. The length ξ describes that envelope in lattice spacings. It is not the largest distance at which correlation is nonzero. At K=0, distinct sites are independent; this chapter explicitly assigns the continuous zero-range limit ξ=0. It does not evaluate a finite ordinary logarithm of zero.

A finite ring has C(N)=C(0)=1 if the offset is extended periodically. Its correlation cannot follow a decaying infinite-line exponential for arbitrarily large offsets. In particular, when ξ is comparable to or larger than the ring circumference, fitting the finite data to a single infinite-chain decay can be misleading. The order “fixed r, then N large” is part of the statement, not a cosmetic condition.

## Understand frustration on an odd opposing ring

An even ring can alternate up and down around the entire circle. An odd ring cannot: returning to the starting site forces at least one equal-spin bond. This is geometric frustration, not a failure of the probability calculation.

For an ideal very strong negative K, outside the finite numerical input interval, an odd ring has 2N ground configurations: choose the frustrated bond and the overall spin orientation. Every such configuration has total spin +1 or −1. In an even ring, the two alternating ground configurations have total spin zero. Consequently,

$$
\lim_{K\to-\infty}\chi_N=\begin{cases}1/N,&N\text{ odd},\\0,&N\text{ even},\end{cases}\qquad \lim_{K\to-\infty}C_N(1)=-1+2/N\quad(N\text{ odd}).
$$

For N=3 the nearest-neighbor limit is −1/3, not −1. All three bonds cannot simultaneously satisfy the opposing preference. The zero site means still hold because both overall orientations are included. A finite simulation stuck in one configuration would measure a different empirical average until it explores the preparation adequately.

These fixed-N cold limits differ from the fixed-finite-K large-N limit. Reporting the two in a different order without checking them can hide the odd-ring effect. The calculator at K=−8 can approach these values closely, but it does not claim to evaluate K=−infinity.

## Long alternating range need not mean a large uniform fluctuation

The infinite-chain uniform response has a particularly simple form:

$$
\chi_\infty=\frac{1+t}{1-t}=e^{2K},\qquad \chi_{\infty,\mathrm{staggered}}=\sum_{r=-\infty}^{\infty}(-1)^r t^{|r|}=\frac{1-t}{1+t}=e^{-2K}.
$$

The second source alternates its sign from site to site on the infinite chain. For strongly negative K, ξ is large but the uniform response is small. Positive and negative correlations cancel in the uniform sum. The staggered source matches the alternating pattern and gives the large response instead. This is why a correlation length and the variance of a particular global observable answer different questions.

The staggered formula here concerns the infinite chain. A periodic odd ring has no globally consistent two-sublattice pattern across its closing bond, so one must specify a finite source pattern before calculating its response there. It is not safe to reuse the infinite-chain expression without that qualification.

The exact one-dimensional zero-field thermodynamic-limit pressure at fixed finite K is log(2 cosh K), with no singularity at a finite K. The nearest-neighbor chain therefore does not acquire the mean-field transition at K=1 from the completely connected Curie–Weiss model. Its long-range low-temperature behavior arises through a different limit and a different interaction graph.

## Preserve tiny correlations in the numerical model

The model uses actual binary64 K as an exact input. Directed exponentials and logarithms produce enclosures; exact integer and rational operations propagate them. It forms u=exp(−2|K|) and v=|t|=(1−u)/(1+u). Correlation powers retain a separate binary exponent rather than being rounded to the current working fixed-point grid. Thus a nonzero far-distance correlation can stay certified even when its ordinary displayed value is unavailable through underflow.

For an odd opposing ring, let d be the shortest distance and define a positive geometric sum G. Then

$$
G_n(v)=\sum_{j=0}^{n-1}v^j,\qquad C_N(d)=(-1)^d v^d\,\frac{G_{N-2d}(v)}{G_N(v)}\quad(K<0,\ N\text{ odd}).
$$

This cancels the shared factor 1−v before division. Positive forms also avoid subtracting nearly canceling terms in χ. Working precision increases from 128 through 8192 bits. A resolved nonzero observation has a certified sign and relative enclosure width at most 2 to the power −48. At a chosen precision ceiling, an unavailable result is not replaced by zero. Known parity signs can remain available independently of the magnitude.

The model reports a directed logarithm of the absolute correlation separately from the signed correlation. The general magnitude-log projection in a numerical observation is only approximate. At the least positive K and N=128, the opposite-site correlation has a logarithm near 64 log K plus log 2, far below the linear floating display range. Its enclosing positive rational bounds still distinguish it from exact independence.

Independent Decimal references cover 20 actual-input preparations and 2,580 observations. Eighteen preparations are also checked by transfer traces, and eleven by direct named-state enumeration. The two extreme tiny-coupling long-ring cases use high-precision closed expressions; they are explicitly not claimed as transfer checks, since cancellation would require a much larger transfer-trace precision budget. Model and manuscript checks alone do not establish correct worker lifecycle or actual-reader usability; the registered view requires separate integration evidence.

## Reproduce a small preparation before scaling up

For an ordinary-sized K, this short Python example enumerates all eight triangle states:

```python
from itertools import product
from math import exp, log

N, K = 3, log(3) / 2
states = list(product((-1, 1), repeat=N))
weights = [exp(K * sum(s[i] * s[(i + 1) % N]
                       for i in range(N))) for s in states]
Z = sum(weights)
C1 = sum(w * s[0] * s[1] for s, w in zip(states, weights)) / Z
chi = sum(w * sum(s)**2 for s, w in zip(states, weights)) / (N * Z)
print(C1, chi)  # approximately 2/3 and 7/3
```

This ordinary floating calculation illustrates the probability sum. It is not a certified replacement for the directed model or independent high-precision oracle. Replace K by its negative and verify the opposing triangle. Then change N to four and check all offsets, including the opposite site. Finally compare a fixed short distance at N=8 and N=128 to see the wrapping correction diminish at a moderate fixed K.

The registered visualization controls N, signed K and a selected offset. It marks both connecting paths, compares finite covariance with the infinite-chain reference and labels uniform response separately. A path highlight is a spatial explanation; changing or animating it does not define physical dynamics. Standalone theory does not by itself verify the implementation of those interactions.

## Connect two spatial paths to the experiment

Move Selected offset r and follow the blue path of length r and orange path of length N−r. The dots are sites, not realized spin values. Changing this viewing control selects an observation of the same equilibrium law; it does not advance time or rerun a spin trajectory. The finite-versus-infinite plot uses shortest distance d, while the complete table retains every offset r.

| Preparation to compare | Predict before changing it | Read in the result |
| --- | --- | --- |
| Positive triangle → positive square | Can the infinite-chain curve replace the finite ring? | Both paths contribute; the triangle gives a different C(1) from its infinite-chain reference. |
| Strong opposing coupling, odd127 → even128 | Does a long envelope require a large uniform response? | Inspect frustration, signed spatial correlations and uniform-field response separately. |
| Independent ring → smallest nonzero coupling | Does an invisible distant value mean independence? | Compare structural zero with positive bounds and the directed log of the selected correlation. |

Commit integer N and signed K with Enter or blur. N counts3–128 distinct sites and bonds. Select r=0 to recover self correlation one, and compare r with N−r to verify reflection. Use Next observation rows to inspect the whole table rather than only its first32 offsets. The envelope length is an infinite-chain reference in lattice spacings, not a finite-ring cutoff. A precision ceiling changes available numerical evidence, not the ring or its boundary.

## Research extension: resolve the ring's response into spatial modes

A uniform field asks whether the whole necklace moves together. An alternating field asks a different question. Fourier modes provide a complete set of spatial patterns between these extremes, and reveal how an odd ring frustrates a pattern whose preferred wavelength is two lattice spacings. The finite allowed wavevectors matter as much as the decay envelope.

### Diagonalize a covariance that depends on the offset

Keep the zero-field finite ring and its exact C_N(r). Label sites j=0,…,N−1, and use the discrete periodic wavevectors only:

$$
q_l=\frac{2\pi l}{N},\qquad z_l=\frac1{\sqrt N}\sum_{j=0}^{N-1}e^{-iq_lj}\sigma_j,\qquad l=0,\ldots,N-1.
$$

Every spin has mean zero. Translation invariance makes the covariance matrix circulant: shifting both indices together leaves it unchanged. Insert the definition of z, group pairs by their offset, and sum the phase factors. This gives the static structure factor

$$
S_N(q_l)=\mathbb E[|z_l|^2]=\sum_{r=0}^{N-1}C_N(r)e^{-iq_lr}=\sum_{r=0}^{N-1}C_N(r)\cos(q_lr)\geq0.
$$

The imaginary terms cancel because C_N(r)=C_N(N−r). S is a variance-like quadratic quantity, so a negative cross correlation cannot make it negative. It is strictly positive at finite K: every spin configuration has positive probability, and no nonzero linear mode is identically zero on all configurations. These are equal-time spatial modes, not temporal frequencies from a simulated trace.

Substitute the two-path formula, sum the two finite geometric series and use exp(iq_l N)=1. With t=tanh K the result is

$$
S_N(q_l)=\frac{1-t^2}{1-2t\cos q_l+t^2}\frac{1-t^N}{1+t^N},\qquad |t|<1.
$$

The denominator is |1−t exp(iq)|² and is positive. The loop correction is positive even when N is odd and t<0. This formula is exact on the allowed grid; plugging arbitrary q into it does not give the finite transform of the same ring. A smooth line between grid points is at most a visual guide or a separately labeled reference.

### Match a real source pattern to the fluctuation

For modes other than q=0 or π, use a real normalized cosine probe. Its sine partner has the same variance. At the self-conjugate modes q=0 and, for even N, q=π, use normalization 1/sqrt(N) rather than sqrt(2/N).

$$
A_l=\sqrt{\frac2N}\sum_j\cos(q_lj)\sigma_j,\qquad \operatorname{Var}(A_l)=S_N(q_l),\qquad \left.\frac{d\mathbb E_b A_l}{db}\right|_{b=0}=S_N(q_l)\quad(p_b\propto p_0e^{bA_l}).
$$

Here b couples to that normalized probe. The q=0 structure factor equals the chapter's uniform response per spin because z₀=M/sqrt(N). Matching the normalization of the measured observable and source avoids an extra factor N or two.

Discrete Fourier orthogonality also supplies a sum rule. The trace of the covariance is N because each diagonal entry is one. A peak can redistribute this total fluctuation power but cannot create additional diagonal variance.

$$
\sum_{l=0}^{N-1}S_N(q_l)=N,\qquad S_N(0)=\chi_N.
$$

### An odd ring cannot place a mode exactly at π

For negative K, the infinite-chain spectrum favors q=π. An even ring admits that exact wavevector; an odd ring's closest pair is π±π/N. At fixed odd N, taking K→−∞ in the finite formula cancels two vanishing factors and yields

$$
\lim_{K\to-\infty}S_N(q_l)=\frac1{N\cos^2(q_l/2)},\qquad \lim_{K\to-\infty}\max_l S_N(q_l)=\frac1{N\sin^2(\pi/(2N))}\quad(N\text{ odd}).
$$

This includes S_N(0)→1/N, consistent with the odd-ring ground states' M=±1. Several modes retain power because the frustrated bond can occupy different positions. In contrast an even ring has only two alternating ground configurations and concentrates all the power in the exact π mode:

$$
\lim_{K\to-\infty}S_N(\pi)=N,\qquad \lim_{K\to-\infty}S_N(q_l)=0\quad(q_l\ne\pi,\ N\text{ even}).
$$

These are fixed-N cold limits, outside the finite K control interval. They do not assert a finite-temperature singularity. At a fixed finite negative K, every eigenvalue stays positive; the peak is bounded independently of N since the loop factor tends to one and |t|<1. Consequently,

$$
\lim_{N\to\infty}\frac{\max_l S_N(q_l)}N=0\quad(K\text{ fixed finite}),\qquad \lim_{\substack{N\to\infty\\N\ \mathrm{odd}}}\frac{\lim_{K\to-\infty}\max_l S_N(q_l)}N=\frac4{\pi^2}.
$$

The order of size and temperature limits is visible in a mode plot, not just in a free-energy derivative. A large correlation length by itself does not specify which finite patterns can fit around the circumference.

![Discrete antiferromagnetic ring spectra and normalized peak power showing finite coupling versus fixed-size cold-limit behavior.](/assets/learning/correlations-research-ring-fourier-modes.en.svg)

The upper panel shows allowed wavevectors only, with the reflected half omitted. The lower panel compares peak/N at K=−1.2 with the separately derived cold limits. The table reports the uniform mode, peak mode, trace check and ξ/N for the same finite preparation. These are theoretical equilibrium sums, not a scattering measurement or a dynamics simulation.

| N | q_peak/π | S(0) | S_peak | ΣS/N | ξ/N |
| --- | --- | --- | --- | --- | --- |
| 7 | 0.857143 | 0.161219 | 2.811737 | 1.000000 | 0.785205 |
| 8 | 1.000000 | 0.056398 | 6.852926 | 1.000000 | 0.687054 |
| 15 | 0.933333 | 0.103390 | 5.422716 | 1.000000 | 0.366429 |
| 16 | 1.000000 | 0.081353 | 9.885286 | 1.000000 | 0.343527 |

```python
from math import tanh, cos, pi, log, fsum

k = -1.2
t = tanh(k)
xi = -1/log(abs(t))
for n in (7, 8, 15, 16):
    corr = [(t**r+t**(n-r))/(1+t**n) for r in range(n)]
    spectrum = [fsum(corr[r]*cos(2*pi*l*r/n) for r in range(n))
                for l in range(n)]
    lpeak = n//2
    print(n, f'{2*lpeak/n:.6f}', f'{spectrum[0]:.6f}',
          f'{spectrum[lpeak]:.6f}', f'{fsum(spectrum)/n:.6f}',
          f'{xi/n:.6f}')
```

### Research checks and scope

#### Why is the strongest negative correlation not a negative structure factor?

**Research explanation.** The Fourier coefficients weight the alternating covariances with a matching pattern. The result equals E[|z|²], which is nonnegative. Matrix positivity constrains quadratic forms, not the sign of each spatial covariance entry.

#### May an odd ring use the even-ring staggered eigenvector unchanged?

**Research explanation.** The sequence (−1)^j fails periodic closure on an odd number of sites. It can be defined as a particular finite source, but is not one periodic eigenmode. Its response must be computed as the full source quadratic form or by decomposition into the allowed Fourier modes.

#### Does a narrow large peak establish a finite-K phase transition?

**Research explanation.** No. At every fixed finite K the spectrum is smooth, its normalized peak vanishes as N grows, and the one-dimensional pressure stays analytic. The finite-size cold preparation and its order of limits are additional assumptions, not evidence of a finite-temperature transition.

[Tong's periodic-chain derivation](https://davidtong.org/pdfs/teaching/statistical-physics/statphys5.pdf), section 5.3.1, grounds the transfer law. The discrete transform, normalization, sum rule and odd/even mode limits are derived here from this chapter's exact C_N. The existing reader still selects a spatial offset; this supplement does not claim a new Fourier control or a temporal spectrum.

## Check your understanding with explained problems

### 1. Find the missing route

At ideal t=1/2 and N=3, why is C(1) equal to 2/3 rather than 1/2?

**Explanation.** The short path contributes 1/2, the wrapping path contributes 1/4, and the loop normalization is 1+1/8. Their ratio is (3/4)/(9/8)=2/3. The value 1/2 belongs to the fixed-distance infinite-chain limit, not this triangle.

### 2. Count the frustrated triangle

At ideal t=−1/2, calculate C(1) and the uniform response per spin. Does the negative correlation contradict a nonnegative variance?

**Explanation.** The numerator is −1/2+1/4=−1/4 and the denominator is 1−1/8=7/8, giving −2/7. Including the self term, χ=1−4/7=3/7. The total-spin variance is Nχ=9/7, which is positive. Cross covariances reduce the sum without making it a negative variance.

### 3. Keep the observable normalization

For the positive square in the table, find Var(M) and Var(m).

**Explanation.** χ=45/17, so Var(M)=4χ=180/17 and Var(m)=χ/4=45/68. The total spin and its per-site average differ by a factor four, hence their variances differ by sixteen. Neither variance should be silently substituted for the per-spin field response.

### 4. Interpret a long alternating envelope

A negative K produces a very large ξ. Must the uniform field response also be very large?

**Explanation.** No. The infinite-chain response is exp(2K), which becomes small for negative K, while the staggered response is exp(−2K). Alternating spatial signs cancel for a uniform source and reinforce for a matching staggered source. The source pattern is part of a response measurement.

### 5. Distinguish zero, underflow and a precision cap

Compare off-site correlations at K=0, at a tiny positive K whose distant value is below the floating display range, and at the same K with an insufficient precision ceiling.

**Explanation.** K=0 gives exact structural independence and zero off-site covariance. At positive K, the exact off-site covariance is positive even if the ordinary displayed number is unavailable; resolved bounds and a separate log absolute correlation can report it. A precision-capped result is unresolved, with a known positive sign but no certified magnitude at the requested width. These are three different states of knowledge.

### 6. State the limit before using a length

Can you use C(N)=exp(−N/ξ) for a finite periodic ring? What happens to the cold uniform response of an odd opposing ring?

**Explanation.** Offset N returns to the original spin, so C(N)=1 exactly. The exponential formula describes fixed distance on the infinite chain and cannot be extended around a finite circumference without the second path. At fixed odd N in the ideal strong opposing limit, the ground states have M=±1, hence χ=1/N. Taking a different size-and-coupling limit requires a separate argument.

## Continue from spatial covariance to the error of an average

This chapter defines a finite spatial probability law, not a temporal sampling algorithm. In an AI energy model, spatial correlations can describe dependence among components of one sample. The uncertainty of a mean over many correlated samples is a second problem: it needs covariances across sample indices, a specified preparation, and conditions for any asymptotic limit. A spatial correlation length alone cannot determine an effective sample count for an arbitrary training stream.

The next family uses a stationary Gaussian AR(1) preparation to calculate the exact finite variance of an average and compare it with an asymptotic integrated correlation time. Negative correlation, finite-size boundary terms, and the degenerate endpoints will be kept explicit. For further research here, extend the ring to nonzero local sources, compare source-specific Fourier modes, and verify a finite-source Hessian against direct enumeration before generalizing to higher-dimensional lattices. Those extensions and the rest of the curriculum remain unfinished.
