## A function limit versus an integral limit

Pointwise convergence alone does not ensure convergence of integrals. Specify both the convergence notion and the integrating measure. Here all integrals use Lebesgue measure on [0,1]. The first sequence x to the nth power tends to zero for x<1 but remains one at x=1. Its limiting function is zero almost everywhere and has integral zero.

$$
\int_0^1x^n\,dx=\frac1{n+1}\longrightarrow0.
$$

## A common bound for dominated convergence

Dominated convergence assumes measurable functions converging almost everywhere to a measurable limit and a single integrable g bounding every absolute value. Then L1 distance to the limit tends to zero, and the integrals converge. For x^n, use g=1. Separate integrability of each term, or a different bound for each n, is not enough. The hypotheses concern the whole sequence, beyond the finite terms shown by a slider.

$$
f_n\to f\text{ a.e.},\quad |f_n|\le g\in L^1\implies\int|f_n-f|\to0.
$$

## A concentrating-mass counterexample

The second sequence has height n on the open interval (0,1/n) and zero elsewhere. Every fixed positive x eventually leaves the support, and the value at zero is always zero. The pointwise limit is therefore zero. Yet width times height is always one, so the integral limit is one. An integrable common dominator cannot exist, since it would contradict dominated convergence.

$$
f_n=n\mathbf1_{(0,1/n)},\qquad\lim_n\int f_n=1\ne0=\int\lim_n f_n.
$$

## Monotone convergence and Fatou

For an increasing nonnegative sequence, monotone convergence permits exchanging limits and integrals, including infinite values. The third example is min(n,1/√x) for x>0 and zero at zero. Its integral is 2−1/n, approaching two. Dominated convergence also applies here because 1/√x is integrable. The graph omits the part of the limit curve above height 2n near zero; the omitted part is not zero. Fatou supplies a one-sided bound for general nonnegative sequences.

$$
\int\liminf_n f_n\le\liminf_n\int f_n,\qquad 0\le f_n\uparrow f\implies\int f_n\uparrow\int f.
$$

## Derive Fatou's bound from monotone convergence

For nonnegative measurable functions f_n, set u_m=inf{f_n:n≥m}. Countable infima are measurable, u_m increases to liminf f_n, and u_m≤f_n for every n≥m. Monotone convergence therefore gives the following chain, including extended nonnegative values:

$$
\int\liminf_n f_n\,d\mu
=\lim_m\int u_m\,d\mu
\le\lim_m\inf_{n\ge m}\int f_n\,d\mu.
$$

The final term is the liminf of the integrals. This proves Fatou's inequality using monotone convergence; it does not assume pointwise convergence or finite integrals. Monotone convergence itself is the foundational theorem used at this step.

## Use Fatou to prove L1 dominated convergence

Now let f_n and f be real-valued measurable functions, with f_n→f almost everywhere and |f_n|≤g for nonnegative g in L1. Outside one null set, all these bounds and the limit hold together, giving |f|≤g. In particular f and every f_n are integrable. Set d_n=|f_n−f|. Since 0≤d_n≤2g, apply Fatou to the nonnegative functions 2g−d_n:

$$
2\int g\,d\mu
\le\liminf_n\left(2\int g\,d\mu-\int d_n\,d\mu\right)
=2\int g\,d\mu-\limsup_n\int d_n\,d\mu.
$$

The integral of g is finite, so cancellation is legitimate and forces the nonnegative distances to tend to zero. Finally, |∫f_n−∫f|≤∫|f_n−f| proves convergence of integrals. The finite common bound does real work; subtracting two infinite quantities would not be a proof.

## Control tails with uniform integrability

On a probability space, a family of integrable X_n is uniformly integrable when large-value contributions vanish uniformly in n:

$$
\lim_{K\to\infty}\sup_n E[|X_n|\mathbf1_{\{|X_n|>K\}}]=0.
$$

Here is a useful sufficient result with explicit assumptions: if X is integrable, X_n→X in probability and this family is uniformly integrable, then X_n→X in L1. To prove it, let A_n={|X_n−X|>δ}. Truncate both magnitudes at K to obtain

$$
E|X_n-X|\le\delta+2K P(A_n)
+E[|X_n|\mathbf1_{\{|X_n|>K\}}]
+E[|X|\mathbf1_{\{|X|>K\}}].
$$

For any desired error, first choose δ small, then K large to control both tails, and finally n large for the probability term with this fixed δ and K. This order of choices proves the claim. It also explains why convergence in probability alone can miss a large expected loss on a rare event.

## L1 convergence can hold without one integrable dominator

For a separate example, take Ω={1,2,…} with P({k})=2⁻ᵏ and define X_n=(2ⁿ/n)1{ω=n}. Each fixed outcome is eventually outside the support, and E|X_n|=1/n→0.

| n | Height | Mass | Mean |
|---|---|---|---|
| 2 | 2 | 1/4 | 1/2 |
| 4 | 4 | 1/16 | 1/4 |
| 8 | 32 | 1/256 | 1/8 |

Any common dominator g must satisfy g(k)≥2ᵏ/k, so E[g]≥∑1/k=∞. For example, each block from N+1 to 2N contributes at least 1/2 to that harmonic sum. Thus no integrable common dominator exists, although L1 convergence holds. The family is uniformly integrable: above K≥2 its largest tail expectation is 1/n(K), where n(K) is the first index with 2ⁿ/n>K, and n(K)→∞. Dominated convergence is a sufficient condition, not a necessary one for every convergent sequence.

## Verify the discrete family with exact fractions

The code computes individual expectations, finite envelope integrals and uniform tails. The divergence and convergence statements use the arguments above; a finite run alone proves neither. This discrete extension is separate from the three Lebesgue-measure curves in the existing controls.

```python
from fractions import Fraction as Q

def amplitude(n):
    return Q(2**n, n)

for n in [2, 4, 8]:
    mass = Q(1, 2**n)
    assert mass*amplitude(n) == Q(1, n)
    print("n, height, atom mass, expectation:", n, amplitude(n), mass, Q(1, n))
for N in [2, 4, 8, 16]:
    envelope_integral = sum(Q(1, k) for k in range(1, N+1))
    next_block = sum(Q(1, k) for k in range(N+1, 2*N+1))
    assert next_block >= Q(1, 2)
    print("finite envelope:", N, envelope_integral, "next block:", next_block)
for K in [2, 4, 16, 256]:
    n = 2
    while amplitude(n) <= K:
        n += 1
    # Amplitudes increase for n >= 2; tail expectations 1/n decrease.
    ui_tail = Q(1, n)
    assert all((Q(1, j) if amplitude(j) > K else 0) <= ui_tail
               for j in range(1, n+10))
    print("threshold, first tail index, uniform tail:", K, n, ui_tail)
```

## Further questions and research use

1. Why may the DCT proof cancel 2∫g? **The common dominator is integrable, so this quantity is finite.**
2. Does a bound on E|X_n| alone imply uniform integrability? **No: the original height-n spike has expectation one but retains all its mean in arbitrarily large tails.**
3. What should be checked for an expected-loss limit? **Specify the convergence mode and a valid common bound or tail-control argument. A small rare-event probability is insufficient by itself.**

For the foundational convergence theorems, see [Tao, §1.4, Corollary 1.4.47 and Theorem 1.4.49](https://terrytao.wordpress.com/wp-content/uploads/2012/12/gsm-126-tao5-measure-book.pdf). Uniform integrability is developed in [Durrett, §4.6](https://sites.math.duke.edu/~rtd/PTE/PTE5_011119.pdf). The discrete family and tail calculations above are direct examples.

## Exercises with explanations

1. Is the pointwise limit of x^n zero everywhere? **No: at one it is one.**
2. Does the concentrating example violate Fatou? **No: zero is at most one.**
3. What is the third example’s integral at n=4? **1.75.**

Further reading: [Durrett, Probability: Theory and Examples](https://sites.math.duke.edu/~rtd/PTE/pte.html); [Tao, An Introduction to Measure Theory](https://terrytao.wordpress.com/wp-content/uploads/2012/12/gsm-126-tao5-measure-book.pdf).
