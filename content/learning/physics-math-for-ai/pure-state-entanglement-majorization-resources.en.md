## 1. State the allowed operations

Alice and Bob hold one copy of a known bipartite pure state and want another bipartite pure state. They may perform local quantum operations and exchange classical messages, abbreviated LOCC. They cannot freely send qubits or create shared entanglement from nothing. The question in this chapter is exact, single-copy conversion under these operations.

This differs from many-copy approximate distillation, mixed-state conversion or conversion with an extra entangled catalyst. The same word “entanglement” appears in these tasks, but their conversion rules need not coincide.

Review [tensor factors and Schmidt decomposition](/book/online/physics-math-for-ai/bipartite-pure-schmidt-partial-trace/).

## 2. Schmidt probabilities are the relevant input

Write the state in Schmidt form:

$$
|\psi\rangle=\sum_{i=1}^d\sqrt{\lambda_i}|i\rangle_A|i\rangle_B,
\qquad\lambda_1\ge\cdots\ge\lambda_d\ge0,\qquad\sum_i\lambda_i=1.
$$

The $\lambda_i$ are probabilities, not the amplitudes $\sqrt{\lambda_i}$. They are eigenvalues of either reduced density matrix. Local basis changes do not change them. Vectors with different listed lengths can be compared after padding the shorter one with zeros in a common local space.

A Bell pair has probabilities $(1/2,1/2)$. A product state has $(1,0)$. The entropy of entanglement is $H(\lambda)$ in bits for a bipartite pure state; that interpretation does not transfer directly to arbitrary mixed states.

## 3. Define majorization with a visible direction

For descending probability vectors, write $\lambda\prec\mu$ when

$$
\sum_{i=1}^k\lambda_i\le\sum_{i=1}^k\mu_i
\quad (k=1,\ldots,d),
$$

with equality for the total sum. A more uniform vector is majorized by a more concentrated one. For example $(1/2,1/2)\prec(1,0)$.

The exact deterministic pure-state LOCC criterion is

$$
|\psi\rangle\longrightarrow|\phi\rangle\quad\Longleftrightarrow\quad
\lambda_\psi\prec\lambda_\phi.
$$

The direction is easy to reverse accidentally. The Bell pair can become a product state by local operations; a product state cannot deterministically become a Bell pair with LOCC alone. Use this physical check whenever implementing the inequality.

## 4. Inspect every prefix, not just the largest weight

Consider

$$
\lambda=(1/2,1/4,1/4),\qquad\mu=(2/5,2/5,1/5).
$$

| $k$ | Prefix of $\lambda$ | Prefix of $\mu$ | $\lambda$ prefix ≤ $\mu$ prefix? |
|---|---:|---:|---|
| 1 | $1/2$ | $2/5$ | no |
| 2 | $3/4$ | $4/5$ | yes |
| 3 | 1 | 1 | yes |

Forward conversion fails at the first prefix. Reverse conversion passes the first prefix but fails at the second because $4/5>3/4$. The pair is incomparable for deterministic single-copy LOCC.

A plot of cumulative sums makes both obstructions visible: the two curves cross. A single scalar score cannot represent this partial order in general.

## 5. Entropy is necessary but not sufficient

For this pair,

$$
H(\lambda)=3/2,\qquad H(\mu)\approx1.521928.
$$

The entropy ordering is consistent with a possible $\mu\to\lambda$ conversion, yet the second prefix forbids it. A nonincreasing entropy is a necessary condition, not a sufficient condition in these dimensions.

For rank-two Schmidt vectors the single nontrivial prefix completely orders the spectra, and entropy captures that order. Do not extrapolate the special two-weight situation to higher dimension. The model reports entropy alongside prefix certificates but never uses it to decide convertibility.

## 6. Probabilistic conversion uses tail constraints

Exact conversion may succeed only on an announced outcome, with another outcome declared failure. For descending spectra define $E_\ell(\lambda)=\sum_{i=\ell}^d\lambda_i$. Vidal's optimal success probability is

$$
p_{\max}(\psi\to\phi)=\min_{\ell:E_\ell(\mu)>0}
\frac{E_\ell(\lambda)}{E_\ell(\mu)}.
$$

The first tail is one for both states, so the minimum is at most one. A zero target tail supplies no restriction and is skipped; do not divide $0/0$. A zero source tail with a positive target tail forces probability zero, expressing the Schmidt-rank obstruction.

The formula identifies an optimal probability under the theorem's assumptions. Computing it does not implement the local measurement sequence that attains it for every input.

## 7. Calculate both success probabilities

For Section 4's pair, the forward tail ratios are

$$
1,\qquad\frac{1/2}{3/5}=\frac56,\qquad
\frac{1/4}{1/5}=\frac54.
$$

Thus $p_{\max}(\lambda\to\mu)=5/6$. Reversing the direction gives $1,6/5,4/5$, so the reverse optimum is $4/5$. Neither deterministic conversion is possible, yet both have positive probabilistic success.

The rates are not reciprocal. Applying a successful conversion and then attempting the reverse has a success probability below one; failed branches must be counted. This does not create an entanglement resource for free.

For a product source $(1,0,0)$ and an entangled target, at least one positive target tail has zero source weight, so the success probability is zero.

## 8. Construct one actual LOCC conversion

Start with a Bell pair and target

$$
|\phi\rangle=\tfrac45|00\rangle+\tfrac35|11\rangle.
$$

Alice uses two measurement operators

$$
M_0=\operatorname{diag}(4/5,3/5),\qquad
M_1=\operatorname{diag}(3/5,4/5).
$$

Their squared operators sum to the identity. Each outcome occurs with probability $1/2$. Outcome0 already produces $|\phi\rangle/\sqrt2$. Outcome1 produces $(3|00\rangle+4|11\rangle)/(5\sqrt2)$. Alice announces the outcome; on outcome1 both parties apply $X$, restoring $|\phi\rangle/\sqrt2$.

This is a deterministic conversion after conditional correction. Its source probabilities $(1/2,1/2)$ are majorized by the sorted target $(16/25,9/25)$. It constructs one useful family, not a general LOCC synthesizer or a proof of Vidal's optimal protocol for all spectra.

## 9. Exercises with worked answers

1. **Probabilities or amplitudes?** What vector belongs to $(4|00\rangle+3|11\rangle)/5$? **Answer:** $(16/25,9/25)$, not $(4/5,3/5)$.
2. **Check direction.** Can a Bell pair become a product state? **Answer:** yes; $(1/2,1/2)\prec(1,0)$. The reverse direction fails.
3. **Incomparability.** Where does $\mu\to\lambda$ fail in Section 4? **Answer:** $k=2$, since $4/5>3/4$.
4. **Tail probability.** Find the forward optimum for that pair. **Answer:** the smallest tail ratio is $5/6$.
5. **Zero tail.** How should a target tail equal to zero be handled? **Answer:** skip it; a zero denominator does not impose a finite ratio constraint.
6. **Actual measurement.** Why does Section 8 not need postselection failure? **Answer:** both outcomes have probability $1/2$ and both can be corrected to the same target, so total success is one.

## 10. Extensions and the boundary of the calculation

Catalysis, approximate transformations, mixed-state entanglement, many-copy distillation and finite-resource rates require additional models and theorems. Exact rational spectra near a prefix equality must not be classified by rounded entropy or rounded cumulative sums. The model preserves exact violations and exact zero tails.

For AI research, the useful analogy is resource-constrained decision making with several incomparable monotones: one aggregate score may hide a forbidden transformation. This is an analogy, not a claim that ordinary model representations obey LOCC.

Primary results are [Nielsen's deterministic conversion theorem](https://arxiv.org/abs/quant-ph/9811053) and [Vidal's optimal probabilistic conversion](https://arxiv.org/abs/quant-ph/9902033). The independent verification uses exact convex combinations of target permutations for deterministic conversion, separately from the implementation's prefix comparisons.
