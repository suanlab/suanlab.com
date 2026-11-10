## 1. A regional average is a map with missing streets

Suppose a city dashboard replaces every street measurement with one neighborhood average. You can compare neighborhoods, but the average cannot tell you which street lies beside a boundary. If interactions cross that boundary, the missing detail can affect predictions about neighboring regions.

Spatial coarse-graining raises the same question. We replace microscopic spins by block averages, then ask which statistical law those averages obey. The answer requires summing the hidden states with their actual weights. Counting how many states were hidden is sufficient only when those weights are equal.

This preparation draft extends the earlier chapters on multiplicity and conditional entropy. Its finite calculation foundation has independent checks. Interactive controls, visual experiments and the published reader remain pending. You need basic probability, logarithms and finite sums; no quantum spin algebra is used.

## 2. Three distinct meanings of magnetization

A classical spin is a number σ_i=−1 or +1. Divide a finite lattice into B disjoint rectangular blocks, each containing v sites. Within block b, let k_b be the number of +1 spins.

$$
M_b=\sum_{i\in b}\sigma_i=2k_b-v,\qquad
m_b=\frac{M_b}{v},\qquad k_b\in\{0,1,\ldots,v\}.
$$

M_b is a block total, m_b a block average, and ⟨m_b⟩ an ensemble average over many configurations. A configuration can have m_b=1 while the ensemble mean is zero. The spacing between allowed m_b values is **2/v**, because flipping one spin changes the total by two.

The ordered list k=(k_1,…,k_B) is the block profile. It retains positions. Its sum K=Σ_b k_b, or the overall magnetization 2K−N, discards those positions.

## 3. Give the microscopic model a precise graph

The energy rewards aligned neighbors when J>0 and can favor up spins through a blockwise field H_b:

$$
E(\sigma)=-J\sum_{\langle ij\rangle}\sigma_i\sigma_j
-\sum_bH_bM_b,\qquad P(\sigma)=\frac{e^{-E(\sigma)/T}}{Z}.
$$

We set Boltzmann's constant to one and use T>0. The field is constant within each block. An open chain has only internal edges. A periodic lattice includes every positive-direction edge, including the wraparound edge. A periodic axis of length two has two parallel edges under this convention; merging them changes J's effective meaning.

The calculation supports one or two dimensions, at most 16 spins and four equal rectangular blocks. Each block side must divide its corresponding lattice side. These are computational bounds, not assumptions about real materials. A 4×4 lattice can have four 2×2 blocks. Every profile is retained; none is removed by a probability cutoff.

## 4. Turn Boltzmann factors into exact rational arithmetic

Let e be the number of edges and d(σ) the number whose two spins disagree. Then Σ_edge σ_iσ_j=e−2d. Introduce positive activities

$$
x=e^{-2J/T},\qquad y_b=e^{2H_b/T},\qquad
w_{\mathrm{rel}}(\sigma)=x^{d(\sigma)}\prod_b y_b^{k_b}.
$$

For x=1/2, each additional disagreeing edge halves the relative weight. For y_b=3, each additional up spin in that block multiplies it by three. x<1 represents ferromagnetic coupling; x>1 represents antiferromagnetic coupling. x=1 removes the interaction.

Rational x,y allow exact finite sums even though the associated J/T and H_b/T involve logarithms. Activities are not the energy parameters themselves. Holding x fixed while varying T also changes physical J; that operation is not a fixed-Hamiltonian temperature derivative.

## 5. A profile is a bag of weighted configurations

For a fixed profile k, count microscopic states by their disagreement number:

$$
Q_k(x)=\sum_d n(k,d)x^d,\qquad
w_k=Q_k(x)\prod_b y_b^{k_b},\qquad
P(k)=\frac{w_k}{Z_{\mathrm{rel}}},\quad Z_{\mathrm{rel}}=\sum_k w_k.
$$

n(k,d) is an integer. The polynomial Q_k records both multiplicity and interaction energy. At x=1 it becomes Ω_k=Π_b binom(v,k_b), the familiar count. Away from x=1, two states inside the same bag can receive different weights.

Conditioning on k fixes the field contribution. Therefore P(σ|k)=x^d/Q_k, independent of the blockwise fields y_b. A field can make a profile more likely without changing how its internal configurations compete. This cancellation relies on the stated blockwise-constant field assumption.

## 6. Worked example: two blocks, four spins

Use an open chain with blocks (1,2) and (3,4), x=1/2 and zero fields. Choose k=(1,1), meaning one up spin per block. There are four hidden states:

| Spins from left to right | Disagreeing edges d | Relative weight | Probability given k |
|---|---:|---:|---:|
| + − + − | 3 | 1/8 | 1/6 |
| + − − + | 2 | 1/4 | 1/3 |
| − + + − | 2 | 1/4 | 1/3 |
| − + − + | 3 | 1/8 | 1/6 |

Thus Q_(1,1)=2x²+2x³=3/4. The unrestricted chain has Zrel=2(1+x)³=27/4, so this profile has probability 1/9. Its conditional mean d is 7/3 and its conditional variance is 2/9.

The multiplicity four does not imply probability 1/4 per state. The two alternating states pay an extra disagreeing-edge penalty. A useful visual experiment will keep the block colors unchanged while cycling these four microscopic arrangements and highlighting the changed boundary edges.

## 7. The same overall magnetization can hide different spatial patterns

Overall magnetization zero permits three block-total profiles: (+2,−2), (0,0) and (−2,+2). At x=1/2 their weights are 1/2, 3/4 and 1/2. Conditioned on the overall total, their probabilities are 2/7, 3/7 and 2/7. The unconditional probability of that total is 7/27.

The profile (+2,−2) is a domain wall: ++|−−. The profile (0,0) includes both alternating and less costly internal arrangements. One specific domain-wall configuration has four times the weight of one specific alternating configuration, but the entire (0,0) bag has more weight than either saturated-block bag. Always specify whether a comparison concerns one configuration or a whole profile.

This is why a single global order parameter cannot describe domain structure. The block profile preserves some spatial detail; it still hides the ordering inside each block.

## 8. Absolute free energy needs the removed common factor

Relative weights determine probabilities. Absolute free energy also needs the common factor removed from the physical Boltzmann weights:

$$
Z=x^{-e/2}\prod_b y_b^{-v/2}Z_{\mathrm{rel}},\qquad
\frac{F_k}{T}=-\log w_k+\frac e2\log x
+\sum_b\frac v2\log y_b.
$$

Here exp(−F_k/T) is the physical constrained partition sum, not a normalized probability. It satisfies Z=Σ_k exp(−F_k/T). The microscopic and profile reference measures both count their discrete states once. Replacing the profile sum by an integral requires a separate measure convention and approximation.

The finite engine returns exact rational operands for these logarithms and the exact rational value of Z². Z itself need not be rational when e or v is odd. A display must not label a floating-point logarithm as an exact rational free energy.

## 9. Effective free energy includes hidden entropy

For the selected profile, average energy using its conditional microscopic distribution. The definition of F_k gives an exact identity:

$$
\frac{F_k}{T}=\frac{\langle E\rangle_k}{T}-S(\sigma\mid k),\qquad
S(\sigma\mid k)=\log Q_k-\langle d\rangle_k\log x.
$$

In the four-spin example, S(σ|k)=log 3+(log 2)/3, which is less than log 4. The distribution is uneven, so it contains less uncertainty than four equally likely states.

For all profiles together, S(σ)=S(k)+Σ_kP(k)S(σ|k). The entropy omitted by keeping k is precisely the weighted conditional entropy. That information identity alone does not determine the spatial form of F_k; the restricted sums are still needed.

## 10. Boundary spins need not equal their block average

Take four open-chain spins in a single block, with exactly three up spins and x=1/2. The minority down spin can sit at either endpoint, creating one disagreement, or at either interior site, creating two. Endpoint placements each have conditional probability 1/3; interior placements each have probability 1/6.

Therefore the conditional spin means are (1/3,2/3,2/3,1/3), while the block average is 1/2. An interaction crossing the block boundary sees an endpoint, not a uniformly sampled site. Replacing every boundary spin by m_b is an additional approximation.

A second example makes this visible with three two-spin blocks: ++|? ?|−−. Fix the middle block average to zero. The middle arrangements +− and −+ have probabilities 4/5 and 1/5, so their two site means are +3/5 and −3/5. A block colored with its zero average would conceal this internal response to the neighbors.

## 11. A concrete local reference model

First compute Q_b(k_b;x) on each isolated block's induced internal graph. Preserve periodic edges lying wholly inside a block. Approximate cross-block spin products by m_bm_c. This defines a particular reference weight:

$$
\bar d_{\mathrm{cross}}=\frac12\sum_{\mathrm{cross\ edges}}(1-m_bm_c),\qquad
w_k^{\mathrm{ref}}=\left[\prod_b Q_b(k_b;x)y_b^{k_b}\right]x^{\bar d_{\mathrm{cross}}}.
$$

This definition keeps the internal constrained sums exact while approximating cross-block interactions. It is one model to test; it is not the exact result of discarding hidden spins.

Write dbar=a/b in lowest terms. Then R=[Q_k/ΠQ_b]^b/x^a is rational and log(w_k/wref_k)=log R/b. For the four-spin zero-block profile, R=9/8 and b=2. The all-down profile has R=1, so the discrepancy cannot be repaired by a single common normalization constant. The residual is an unnormalized log-weight error, not a KL divergence; no general sign bound is asserted.

## 12. Where a gradient penalty comes from

Let c_bc count edges crossing between blocks b and c, with each unordered block pair counted once. The reference interaction has the exact algebraic decomposition

$$
-J\sum_{b<c}c_{bc}m_bm_c
=\frac J2\sum_{b<c}c_{bc}(m_b-m_c)^2
-\frac J2\sum_b\mathrm{degree}_b m_b^2.
$$

For J>0 the first term penalizes differences. It is a discrete relative of a gradient-squared term. The second shifts local quadratic terms. At an open boundary the degree can change, so that shift need not be spatially constant. For J<0 the sign of the difference term reverses; a slowly varying ferromagnetic field is then a poor default description.

Moving from this reference to a continuum free-energy density additionally needs a separation of scales, a slowly varying field, a measure convention and a justified truncation of powers and derivatives. A finite algebraic identity does not establish these assumptions.

## 13. An exact test that every nearest-neighbor pair model must pass

Could a better choice of local potentials repair the reference? Choose nonneighbor blocks A and C and fix every other block. Any strictly positive probability that factorizes into site factors and nearest-neighbor pair factors must make A and C conditionally independent.

For four choices A=±v,C=±v, independence requires the cross ratio

$$
\mathrm{OR}=\frac{w_{++}w_{--}}{w_{+-}w_{-+}}=1.
$$

Normalizations and one-block fields cancel. With six open-chain spins in three blocks of two, fix middle M=0. At x=1/2 the exact ratio is 16/25. Thus no positive nearest-neighbor pair model on that coarse open chain can reproduce the entire exact block distribution.

This conditions on the middle profile, not on an additional fixed overall magnetization. Conditioning on a global total could introduce a separate dependence and would answer a different question.

## 14. What the obstruction does and does not prove

The calculation shows that the selected coarse variables can require longer-range or multiblock terms. Microscopic nearest-neighbor interactions do not guarantee nearest-neighbor pair closure after hidden spins are summed out.

If the same three blocks have periodic boundaries, A and C are neighbors. Their nonunit cross ratio is then permitted by a pair interaction; it is not an obstruction. Conversely, one cross ratio equal to one does not establish independence for all values or all pairs.

These finite examples do not invalidate effective local continuum descriptions at a suitable scale. They identify an assumption that must be checked or controlled, rather than inferred solely from the original interaction graph.

## 15. A self-contained Python experiment

This code uses only Python's standard library. It enumerates six spins directly, so it is also a transparent small reference for the faster calculation. The application foundation uses Gray-code updates and a separate row-transfer check. This snippet is an executable example, not a newly published Colab course.

```python
from collections import defaultdict
from fractions import Fraction as F
from itertools import product

def chain_weights(n, block_size, x):
    weights = defaultdict(F)
    for spins in product((-1, 1), repeat=n):
        counts = tuple(sum(s == 1 for s in spins[j:j + block_size])
                       for j in range(0, n, block_size))
        d = sum(a != b for a, b in zip(spins, spins[1:]))
        weights[counts] += x ** d
    return weights

x = F(1, 2)
w4 = chain_weights(4, 2, x)
assert w4[1, 1] == F(3, 4)
assert sum(w4.values()) == F(27, 4)
assert w4[1, 1] / sum(w4.values()) == F(1, 9)
total_zero = sum(w for ks, w in w4.items() if sum(ks) == 2)
assert total_zero / sum(w4.values()) == F(7, 27)
assert [w4[k] / total_zero for k in [(2, 0), (1, 1), (0, 2)]] == [F(2, 7), F(3, 7), F(2, 7)]
w6 = chain_weights(6, 2, x)
odds = w6[2, 1, 2] * w6[0, 1, 0] / (w6[2, 1, 0] * w6[0, 1, 2])
assert odds == F(16, 25)
print('profile probability:', w4[1, 1] / sum(w4.values()))
print('nonneighbor conditional cross ratio:', odds)
```

After running the assertions, compute a fresh set of weights with `chain_weights(6, 2, F(1))` and recalculate the cross ratio. It becomes one and all microstates inside a profile become equally weighted. Then try F(2): the finite calculation remains valid, but an alternating pattern is energetically favored. Do not interpret either small system as a measured critical point.

## 16. Plan the visual comparisons around questions

| Comparison | What changes | What to inspect |
|---|---|---|
| Arrangement versus profile | Cycle hidden states at fixed k | Edge disagreements and conditional probabilities |
| Global total versus spatial profile | Keep overall M fixed | Different block patterns and summed weights |
| Endpoint versus average | Select the three-up, one-block case | Conditional site means 1/3 and 2/3 |
| Exact versus local reference | Change x and block size | Exact residual operands, not rounded equality |
| Open versus periodic | Change the coarse graph | Whether witness blocks are actually neighbors |

The planned lattice view will use the existing site's theme. A numerical table must remain available when animation is paused or WebGL is unavailable. Selected-state examples are the smallest numeric masks, not random draws. A profile selector must not suggest that an ensemble mean is an individual configuration.

## 17. Finite symmetry and uncertainty

At zero field, flipping every spin leaves the energy unchanged. Every profile k has a partner v−k with equal probability. Consequently ⟨m_b⟩=0 in this finite ensemble. That does not mean each configuration is disordered: strongly weighted profiles can lie near the two ordered extremes.

The model has finitely many positive weights at every allowed x,y. It does not contain a thermodynamic singularity or establish spontaneous symmetry breaking. Infinite-volume limits, field limits and finite-size scaling belong to later treatments. An animation that settles on one ordered state must not replace the full symmetric distribution.

## 18. Connections to AI models and research

Pooling turns many features into a smaller representation. Latent-variable marginalization turns a simple joint graph into a potentially more complicated observed law. Both resemble the finite construction here. The exact block map gives a controlled example for asking whether a retained representation supports the assumed prediction rule.

This is an analogy, not a theorem about all neural representations. In a learned model, the map, the conditional distribution and its loss must be specified. A useful research question is whether a chosen representation permits a sparse interaction model within an explicit error budget. The exact distribution here can serve as a small benchmark for that question; it does not establish a universal architecture or scaling law.

## 19. Explained exercises

1. **Why is the spacing 2/v rather than 1/v?** One spin changes from −1 to +1, increasing M by two; dividing by v gives 2/v.
2. **Why does y disappear from P(σ|k)?** Its entire product is constant within that profile and cancels between numerator and denominator. Site-varying fields would require a different calculation.
3. **Is Q_k the multiplicity?** Only at x=1. In the four-spin example Ω=4 but Q=3/4 at x=1/2.
4. **What is the probability of global M=0?** Sum all three compatible profile weights, giving (7/4)/(27/4)=7/27. The selected profile alone has probability 1/9.
5. **Can one common constant fix the mean-product reference?** No: the zero-block profile has residual log(9/8)/2 while a saturated profile has residual zero. Their relative weights already disagree.
6. **Why does OR=16/25 rule out coarse nearest-neighbor pair closure?** The end blocks are nonneighbors; with the middle block fixed, any such factorization separates their weights, forcing OR=1.
7. **Does OR≠1 rule out that factorization on a periodic three-block chain?** No. The tested end blocks are neighbors in that graph, so their interaction is allowed.
8. **Does ensemble mean zero prove no ordered configurations occur?** No. Spin-flip symmetry pairs ordered configurations with their opposites. Their mean cancels even if both are highly probable.

## 20. Reading and verification limits

[Tong, Statistical Field Theory §1.1.1 and §1.3](https://davidtong.org/teaching/statistical-field-theory/sfthtml/S1) develops the passage from a global magnetization to a spatial order parameter. [Kadanoff's 1966 paper, pp.264–265](https://doi.org/10.1103/PhysicsPhysiqueFizika.2.263) makes explicit modeling assumptions about cell variables and interactions. The exact small-system calculations and counterexamples above are independently derived for this chapter.

The current foundation compares 24 models, including two-dimensional grids, transposed coordinates, periodic axes of length two, odd block volumes and rational activity bounds. Agreement of exact finite sums does not verify a browser implementation, an interacting continuum integral, critical exponents or a full renormalization procedure. Those remain separate requirements.
