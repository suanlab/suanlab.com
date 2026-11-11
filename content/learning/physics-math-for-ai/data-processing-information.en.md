## Can postprocessing create source information?

Generate Y from input X, then generate Z from Y alone. The essential restriction is that Z has no additional access to X. This is the Markov condition X→Y→Z, equivalently conditional independence of X and Z given Y.

$$
I(X;Z)\leq I(X;Y).
$$

## Calculate two flip channels

Use X~Bernoulli(p) and independent noises: Y=X⊕N₁, Z=Y⊕N₂. The final bit differs exactly when one noise flips, so effective error is ε₁(1−ε₂)+(1−ε₁)ε₂.

At p=.5, ε₁=.1 and ε₂=.2, effective error is .26. First-stage information is about .531 bits, while final information is 1−h₂(.26)≈.173 bits. The eight table masses are P(x)P(y|x)P(z|y) and sum to one.

## Derive the inequality through chain rules

Expand information between X and the pair (Y,Z) in two orders. The Markov condition makes one conditional information term zero; the remaining conditional information in the other expansion is nonnegative.

$$
I(X;Y,Z)=I(X;Y)+I(X;Z\mid Y)=I(X;Y),
$$
$$
I(X;Y,Z)=I(X;Z)+I(X;Y\mid Z)\geq I(X;Z).
$$

The argument applies beyond symmetric noise. All variables here are finite, avoiding ambiguities involving differences of infinite entropies.

## Preserved and erased information

At ε₂=1, Z=1−Y is invertible and loses no information. At ε₂=.5, Z becomes independent of the source. Blue plots final information against the second noise rate; orange is first-stage information as an upper bound. Some noninvertible processing can also attain equality when it preserves relevant information under the distribution.

## Interpreting learned representations

The result can describe fixed preprocessing or downstream transformations, but extra access to X, labels or other data changes the Markov assumptions. If Y is noise and Z directly copies X, final information can exceed first-stage information. This violates the condition, not the theorem. Information preservation also differs from task convenience or computational cost.

## Equality requires less than an inverse map

Keep a finite Markov chain X→Y→Z. The chain rule above identifies the information loss as I(X;Y|Z). Rewriting conditional mutual information as an average of posterior KL divergences reveals what must be preserved. Sum only over pairs with p(y,z)>0, using base-two logarithms.

$$
I(X;Y)-I(X;Z)=\sum_{y,z}p(y,z)
D_{\rm KL}\!\left(P_{X\mid Y=y}\Vert P_{X\mid Z=z}\right).
$$

The definition initially contains P(X|y,z); the Markov condition replaces it by P(X|y). Every KL is nonnegative and vanishes exactly when the two posteriors agree on each active pair. Equality therefore holds precisely when P(X|y)=P(X|z) almost surely, equivalently X→Z→Y also holds. Recovering all of Y is unnecessary if the posterior about X is preserved. This conclusion concerns the fixed joint law, not automatically sufficiency for every parameter and prior.

## Merge observations with identical posteriors

Give Y's three states a,b,c masses (1/4,1/4,1/2), with P(X=1|Y)=(1/5,1/5,4/5). These specify a valid joint law and P(X=1)=1/2. Reading X as the source, obtain P(Y|X) by dividing joint masses by P(X). Postprocessing is a deterministic function of Y alone.

- Preserving merge: send a,b to z=0 and c to z=1. The merged states had identical posteriors, leaving posteriors (1/5,4/5).
- Lossy merge: send a,c to z=0 and b to z=1. The first posterior becomes (1/4×1/5+1/2×4/5)/(3/4)=3/5.

![Merging a,b with equal posterior 1/5 preserves information; merging a,c with different posteriors changes the result to 3/5 and loses information.](/assets/learning/information-posterior-merges-en.svg)

| Map | I(X;Z) | Loss |
|---|---|---|
| Y | 0.27807 | 0 |
| a,b | 0.27807 | 0 |
| a,c | 0.09131 | 0.18677 |

Information is in bits. Even the preserving merge cannot recover whether a or b occurred, so some raw state information about Y is lost. The preserved object is information about X. Changing the target variable can make the same merge lose information. Both merges satisfy the original Markov condition.

## Different posteriors identify discarded information

After the lossy merge, observing z=0 gives success probability 3/5 for X. Learning whether the original observation was a or c changes it to 1/5 or 4/5. That is the remaining information I(X;Y|Z)>0. Under the preserving merge, learning a versus b within z=0 leaves the posterior unchanged.

The code constructs rational joint masses and independently compares the entropy-based loss with its posterior-KL average. This is a separate three-state experiment; the XOR controls above do not manipulate these merges.

```python
from fractions import Fraction as Q
from math import log2, isclose

py, posterior = [Q(1, 4), Q(1, 4), Q(1, 2)], [Q(1, 5), Q(1, 5), Q(4, 5)]
joint = [[py[y]*(posterior[y] if x else 1-posterior[y])
          for y in range(3)] for x in range(2)]
def entropy(mass):
    return -sum(float(v)*log2(float(v)) for v in mass if v)
def mi(table):
    px = [sum(row) for row in table]
    pz = [sum(row[z] for row in table) for z in range(len(table[0]))]
    return entropy(px)+entropy(pz)-entropy([v for row in table for v in row])
def kl(p, q):
    return sum(float(a)*log2(float(a/b)) for a, b in zip(p, q) if a)

base = mi(joint)
for mapping in [[0, 1, 2], [0, 0, 1], [0, 1, 0]]:
    out = [[sum(joint[x][y] for y in range(3) if mapping[y] == z)
            for z in range(max(mapping)+1)] for x in range(2)]
    pz = [sum(row[z] for row in out) for z in range(len(out[0]))]
    post = [out[1][z]/pz[z] for z in range(len(pz))]
    gap = sum(float(py[y])*kl([1-posterior[y], posterior[y]],
                               [1-post[mapping[y]], post[mapping[y]]])
              for y in range(3))
    assert isclose(base-mi(out), gap, abs_tol=1e-12)
    if mapping == [0, 0, 1]:
        assert post == [Q(1, 5), Q(4, 5)] and gap == 0
    if mapping == [0, 1, 0]:
        assert post == [Q(3, 5), Q(1, 5)] and gap > .18
    print("map, posterior, retained bits, loss:", mapping, post, mi(out), gap)
```

## Questions for research use

1. Must noninvertible processing lose information about X? **No: states with identical X posteriors can be merged.**
2. How should the preserved object be specified? **Record the source or target X, observation Y, processing Z and joint law.**
3. Does zero loss imply equivalence for every task and computational cost? **No: this comparison concerns distributional information about the specified X.**

For background, see [MIT 6.441, Chapter 3 §3.1–3.2](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/486b7d83cc428acf75b74323ecafcc11_MIT6_441S16_chapter_3.pdf). The merges and numerical values are computed directly from the stated joint law.

## Small composed information and display precision

Channel contrasts 1−2ε multiply under composition. Computation retains this product, preserving small positive information even when the effective flip rounds to 0.5. For p=.5, ε₁=.5−2⁻³⁰, ε₂=.5−2⁻³⁵, final information is approximately 8.48×10⁻³⁹ bits. Do not infer exact independence from the rounded flip readout alone. If mathematically positive information falls below numeric representation range, the selected information and marker are withheld with a numeric-range notice.

## Exercises with explanations

1. What happens when both noises flip? **The original bit is restored.**
2. Does always-flipping postprocessing erase information? **No: it is invertible.**
3. Does the same comparison automatically hold if Z receives X again? **No: recheck the Markov condition.**

Further reading: [MIT 6.441, Information Theory — lecture notes by Polyanskiy and Wu](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/pages/lecture-notes/).
