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

## Exercises with explanations

1. What happens when both noises flip? **The original bit is restored.**
2. Does always-flipping postprocessing erase information? **No: it is invertible.**
3. Does the same comparison automatically hold if Z receives X again? **No: recheck the Markov condition.**

Further reading: [MIT 6.441, Information Theory — lecture notes by Polyanskiy and Wu](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/pages/lecture-notes/).
