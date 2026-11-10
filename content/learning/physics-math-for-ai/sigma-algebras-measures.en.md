## Choose measurable events before a measure

A sample space alone does not assign probabilities to every subset. A sigma-algebra specifies the events on which a measure is defined. It contains the whole space and is closed under complements and countable unions. On Ω={0,1,2,3}, suppose information distinguishes {0,1} from {2,3} but not the points within either pair. Only the four unions of these two atoms are measurable events.

$$
\mathcal F=\{\varnothing,\{0,1\},\{2,3\},\Omega\}.
$$

## Atoms and information resolution

A finite sigma-algebra is described by its minimal distinguishable blocks, or atoms. With k atoms there are 2 to the k possible unions. Full information gives four singleton atoms and sixteen events. Trivial information has one atom, Ω, and only two events. General infinite measurable spaces need not have this finite atomic description.

$$
|\mathcal F|=2^k\quad\text{for a finite partition with }k\text{ atoms}.
$$

## Measures are assigned to measurable sets

A measure is a nonnegative countably additive function on the sigma-algebra, with empty-set measure zero. Total measure one makes it a probability measure. The underlying weights (0.1,0.2,0.3,0.4) aggregate to atom masses 0.3 and 0.7. They construct a measure on the coarse sigma-algebra. Individual weights in the table come from a chosen finer extension; they do not make {0} measurable under the coarse information. That event’s measure is left undefined.

$$
\mu\!\left(\bigsqcup_n A_n\right)=\sum_n\mu(A_n),\qquad P(\Omega)=1.
$$

## Which questions does measurability permit?

A real function is measurable when the preimage of every Borel set is an allowed event. On a finite information atom it must be constant: different values could be separated by a real interval whose preimage splits the atom. This connects measurability to the information available to a predictor. Define observable distinctions before forming a conditional prediction or an information-compressed representation.

$$
X:(\Omega,\mathcal F)\to(\mathbb R,\mathcal B(\mathbb R))\text{ measurable}\iff X^{-1}(B)\in\mathcal F\ \forall B\in\mathcal B(\mathbb R).
$$

## Exercises with explanations

1. Is the union of both atoms measurable? **Yes: it is Ω, with probability one.**
2. Does nonmeasurability of {0} mean probability zero? **No: it lies outside this measure’s domain.**
3. How many events does full information allow? **Sixteen.**

Further reading: [Durrett, Probability: Theory and Examples](https://sites.math.duke.edu/~rtd/PTE/pte.html); [Tao, An Introduction to Measure Theory](https://terrytao.wordpress.com/wp-content/uploads/2012/12/gsm-126-tao5-measure-book.pdf).
