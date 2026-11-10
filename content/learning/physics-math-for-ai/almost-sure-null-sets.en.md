## A weaker claim than having no exceptions

A property holds almost everywhere when its failure set is contained in a measurable null set. Under a probability measure this is called almost surely. Exceptional points are not deleted from the sample space. The lab uses the full power-set sigma-algebra and point masses (1/2,1/2,0,0). The last two points exist but together have probability zero.

$$
X=Y\quad P\text{-a.s.}\iff P(\{X\ne Y\})=0\quad\text{for measurable }X,Y.
$$

## Compare pointwise and integral differences

The defaults f=(0,1,2,−1) and g=(0,1,0,0) differ at the last two points. Their maximum absolute difference is two, but their probability-weighted L1 distance is zero and both expectations are one half. Changing the exceptional values preserves almost-sure equality and the integrals. Pointwise equality requires matching the exceptional values too.

$$
E|f-g|=\sum_i p_i|f_i-g_i|=0,\qquad E[f]=E[g]=\tfrac12.
$$

## Why L1 identifies these functions

The L1 size of an integrable function is its absolute integral. Distinct pointwise functions can have zero difference under this quantity, so it is a seminorm before quotienting. Identifying functions equal almost everywhere turns it into a norm. Conditional expectations and Radon–Nikodym derivatives similarly have uniqueness only up to null-set changes.

$$
\|f\|_{L^1}=\int|f|\,dP,\qquad f\sim g\iff f=g\ P\text{-a.s.}
$$

## Completeness and interpretation in learning

A measure space is complete if every subset of every measurable null set is measurable. This finite power-set space is complete. On an incomplete space, arbitrary changes on subsets of null sets can break measurability, so the modified function must still be checked. An expected training objective may ignore a null region of its training law. That does not make the region irrelevant under another law or deployment distribution.

## Exercises with explanations

1. Can maximum difference three coexist with L1 distance zero? **Yes, if differences lie on a null set.**
2. Which measure determines almost-sure equality? **The specified P.**
3. Does a failure at an exceptional point refute an a.s. claim? **Not by itself: the claim is not pointwise equality.**

Further reading: [Durrett, Probability: Theory and Examples](https://sites.math.duke.edu/~rtd/PTE/pte.html); [Tao, An Introduction to Measure Theory](https://terrytao.wordpress.com/wp-content/uploads/2012/12/gsm-126-tao5-measure-book.pdf).
