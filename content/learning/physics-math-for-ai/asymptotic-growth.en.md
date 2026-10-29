## An eventual bound is a quantified claim

Asymptotic notation describes how a function grows for sufficiently large inputs. It does not report exact runtime, and the notation by itself does not specify best case, average case or worst case. First define the cost function and its sampling or computational model.

For eventually nonnegative f and positive comparison g, f∈O(g) means there exist C>0 and a threshold n₀ such that f(n)≤C g(n) for every n≥n₀. A finite sample of input sizes cannot by itself establish that universal tail condition.

## Constants and thresholds are witnesses

The lab uses f(n)=an+b with small positive integer a and nonnegative integer b. It compares f(n)/n with a candidate constant C. For C>a,

$$
an+b\le Cn\iff n\ge\frac b{C-a}.
$$

An integer threshold is therefore max(1,ceil(b/(C−a))). For the reference f(n)=2n+3 and C=3, n₀=3 works. Starting at one fails because f(1)=5>3. That failure refutes this particular pair of witnesses, not membership in O(n).

For C=2 and b=3, no finite threshold works: 2+3/n always exceeds two. Still, C=3 gives a valid O(n) bound. Confusing a failed certificate with impossibility reverses the quantifiers.

## Big-O is not a tight equality

Since 2n≤2n+3≤5n for n≥1, this function is Θ(n): it has both an eventual upper and lower bound by constant multiples of n. Ω denotes the eventual lower-bound counterpart under the stated convention.

The same function is also O(n²), because n≤n² for n≥1. A true Big-O statement need not be the tightest useful description. Writing f(n)=O(g(n)) is customary notation for a class-membership claim, not equality with one ordinary function.

The literal logical negation of an eventual upper bound is

$$
\forall C>0\ \forall n_0\ \exists n\ge n_0:
 f(n)>C g(n).
$$

For f(n)=n²,g(n)=n, choose an integer n larger than both C and n₀. Then n²>Cn. This proves n² is not O(n), rather than merely showing a few large ratios.

## Compare familiar growth families

For logarithmic growth comparisons use fixed bases greater than one. Changing between them multiplies the positive logarithm by a positive constant, so its Big-O class is unchanged. Polynomial degree and exponentials differ more fundamentally. For example, at n=8, n²=64 and 2ⁿ=256; at n=16 they are 256 and 65,536. These examples illustrate a difference but do not prove an asymptotic theorem by themselves.

A rigorous comparison of exponential and polynomial growth requires an argument valid beyond a threshold, using ratios, bounds or later analytic limits. Do not promote a finite log-log slope fit into a universal complexity theorem.

Little-o is stricter than Big-O: under positive comparison functions, f=o(g) means the ratio tends to zero. A fixed nonzero multiple of n is O(n) but not o(n). Specify the exact convention when comparing statements from different sources.

## Read the ratio plot

Blue is f(n)/n across integer n=1,…,128; orange is candidate C. The chosen n₀ is tested analytically in the readout, not certified by the finite drawn interval. Ratios can be below C in a plotted prefix without establishing all future inputs for an arbitrary unknown function.

For this explicit linear family, the algebra supplies the universal result. Lab controls use integer constants to keep threshold comparisons exact at the illustrated boundaries. The cost values are synthetic operation counts, not measurements of a real computer.

## AI systems and research comparisons

Computational complexity depends on what grows: examples, tokens, dimensions, parameters or search depth. A claim O(n) is incomplete if n is unnamed. Memory, communication and arithmetic costs can scale differently, and hardware constants can dominate finite problem sizes.

State implementation assumptions, sparsity and precision. Measured runtime supports an empirical claim under specified conditions; an asymptotic bound supports a mathematical claim about a model. Neither automatically implies lower energy use or better predictive quality.

## Exercises and reference

1. For f(n)=3n+4,C=5, find a valid integer threshold. **Answer:** n₀=2.
2. Is n also O(n²)? **Answer:** yes for n≥1, though the upper bound is loose.
3. Write the quantifiers needed to refute a proposed Big-O class.
4. For an AI operation, name the input-size variables and distinguish its theoretical count from a wall-clock measurement.

For asymptotic reasoning and algorithmic context see [MIT Mathematics for Computer Science](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/resources/mit6_042js15_textbook_pdf/). The linear certificates and quantifier argument above are directly derived examples.
