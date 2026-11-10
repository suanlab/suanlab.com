## The optimum differs from the current discovery

Finding short programs consumes resources. Results under a budget depend on search order as well as the target. We reuse the fixed 278-program decoder and limit evaluations. Every candidate terminates immediately, so universal-program nontermination is absent from this experiment.

A candidate must reconstruct the target byte exactly. We approximate the optimum description length, **not the reconstruction itself**.

$$
D(p)=x\quad\Longrightarrow\quad K_D(x)\le |p|.
$$

## Equal budgets can produce different answers

Shortest-first orders programs by length and then lexically. A first match is already shortest within this finite family. Literal-first visits 256 literals, two uniform commands, four period-two commands and sixteen period-four commands. It may find a long explanation before improving it.

For target zero, the first literal gives nine bits. Evaluation 257 reaches program 100 and improves the bound to three. Shortest-first discovers the three-bit uniform program immediately. In contrast, its first candidate does not produce byte 173.

## Plot bounds and unknowns honestly

Blue shows the best discovered description length; orange shows a separately computed exhaustive finite-family optimum. Before discovery there is no blue point and the readout is —. Unknown is neither zero bits nor infinite complexity.

After discovery, retaining the previous best ensures

$$
B_{t+1}\le B_t\quad\text{once a candidate is found}.
$$

The table retains the first evaluation, improvement events and the final evaluation. Omitted repeated values do not erase their computational cost. At budget 278, both orders reach the same finite optimum.

## Resource limits and research claims

Evaluation count is not elapsed seconds. Real candidates can have different runtimes. Time-bounded universal complexity, such as Kᵗ, defines a separate object; this finite budget experiment does not compute it.

AI-generated descriptions or programs can be executed and verified. Limited search quality does not immediately reveal a representation's fundamental capacity. Align search order, allowed language, evaluation cost and reconstruction criteria before comparing methods.

## Exercises with explanations

1. Does discovering a nine-bit description for zero establish optimality? **Not with budget-one literal-first.** A three-bit command exists in the same family.
2. Does no match prove a large lower bound? **Generally no.** Arbitrary partial enumeration leaves shorter candidates unchecked. Structured length-first search can justify conclusions limited to the lengths fully exhausted.
3. Can universal search simply wait for every shorter candidate? **Some never halt.** The finite lab's termination guarantee does not generalize.

Further reading: [Grünwald and Vitányi, Algorithmic Information Theory](https://arxiv.org/abs/0809.2754).
