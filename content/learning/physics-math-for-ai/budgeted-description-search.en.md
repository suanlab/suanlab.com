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

## When can a discovered bound be certified optimal?

In a fixed finite candidate family, suppose a verified description of length b is found. If every strictly shorter candidate has been executed and none produces the target, b is the family minimum. Unchecked programs of equal or greater length cannot improve it. This is a sufficient certificate; other structural knowledge might sometimes prove optimality sooner.

Shortest-first search certifies its first match because all shorter candidates were tested. Literal-first finds a nine-bit description for zero immediately, but twenty-two shorter candidates remain. At evaluation 257 it finds length three, the grammar's minimum possible program length, so remaining outputs are unnecessary for certification.

| Target | Evaluations | Order | Best found | Shorter-candidate certificate |
|---|---|---|---|---|
| 0 | 1 | literal | 9 | No |
| 0 | 257 | literal | 3 | Yes |
| 173 | 174 | literal | 9 | No |
| 173 | 278 | literal | 9 | Yes |
| 51 | 266 | literal | 8 | Yes |
| 173 | 196 | shortest | 9 | Yes |

The separate code checks this table from the family and visited set; it does not add a certification control to the screen. The finite oracle remains a separately computed reference, not free information credited to the search.

## Specify both length and time limits

Even for a universal machine, bounding program length by B and each simulation by T steps gives a finite computable problem:

$$
K_{U}^{\le B,T}(x)=\min\{|p|:|p|\le B,\ U(p)=x\text{ within }T\text{ steps}\}.
$$

If the set is empty there is no feasible minimum, conventionally +∞. This does not mean unrestricted K_U(x) is infinite. Increasing B or T enlarges the candidate set, so the minimum cannot increase. Once T permits a known valid literal program to finish, a finite bound exists too. A universal procedure certifying that T has reached the unrestricted optimum for every x would conflict with uncomputability.

The current UI budget counts immediately terminating finite commands, rather than the T of this definition. In research, record representation power, search order, execution cost and optimality evidence separately.

**Exercise:** After finding length eight, must all remaining programs of length≥8 be executed? **Explanation:** Not to certify the minimum length. Finding every shortest program would be a different task.

```python
from fractions import Fraction as F

def programs():
    out=[('0'+f'{x:08b}',x) for x in range(256)]
    out += [('10'+str(x),255*x) for x in range(2)]
    out += [('110'+f'{x:02b}',int(f'{x:02b}'*4,2)) for x in range(4)]
    out += [('1110'+f'{x:04b}',int(f'{x:04b}'*2,2)) for x in range(16)]
    return out

def certificate(target,budget,shortest_first=False):
    family=programs()
    if shortest_first: family.sort(key=lambda row:(len(row[0]),row[0]))
    visited=family[:budget]
    best=min((len(code) for code,x in visited if x==target),default=None)
    # Sufficient certificate: every strictly shorter candidate was executed.
    certified=best is not None and all(len(code)>=best for code,x in family[budget:])
    oracle=min(len(code) for code,x in family if x==target)
    if certified: assert best==oracle
    return best,certified

for target,budget,shortest in [(0,1,False),(0,257,False),(173,174,False),
                               (173,278,False),(51,266,False),(173,196,True)]:
    best,certified=certificate(target,budget,shortest)
    print('target, budget, shortest-first, best found, certified:',
          target,budget,shortest,best,certified)
```


## Exercises with explanations

1. Does discovering a nine-bit description for zero establish optimality? **Not with budget-one literal-first.** A three-bit command exists in the same family.
2. Does no match prove a large lower bound? **Generally no.** Arbitrary partial enumeration leaves shorter candidates unchecked. Structured length-first search can justify conclusions limited to the lengths fully exhausted.
3. Can universal search simply wait for every shorter candidate? **Some never halt.** The finite lab's termination guarantee does not generalize.

Further reading: [Grünwald and Vitányi, Algorithmic Information Theory](https://arxiv.org/abs/0809.2754).
