## Synthesis needs a grammar, a specification and a verifier

A program can fit its examples, type check and still implement the wrong function. Program synthesis asks for a program satisfying a specification within a stated language. This unit separates candidate construction from verification, then uses counterexamples to guide the next proposal. The method is counterexample-guided inductive synthesis, or CEGIS.

Our finite language is a parameterized polynomial DSL:

$$
G_k=\{p(x)=ax^2+bx+c:\ a,b,c\in\mathbb Z,\ -k\le a,b,c\le k\}.
$$

Bounds k=1,2,3 give27,125,343 programs. The specification is one of2x+1, x squared, absolute value or3x+1. The verifier checks only the explicitly chosen integer domain

$$
D_r=\{-r,-r+1,\ldots,r\},\qquad r\in\{0,1,2\}.
$$

Even r=0 contains one input, zero. The synthesis claim is existential over the bounded grammar and universal over this finite domain. It is not a claim over every integer or real input.

## Examples define a version space

Let E be the inputs whose required outputs have already been supplied. Define the remaining candidates by

$$
V(E)=\{p\in G_k:\ \forall x\in E,\ p(x)=\operatorname{spec}(x)\}.
$$

The synthesizer enumerates this set and selects the first candidate. Ordering uses coefficient cost |a|+|b|+|c|, then lexicographic(a,b,c). This is an explicit preference for small coefficients, not a proof that the chosen candidate is correct beyond E or a claim of minimum source-code length. Target coefficients are not copied into the candidate. The synthesizer uses the specified values only at stored examples.

Seed bits1,2,4,8,16 correspond to inputs−2,−1,0,1,2. Requested seeds outside D are ignored and listed separately. A probe from−3 through3 is an independent diagnostic: moving it never changes E or the synthesized program.

## Verification produces a witness of failure

The finite verifier evaluates every input in D in ascending order. If the current program differs from the specification, it returns the first failing input z:

$$
z\in D,\qquad p(z)\ne\operatorname{spec}(z).
$$

The returned input is a concrete counterexample. Its two output values explain the failure. Add z only to the next round's examples. Since the selected program already matches all of E, z cannot already belong to E. Moreover,

$$
V(E\cup\{z\})\subsetneq V(E).
$$

The failed candidate is removed, together with every other candidate that makes the same mistake. A counterexample is therefore useful information for an entire candidate set, not merely a negative score for one proposed program.

If no candidate remains, the bounded grammar cannot satisfy the stored examples and hence cannot satisfy the full finite specification. This is different from an inconsistent specification: the desired function may exist outside the grammar. If a candidate has no failing input in D, the result is verified on D. Both outcomes are exact here because the grammar and verifier domain are fully enumerable.

## Read the two diagrams

The coefficient grid is one fixed-c slice. Columns vary a and rows vary b. A circle fits E; an exclamation mark fits E now but would fail the current counterexample; a star marks the selected candidate; a cross was already eliminated by E. Total counts include all c layers, not just visible cells. With no selected program, the diagram inspects c=0 without inventing a candidate.

The loop diagram shows the current examples, selected program and finite check. The counterexample belongs to the next round, so changing the requested round exposes an actual trace prefix rather than borrowing a future candidate. Requests past the end clamp to the last state.

The comparison plot evaluates seven integer probes from−3 through3. Blue is the current program and orange the specification; these are discrete evaluations, not a continuous proof. Some plotted points may lie outside D. An absent candidate has no blue curve and no numerical output, whereas the zero program is a real candidate with zero-valued outputs.

## Worked experiment: an example is not a specification

Use target2x+1, k=2, radius2 and seed mask4, which supplies only x=0. At round0 the smallest consistent program is p(x)=1. Twenty-five programs fit that one example. At x=−2 the candidate gives1 but the target gives−3, so the verifier returns−2.

Round1 uses E={−2,0}. Only three programs remain. The preferred candidate1−x squared fits those points, but at−1 it gives0 instead of−1. Round2 uses {−2,−1,0}, leaving coefficients(0,2,1). The program2x+1 matches all five verifier inputs.

The displayed coefficient slice moves from c=1 with25 live cells to three, then one. At round0,22 candidates in that slice fail the new counterexample, including the selected constant. Their removal is shown only after advancing to round1.

## Worked experiment: finite success and an outside failure

Use absolute value, k=2 and radius1. Starting with x=0, proposals progress from zero to−x and finally to x squared. The final program matches−1,0,1 and has a valid finite-domain certificate. The independent probe x=2 gives4 while absolute value gives2. That outside failure does not invalidate the certificate's stated domain; it invalidates extending the claim beyond it.

Now expand the verifier to radius2. No bounded quadratic candidate matches absolute value on all five points. The final candidate set becomes empty. This does not make absolute value inconsistent: a language with an absolute-value constructor can express it directly.

## Worked experiment: expand the grammar rather than change the target

Target3x+1 is also a well-defined function. With k=2 and radius2, no candidate can match the five-point specification. Raising k to3 permits coefficients(0,3,1), and the same counterexample loop finds it. Keep the target fixed while changing the grammar to see which assumption caused the original synthesis failure.

Grammar expansion increases the initial candidate count from125 to343. It need not improve every aspect of runtime in a larger system: candidate order, verification cost and the expressiveness of the language interact. This panel reports exact candidate counts and rounds, not elapsed-time benchmarks.

## Why this finite loop terminates

A successful refinement adds a previously unseen input from D. Thus there can be at most |D|−|E_initial| refinements and at most one additional final proposal state. Here the trace has at most six states. It stops with finite verification or an empty version space. More seed examples can shorten a trace, but their number alone does not prove that they identify the intended function.

The finite-domain assumption matters. Outside this panel, consider synthesizing an integer constant c satisfying c≠x for every integer x. No such constant exists, yet any finite example set allows another c outside that set. An unbounded CEGIS process need not terminate merely because each new counterexample eliminates some candidates. A real verifier can also return unknown or time out; either result must stay distinct from a proof of no counterexample.

## Research extension: exact identification is different from stable identification

Three distinct exact samples identify a quadratic polynomial, but noisy samples can be amplified by interpolation. At nodes $-1,0,1$, the Lagrange basis is $L_{-1}(x)=x(x-1)/2$, $L_0(x)=1-x^2$, $L_1(x)=x(x+1)/2$. If each measured value differs from truth by at most $\delta$, then

$$|\widehat p(x)-p(x)|\le\delta\sum_i|L_i(x)|.$$

This follows directly by expressing the interpolated error as $\sum_i e_iL_i(x)$ and applying the triangle inequality. The bound is sharp at a fixed probe: choose each node error to match that basis sign. Extrapolation can therefore be fragile even when exact noiseless identification is unique.

| Probe $x$ | Worst-case amplification |
|---|---:|
| 0 | 1 |
| 2 | 7 |
| 3 | 17 |

```python
from fractions import Fraction as F
delta=F(1,100); node_errors=[delta,-delta,delta]
for x,expected in [(F(0),1),(F(2),7),(F(3),17)]:
    basis=[x*(x-1)/2,1-x*x,x*(x+1)/2]
    amplification=sum(map(abs,basis))
    error=sum(e*l for e,l in zip(node_errors,basis))
    assert amplification==expected and abs(error)==delta*amplification
    print(x,amplification,error)
for h in [F(1),F(1,10),F(1,100)]:
    quadratic_error=(delta-2*(-delta)+delta)/(2*h*h)
    assert quadratic_error==2*delta/(h*h)
print('Exact sharp extrapolation and coefficient-error checks passed')
```

For nearby nodes $0,h,2h$, the fitted quadratic coefficient has error at most $2\delta/h^2$, while the linear coefficient has error at most $4\delta/h$. Small spacing creates a different conditioning problem. This auxiliary analysis allows real coefficients and noisy data; the native finite integer DSL and exact verifier remain unchanged. A finite verification result neither estimates noise nor supplies an out-of-domain stability certificate.

## Exercise and solution: what extra reasoning would justify generalization?

Exercise: if the candidate and specification are both known polynomials of degree at most two, what does agreement at three distinct inputs imply? Can the same reasoning justify x squared as a global implementation of absolute value?

Solution: their difference has degree at most two. A nonzero polynomial of that degree has at most two distinct roots, so three agreeing points force the difference to be identically zero. This supplies an additional mathematical argument for all real inputs. Absolute value is not known to be a polynomial of degree at most two, so that hypothesis fails. The panel intentionally reports the finite verifier result rather than silently applying a theorem with unstated assumptions.

Additional exercise: set radius0 and seed mask0 for target2x+1. The zero program initially fits the empty example set. The failing input is zero, which must not be confused with “no counterexample.” After adding it, the constant1 verifies on the singleton domain. If all five seeds are requested at radius0, only zero is used and the other four are reported as ignored.

## AI research connections and source

A language model can propose candidates or prioritize the grammar while a verifier supplies counterexamples. A type-correct program still needs to meet its behavioral specification. A correct finite test result, a grammar-exhaustion result and a proof over an unbounded domain have different meanings. Keeping those meanings explicit is essential when using generated programs as research evidence.

[Armando Solar-Lezama's MIT synthesis course, Lecture17](https://people.csail.mit.edu/asolar/SynthesisCourse/Lecture17.htm) describes the interaction between specifications, checking and counterexample-guided refinement. The finite integer grammar, coefficient order, domain controls and worked examples here are constructed for this textbook.
