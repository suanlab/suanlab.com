## Who chooses first?

“For every task there is a good model” differs from “there is one model good for every task.” The first allows the model to depend on the task; the second requires one shared choice. Quantifiers express this dependency and are central to reading optimization guarantees and research claims.

A predicate $R(x,y)$ becomes true or false after inputs are chosen. A quantified sentence also specifies where those inputs range. In the lab both variables belong to the explicitly bounded set $D=\{0,\ldots,N-1\}$.

## Four basic forms

| Formula | Reading |
| --- | --- |
| $\forall x\in D\,\exists y\in D:R(x,y)$ | Each x has at least one suitable y |
| $\exists y\in D\,\forall x\in D:R(x,y)$ | One y works for every x |
| $\forall x\in D\,\forall y\in D:R(x,y)$ | Every pair works |
| $\exists x\in D\,\exists y\in D:R(x,y)$ | At least one pair works |

Repeated universal quantifiers can exchange order, as can repeated existential quantifiers, over the same independent domains. Mixed quantifiers generally cannot. Parentheses and dependencies determine what choices may vary.

## A completely inspectable example

Set $N=3$ and $R(x,y)$ to mean $y=x$. The relation table is

| x\y | 0 | 1 | 2 |
| --- | --- | --- | --- |
| 0 | True | False | False |
| 1 | False | True | False |
| 2 | False | False | True |

Every row has a true cell, so $\forall x\exists y R$ holds: choose $y=x$. No column is entirely true, so $\exists y\forall x R$ fails. A witness to an existential statement must meet the entire scope after it; one successful pair does not meet a universal requirement.

The second formula implies the first whenever a shared witness exists: simply reuse that witness for each input. The equality example shows why the reverse direction fails.

## Negate the whole statement

To negate a universal, supply a counterexample. To negate an existential, rule out all candidates:

$$
\neg\forall x\,P(x)\equiv\exists x\,\neg P(x),\qquad
\neg\exists y\,Q(y)\equiv\forall y\,\neg Q(y).
$$

Applying these successively gives

$$
\neg\bigl(\forall x\exists y\,R(x,y)\bigr)
\equiv\exists x\forall y\,\neg R(x,y).
$$

One failed pair $(x,y)$ is insufficient to refute “every x has some y.” You need one x for which every allowed y fails. The lab lists exactly such failed rows.

## Changing the domain changes the theorem

Choose $R(x,y):y>x$. On the finite set $\{0,1,2\}$, x=2 has no larger allowed y. Thus the universal-existential sentence fails. Over all natural numbers, choose $y=x+1$, so the corresponding sentence is true.

Over the natural numbers there is still no single y larger than every x. Given a proposed y, select x=y. This is a general refutation, not merely failure to find a large enough candidate in a search window.

Domain restrictions can also alter an AI guarantee. A statement about all inputs in a bounded set is not a statement about every possible input. A finite benchmark cannot by itself verify a claim over an unrestricted distribution or all future tasks.

## Empty domains and vacuity

The lab allows $D=\varnothing$ using **set-bounded quantification**. There is no x violating a universal statement, so $\forall x\in D\exists y\in D R$ is true. There is no y witnessing an existential, so $\exists y\in D\forall x\in D R$ is false.

This convention is about quantifying over an explicitly empty set. Standard presentations of first-order structures may require a nonempty underlying domain; do not silently conflate those conventions. No witness is invented by an empty-domain universal statement.

An empty training subgroup can similarly make a universally phrased condition vacuously hold. Report subgroup counts so an apparent guarantee is not mistaken for observed coverage.

## Witnesses as constructive information

For a finite model, an existential can be checked by returning a successful candidate, and a failed universal by returning an offending element. For infinite domains, enumerating candidates is generally not enough to decide every statement. A mathematical construction or impossibility argument supplies information beyond a bounded search.

The function $y(x)=x$ is a constructive family of witnesses for equality. A shared scalar y is a different type of object. This is the difference between selecting a policy dependent on an observation and selecting one fixed action in advance.

## Read the visualization

Rows are x and columns are y. T/F cells encode the predicate directly. Each row’s smallest witness is listed separately in the accessible table. The shared-witness readout lists complete true columns. The empty grid is labeled as an empty domain instead of displaying fabricated successful cells.

Changing the predicate code changes the meaning of R: 0 means equality, 1 strict greater-than, and 2 greater-than-or-equal. These compare y with x. Always restate the predicate before interpreting its truth values.

## Exercises and reference

1. On D={0,1}, which mixed-quantifier equality statement holds? **Answer:** only $\forall x\exists y$.
2. Negate $\exists y\forall x R(x,y)$. **Answer:** $\forall y\exists x\neg R(x,y)$.
3. For finite D and $y\ge x$, find a shared witness. **Answer:** its maximum, when nonempty.
4. Translate “for every tolerance there is a sample size” and explain which chosen quantity may depend on the other.

For predicate logic and quantifiers, see the [MIT Mathematics for Computer Science readings](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/pages/readings/). The finite models here are explicit teaching constructions, not proofs about unrestricted domains.
