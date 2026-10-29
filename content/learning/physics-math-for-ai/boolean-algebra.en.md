## Boolean values are mathematical objects

Use zero and one for false and true. AND, written ∧, is one only when both inputs are one. OR, written ∨, is one when at least one input is one. NOT, written ¬, exchanges zero and one. These operations should not be silently confused with ordinary real addition.

Two-input functions have four input rows. Equality of functions means agreement on all four, not just a sampled subset.

## Encode a truth table

The row order is (0,0),(0,1),(1,0),(1,1), indexed by 2x+y. The corresponding mask bit supplies the output. Mask 6=0110₂ is XOR. A binary numeral is normally written with its highest bit first, while the table starts at row zero; track the convention.

There are 2⁴=16 two-input Boolean functions. For n inputs there are 2ⁿ rows and therefore 2^(2ⁿ) possible functions. The function space grows rapidly even when each input has only two values.

## Construct a formula from minterms

Each row has a minterm true only on that row: ¬x∧¬y, ¬x∧y, x∧¬y, or x∧y. OR together the minterms of the true rows to reconstruct any function in canonical DNF. For example,

$$
x\oplus y=(\neg x\land y)\lor(x\land\neg y).
$$

An empty OR is zero; all true rows give the constant-one function. This construction proves representability, not minimal expression size. Logic minimization is a separate problem.

## Prove De Morgan’s laws

The formula ¬(x∧y) is false only when both inputs are true. The formula ¬x∨¬y has exactly the same behavior. Check all four rows to establish

$$
\neg(x\land y)=\neg x\lor\neg y,\qquad
\neg(x\lor y)=\neg x\land\neg y.
$$

Negating “authorized AND valid” yields “not authorized OR not valid.” The inner connective changes as well as the literals.

## Algebraic normal form uses GF(2)

Over GF(2), addition is XOR, so 1+1=0, while multiplication agrees with AND. Every two-input function has the form

$$
F(x,y)=a_0\oplus a_xx\oplus a_yy\oplus a_{xy}xy.
$$

Substitution of all four rows gives a₀=F(0,0), aₓ=F(1,0)⊕F(0,0), aᵧ=F(0,1)⊕F(0,0), and aₓᵧ=F(0,0)⊕F(0,1)⊕F(1,0)⊕F(1,1). XOR has coefficient vector (0,1,1,0); NAND has (1,0,0,1).

Do not read ANF addition as OR or real addition. OR becomes x⊕y⊕xy over GF(2). Testing x=y=1 exposes the difference.

## AI connections and limits

Filters, rule-based labels and exact logical constraints can be Boolean functions. A sigmoid score is a continuous quantity and does not automatically obey Boolean identities. Specify thresholding and the resulting decision rule.

Agreement between a learned model and a formula on training samples does not imply global equivalence. This tiny lab can exhaust all four inputs. Larger verification tasks may need tools such as SAT or BDD methods.

## Exercises and answers

1. XOR and NAND differ on **row 00**.
2. OR has ANF **x⊕y⊕xy**.
3. Mask zero has **no active minterms and constant output zero**.
4. Over the reals, x+y−xy restricted to binary inputs equals **OR**. Explain why its arithmetic convention differs from the GF(2) expression.

Further reading: [MIT Mathematics for Computer Science](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/). The coefficient derivation and small tables above are independently constructed.
