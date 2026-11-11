## Perfect observations need not identify a function

Suppose n uniform locations in [−1,1] observe synthetic truth f*=x². Identification depends on the allowed function class and prior. Add a polynomial vanishing at every sample:

$$
p_n(x)=\prod_{i=1}^n(x-x_i),\qquad p_n(x_i)=0.
$$

This is a **constructed ambiguity example**, not a claim that a particular learner returns this alternative. Truth is available only because the experiment is synthetic.

## Normalize population ambiguity

Use population measure dx/2 and qₙ=pₙ/‖pₙ‖₂. Then alternatives fA=f*+Aqₙ have identical training values but

$$
E_{X\sim U[-1,1]}[(f_A(X)-f_*(X))^2]=A^2.
$$

At n=5,A=1, training MSE is zero and population MSE one. Halving amplitude leaves training error zero and reduces population error to 1/4. Compare every table row with the between-sample orange curve.

## Interpret growing sample count carefully

At n=9 the construction uses a new ninth-degree vanishing polynomial. The candidate family grows with n. This is not a learning curve for a fixed quadratic model or evidence against consistency of a specific learner. It shows that an unrestricted class permits new ambiguities.

Expressiveness asks whether a class contains a target. Sample efficiency asks how much data suffices under a class, distribution, noise model, tolerance and confidence level. Parameter count alone cannot equate them. Norm, smoothness, margin or symmetry can constrain between-sample behavior.

## Connect to kernels and generalization

This resembles the representer lesson's invisible directions, now measured by population risk. Norm regularization or implicit bias can select among interpolating functions, but useful generalization needs additional assumptions and evidence.

Interpolation is not inherently bad. It is simply insufficient by itself as a guarantee. Use independent test evidence and a complexity analysis appropriate to the chosen class.

## Add a fixed-class sample guarantee with a bounded loss

The vanishing-polynomial construction grows its alternative class with n. For a different question, fix a finite class H of M predictors before iid sampling and assume each loss lies in [0,1]. Hoeffding for each predictor and a union bound give, with probability at least 1−δ,

$$
\sup_{h\in H}|R(h)-R_n(h)|\le
\sqrt{\frac{\log(2M/\delta)}{2n}}=:\Delta_n.
$$

For an empirical risk minimizer ĥ and population class minimizer h_H, insert empirical risks between R(ĥ) and R(h_H). The ERM difference is nonpositive and the two deviations cost at most 2Δ_n. Thus R(ĥ)−R(h_H)≤2Δ_n. Bounded squared loss would need bounded targets and predictions; the live normalized polynomials do not automatically satisfy the unit-loss assumption.

## Enumerate a fixed finite classification problem

Take four equally likely input symbols and all M=16 binary labelings as a fixed hypothesis class. Truth is (0,0,1,1), and labels are noiseless. ERM chooses the lexicographically first consistent labeling, equivalently assigning zero to unseen symbols. This algorithm and class do not change with n. A positive-labeled symbol is unseen with probability (3/4)ⁿ, so expected classification error is (1/2)(3/4)ⁿ.

| iid observations n | Exact expected classification error | Decimal |
|---|---|---|
| 1 | 3/8 | .375 |
| 2 | 9/32 | .28125 |
| 4 | 81/512 | .158203 |
| 6 | 729/8192 | .088989 |

The code enumerates all 4ⁿ input sequences, not only distinct sample sets, and solves ERM over all sixteen hypotheses. Repeated observations matter for iid probabilities. This gives an actual fixed-class learning curve, unlike constructing a fresh higher-degree alternative for every sample count. It is a separate classification example; its error is not the live regression MSE.

**Exercise:** Can the union bound certify a class chosen after inspecting labels? **Explanation:** Not directly. The fixed-class argument needs a uniform bound covering the selection procedure or another valid data-dependent analysis.

```python
from itertools import product
from fractions import Fraction as F

# A separate fixed classification class: all 16 labelings of four equally likely inputs.
H=list(product([0,1],repeat=4))
truth=(0,0,1,1)
for n in [1,2,4,6]:
    total=F(0)
    for sample in product(range(4),repeat=n):
        # Lexicographic tie-breaking fills unseen labels with zero.
        fitted=min(H,key=lambda h:sum(h[x]!=truth[x] for x in sample))
        assert all(fitted[x]==truth[x] for x in sample)
        total+=F(sum(a!=b for a,b in zip(fitted,truth)),4)
    expected=total/4**n
    assert expected==F(1,2)*F(3,4)**n
    print(n,str(expected),round(float(expected),6))
```

## Exercises with explanations

1. What if A=0? **Both functions and risks coincide at zero error.**
2. Why does risk remain one with more samples? **The alternative family also expands.** This is not a fixed learner's failure curve.
3. Is population risk merely a grid estimate? **No.** Polynomial-square integration normalizes it analytically and independent quadrature verifies it.

Further reading: [MIT 9.520, Statistical Learning Theory](https://ocw.mit.edu/courses/9-520-statistical-learning-theory-and-applications-spring-2003/).
