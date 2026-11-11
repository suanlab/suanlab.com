## When can function search become finite coefficient search?

The representer theorem applies when loss depends on finitely many function evaluations and regularization grows appropriately with RKHS norm. If a minimizer exists, one can choose it in the observed kernel-section span. A strictly increasing penalty removes unnecessary orthogonal components. Existence, uniqueness and coefficient uniqueness need separate assumptions.

$$
f=f_\parallel+f_\perp,\qquad f_\perp\perp\operatorname{span}\{k_{x_i}\}.
$$

Reproduction gives f⊥(xᵢ)=0, leaving data loss unchanged. Squared norm splits orthogonally, so discarding the complement reduces penalty.

## Construct an invisible direction

Use features (1,x,x²), kernel 1+xz+x²z² and observations (−1,1),(1,1). The unit coefficient vector v=(1,0,−1)/√2 is orthogonal to both observed features:

$$
v\cdot\varphi(x)=\frac{1-x^2}{\sqrt2},\qquad v\cdot\varphi(\pm1)=0.
$$

Adding cv preserves training predictions but changes the central value by c/√2. Equal observations do not identify an entire function.

## Solve the penalized example

The objective is mean squared residual plus λ‖w‖² with λ>0. Symmetry and stationarity give w=(t,0,t), t=1/(2+λ). Adding the complement yields norm squared 2t²+c²:

$$
J(w+cv)-J(w)=\lambda c^2.
$$

At λ=1, t=1/3, training MSE is 1/9 and minimum objective 1/3. Adding c=1 gives objective 4/3 without changing observations. At λ=.1, the same perturbation costs .1. Blue is the minimizer, orange the alternative and dots the data.

## What the theorem does not guarantee

It does not establish accurate extrapolation. Derivative or integral observations require checking their continuous representatives. Singular Gram matrices can also allow different coefficient vectors representing one function: function and coefficient uniqueness differ.

This finite RKHS exposes the same orthogonal argument used in infinite-dimensional kernel regression. Numerical conditioning and statistical generalization remain separate.

## Separate representer reduction, existence and uniqueness

Let L:Rⁿ→[0,∞) be lower semicontinuous and finite somewhere on the attainable evaluation vectors. With λ>0, minimizing L(f(x₁),…,f(x_n))+λ‖f‖² can first be restricted to the finite-dimensional closed section span by orthogonal projection. A minimizing sequence there is bounded because L≥0 and the quadratic penalty is coercive. A convergent subsequence and lower semicontinuity give an attained minimum. If L is convex, the strictly convex squared Hilbert norm makes the minimizing function unique.

These assumptions add an existence proof to the representer statement. For example, the unregularized restricted class of constants t>0 with a zero label has infimum squared loss zero but no minimizer. Closedness and coercivity cannot simply be omitted. A nondecreasing penalty allows a span minimizer when a minimizer exists; a strictly increasing penalty forces every minimizer to have zero complement.

## A unique function may still have several coefficient vectors

Use the constant kernel k=1 with two observations, both labeled one, and sum loss with λ=1/2. Every function is a constant t, so its objective is 2(t−1)²+λt². Differentiation gives t=4/5 and objective 2/5. But f=α₁k(·,x₁)+α₂k(·,x₂) depends only on α₁+α₂.

| Coefficients (α₁,α₂) | Function t | Objective | Solves (K+λI)α=y |
|---|---|---|---|
| (4/5,0) | 4/5 | 2/5 | No |
| (0,4/5) | 4/5 | 2/5 | No |
| (2/5,2/5) | 4/5 | 2/5 | Yes |

For singular K, the coefficient stationarity condition is

$$
K\big((K+\lambda I)\alpha-y\big)=0.
$$

Solving the invertible system (K+λI)α=y selects one valid canonical coefficient vector. It is sufficient, but the first two rows show it is not necessary for every representation of the same minimizing function. The code checks all three exactly. This constant-kernel example is separate from the live three-feature, mean-loss experiment.

**Exercise:** Why does λI not make the function's representation injective? **Explanation:** The function map α↦∑α_i k_i still has the original Gram nullspace. The chosen coefficient-solving convention is a different map.

```python
from fractions import Fraction as F

lam=F(1,2)
t=F(2)/(2+lam)
coefficients=[(t,F(0)),(F(0),t),(t/2,t/2)]
for alpha in coefficients:
    fitted=sum(alpha)
    objective=2*(fitted-1)**2+lam*fitted**2
    assert fitted==F(4,5) and objective==F(2,5)
    # K alpha depends only on the sum; K is the all-ones matrix.
    stationarity=[fitted-1+lam*a for a in alpha]
    assert sum(stationarity)==0  # K(K+lambda I)alpha=K y.
    canonical=all(v==0 for v in stationarity)
    print(tuple(str(x) for x in alpha),str(fitted),str(objective),canonical)
# For the separate unregularized class g_t=t, t>0, with zero labels,
# inf_t g_t(0)^2=0 is approached but not attained: the class is not closed.
```

## Exercises with explanations

1. Double c: added cost? **Four times larger**, because the penalty is quadratic.
2. Does invisibility at training points imply invisibility everywhere? **No:** the central change is a counterexample.
3. Must the complement vanish at λ=0? **No.** Without a penalty multiple minimizers can remain. The controls here keep λ positive.

Further reading: [Schölkopf, Herbrich and Smola, A Generalized Representer Theorem](https://alex.smola.org/papers/2001/SchHerSmo01.pdf).
