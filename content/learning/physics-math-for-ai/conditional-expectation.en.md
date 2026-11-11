## Condition on information rather than one event

Conditional expectation given G is a random variable expressible using the available information, not merely one number. Assume X is integrable and G is a sub-sigma-algebra. The defining requirements are G-measurability and preservation of integrals over every G-event. A prediction distinguishing points within one information atom would use information not present in G.

$$
Y=E[X\mid\mathcal G]:\quad Y\text{ is }\mathcal G\text{-measurable},\quad\int_A Y\,dP=\int_A X\,dP\ \forall A\in\mathcal G.
$$

## Construct it on finite atoms

With X=(0,2,1,3), weights (0.1,0.2,0.3,0.4) and atoms {0,1}, {2,3}, choose a constant on each atom. The weighted averages are 4/3 and 15/7. They preserve the integral on each atom, hence on every union of atoms and therefore every G-event.

$$
Y(\omega)=\frac{\sum_{i\in A}p_iX_i}{P(A)}\quad(\omega\in A,\ P(A)>0).
$$

## The tower property and information resolution

Taking the entire space as an event gives the common mean 1.9. Full information recovers X; trivial information gives the constant overall mean. For nested information H contained in G, averaging first under G and then under H agrees with averaging directly under H. General existence connects to the Radon–Nikodym theorem, while uniqueness is only almost sure.

$$
\mathcal H\subseteq\mathcal G\implies E[E[X\mid\mathcal G]\mid\mathcal H]=E[X\mid\mathcal H].
$$

## Null atoms and least-squares interpretation

A null atom has no defined averaging ratio, but any constant chosen there leaves integral preservation unchanged. The lab explicitly chooses zero as a version. This does not define elementary conditioning on a null event to equal zero. For square-integrable X, conditional expectation minimizes squared error among G-measurable predictors. Weighted residuals integrate to zero on every atom, exposing the underlying orthogonality.

$$
E[(X-Z)^2]=E[(X-Y)^2]+E[(Y-Z)^2]\quad(Y=E[X\mid\mathcal G],\ Z\in L^2(\mathcal G)).
$$

## Construct conditional expectation using Radon–Nikodym

Let X be integrable on a probability space and G a sub-sigma-algebra. On G define two finite measures ν±(A)=∫A X± dP. Each is absolutely continuous with respect to P restricted to G. The Radon–Nikodym theorem supplies nonnegative G-measurable densities Y±. Their integrals are E[X±], hence finite, and Y=Y+−Y− is integrable and has the required event integrals. This constructs conditional expectation from the RN theorem; it does not prove the RN theorem itself.

For uniqueness, suppose Y and Y′ satisfy the definition. Each G-event A_k={Y−Y′>1/k} has zero integral of Y−Y′, but that integral is at least P(A_k)/k. Thus P(A_k)=0. Taking their countable union gives P(Y>Y′)=0; reversing the variables gives equality almost surely. Values on null information atoms remain a choice of version.

## Prove the tower using its information hypothesis

If H is contained in G, every H-event is also a G-event. Conditional expectation of Y=E[X|G] onto H preserves ∫A Y=∫A X for each such event and is H-measurable. Uniqueness then identifies it with E[X|H]. The inclusion hypothesis is exactly what makes the middle integral replacement valid; two arbitrary partitions need not satisfy it.

## Justify the L2 projection without assuming the conclusion

Assume X is square integrable. First show that Y=E[X|G] also belongs to L2. Let Y_K be Y clipped to [−K,K]. Integral preservation extends from indicators to bounded G-measurable functions by simple approximation. Since YY_K≥Y_K², Cauchy–Schwarz gives

$$
\|Y_K\|_2^2\le E[YY_K]=E[XY_K]\le\|X\|_2\|Y_K\|_2.
$$

Therefore ||Y_K||2≤||X||2. Monotone convergence of Y_K² to Y² proves the L2 claim. The residual X−Y has zero inner product with every bounded G-measurable function. Approximate an arbitrary L2 G-measurable function by its clipped versions and use Cauchy–Schwarz again to extend the orthogonality. Expanding X−Z=(X−Y)+(Y−Z) now proves the squared-error identity above. This argument explains both admissibility of the predictor and why the cross term vanishes.

## Two nonnested information sets can give a different answer

For four equiprobable outcomes, take X=(0,0,0,4). Let G have atoms {0,1},{2,3}, and H have atoms {0,1,2},{3}. Neither sigma-algebra contains the other. Write P_G and P_H for their conditional-expectation operators. Since X is already H-measurable, P_H X=X. But the following calculation gives P_H P_G X≠P_H X:

| $\omega$ | X | $P_GX$ | $P_HP_GX$ |
|---|---|---|---|
| 0 | 0 | 0 | $2/3$ |
| 1 | 0 | 0 | $2/3$ |
| 2 | 0 | 2 | $2/3$ |
| 3 | 4 | 2 | 2 |

Here P_G P_H X=P_G X also differs from P_H P_G X, so the operators do not commute. Their intersection information is trivial, whose prediction would be the constant mean one; the displayed double projection is not that constant either. Sequential averaging over unrelated information is not a general substitute for conditioning on an intersection.

For a valid G-measurable comparison Z=(3,3,−1,−1), the total squared error is 11: residual error 2 plus predictor-gap error 9. If an information set is genuinely enlarged, all earlier admissible predictors remain available, so the best squared error cannot increase. The task's available information must therefore be specified before comparing prediction risks.

## Compute event identities and the counterexample exactly

The standard-library code enumerates all four G-events, checks the risk identity and computes the nonnested projections. It also reproduces the current lab's weighted reference values and its chosen zero version on a null atom. The nonnested partition H is a separate executable example, not an additional option in the existing controls.

```python
from fractions import Fraction as Q

def expect(x, p):
    return sum(a*b for a, b in zip(x, p))

def condition(x, p, atoms):
    out = [Q(0)] * len(x)
    for atom in atoms:
        mass = sum(p[i] for i in atom)
        value = sum(p[i]*x[i] for i in atom)/mass if mass else Q(0)
        for i in atom:
            out[i] = value
    return tuple(out)

def squared_error(x, y, p):
    return expect(tuple((a-b)**2 for a, b in zip(x, y)), p)

G, H = [[0, 1], [2, 3]], [[0, 1, 2], [3]]
p, x = (Q(1, 4),)*4, (Q(0), Q(0), Q(0), Q(4))
y = condition(x, p, G)
hx, hy = condition(x, p, H), condition(y, p, H)
assert y == (0, 0, 2, 2) and hx == x
assert hy == (Q(2, 3), Q(2, 3), Q(2, 3), Q(2))
assert hy != hx and condition(hx, p, G) != hy
for mask in range(4):
    event = [i for j, atom in enumerate(G) if (mask >> j) & 1 for i in atom]
    assert sum(p[i]*(x[i]-y[i]) for i in event) == 0
z = (Q(3), Q(3), Q(-1), Q(-1))
assert squared_error(x, z, p) == squared_error(x, y, p)+squared_error(y, z, p)
assert squared_error(x, z, p) == 11 and squared_error(x, y, p) == 2
assert condition(y, p, [[0, 1, 2, 3]]) == (Q(1),)*4
reference_p = tuple(Q(i, 10) for i in [1, 2, 3, 4])
reference_x = tuple(Q(i) for i in [0, 2, 1, 3])
assert condition(reference_x, reference_p, G) == (Q(4, 3), Q(4, 3), Q(15, 7), Q(15, 7))
assert condition(reference_x, (Q(1, 2), Q(1, 2), Q(0), Q(0)), G) == (1, 1, 0, 0)
for i in range(4):
    print("point, X, P_G X, P_H P_G X:", i, x[i], y[i], hy[i])
print("squared errors: total, residual, predictor gap:", 11, 2, 9)
```

## Further questions and research use

1. Why is the optimal predictor unique only almost surely? **The norm and event integrals cannot distinguish changes on null sets.**
2. Can the tower identity be used for arbitrary two information sets? **No: its inclusion condition is essential, as the table shows.**
3. Does a feature unavailable at prediction time belong in G? **No. Using it changes the information problem and can introduce leakage.**

For existence, tower and squared-error prediction, see [Durrett, §4.1, Theorems 4.1.13 and 4.1.15; RN Theorem A.4.8](https://sites.math.duke.edu/~rtd/PTE/PTE5_011119.pdf). The four-outcome comparison is calculated directly. Compare [Gaussian prediction](../multivariate-gaussians/) as a particular conditional-mean model.

## Exercises with explanations

1. Is the overall mean preserved? **Yes: Ω belongs to G.**
2. Is the value on a null atom unique? **No: versions may differ on null sets.**
3. Must more information increase optimal squared error? **No: previous predictors remain admissible.**

Further reading: [Durrett, Probability: Theory and Examples](https://sites.math.duke.edu/~rtd/PTE/pte.html); [Tao, An Introduction to Measure Theory](https://terrytao.wordpress.com/wp-content/uploads/2012/12/gsm-126-tao5-measure-book.pdf).
