## 1. Applying two generators creates a new question

A constant Hamiltonian rotates amplitudes with a fixed rule. Suppose we can switch that rule: apply Hamiltonian A for a duration, then Hamiltonian B for another duration. Does reversing the schedule produce the same final preparation? Does replacing the two rules by their weighted average reproduce the experiment? These questions concern operators and physical time, not the order of measurements.

We retain a closed C² pure state, Hermitian matrices and a positive ℏ. The switch is externally prescribed and instantaneous in this idealized model. We do not model a bath or a measurement apparatus. Different times and schedules are compared using separately prepared copies. No projective collapse is inserted between the pulses.

## 2. The earliest action stands on the right

Let UA(s) and UB(s) be the constant-Hamiltonian propagators. Applying A first and B second means multiplying the initial column vector by UA, then multiplying that result by UB. Matrix notation places the first action on the right.

$$
U_A(s)=e^{-iH_As/\hbar},\quad U_B(s)=e^{-iH_Bs/\hbar},\qquad
U_{AB}(T)=U_B(\tau_B)U_A(\tau_A),\quad
U_{BA}(T)=U_A(\tau_A)U_B(\tau_B),\quad T=\tau_A+\tau_B.
$$

Both schedules spend the same total time under each Hamiltonian. They need not produce the same operator. The issue resembles composing two different linear layers: the output of the first layer is the input of the second, and exchanging their positions generally matters. Here every layer is unitary, so a difference cannot be attributed simply to loss of norm.

## 3. Compare prefixes at the same physical time

The endpoint product alone does not describe the trajectory. At a common physical clock time t, one schedule may already have switched while the other has not. We must use the time actually spent under each generator.

$$
U_{AB}(t)=U_B(\max(0,t-\tau_A))U_A(\min(t,\tau_A)),\qquad
U_{BA}(t)=U_A(\max(0,t-\tau_B))U_B(\min(t,\tau_B)),\quad 0\le t\le T.
$$

For τA=2/3 and τB=4/5, T=22/15. At t=4/5 the forward schedule has used A for 2/3 and B for 2/15. The reversed schedule has used B for 4/5 and A for 0. Each pair is nonnegative and sums to 4/5. Comparing equal step indices without their time labels would obscure this difference.

The common display grid includes both exact switch times, even when neither lies on the regular sampling grid. The state is continuous at a switch because U(0)=I. Its derivative can jump when H changes. A right-continuous label identifies the newly active pulse; it does not mean the state jumps.

## 4. A constant weighted average is a different schedule

Define dimensionless integrated actions GA and GB. For positive total duration, the weighted average Hamiltonian and its constant evolution are as follows.

$$
G_A=H_A\tau_A/\hbar,\quad G_B=H_B\tau_B/\hbar,\quad G=G_A+G_B,\qquad
\overline H=\frac{\tau_AH_A+\tau_BH_B}{T},\quad
U_{\mathrm{avg}}(t)=e^{-iGt/T}.
$$

At the endpoint this is exp[−i(GA+GB)], which generally differs from exp(−iGB)exp(−iGA). Averaging the matrix entries is a valid way to construct a Hermitian comparison generator; it is not a proof that exponentiation distributes over addition. For perpendicular axes X and Z, the average points along a diagonal axis. One must exponentiate that full generator rather than separately add two independently rotated states.

In two dimensions, any Hermitian integrated action M can be separated into its scalar center c and traceless part D. Its exponential has a controlled closed form.

$$
c=\tfrac12\operatorname{Tr}M,\quad D=M-cI,\quad r^2=\tfrac12\operatorname{Tr}(D^2),\quad D^2=r^2I,\qquad
e^{-iM}=e^{-ic}\left(\cos r\,I-i\frac{\sin r}{r}D\right).
$$

When r=0, D=0 and the bracket is I. The global scalar phase still remains. Cancelling the traceless actions does not necessarily make the full average generator zero.

## 5. Work through X followed by Z

Take HA=X, HB=Z, durations 1 and 1, ℏ=1 and initial |0⟩. Put c=cos1 and s=sin1. Direct multiplication gives two endpoint vectors.

$$
\psi_{AB}=\begin{pmatrix}ce^{-i}\\-is e^{i}\end{pmatrix},\qquad
\psi_{BA}=\begin{pmatrix}ce^{-i}\\-is e^{-i}\end{pmatrix},\qquad
p_Z(+1)=c^2,\quad p_Z(-1)=s^2.
$$

Their Z probabilities are identical. Their second components have different relative phases, so another measurement can detect the order. For +X=(1,1)/√2, calculate the squared modulus of the sum of components divided by 2. The results are (1+sin²2)/2 for A→B and 1/2 for B→A. The first is approximately 0.913411; the second is exactly 0.5. Their state fidelity is 1−sin²2 sin²1, approximately 0.414549.

| Endpoint diagnostic | A→B | B→A |
| --- | --- | --- |
| Norm squared | 1 | 1 |
| Z outcome +1 | cos²1 | cos²1 |
| +X probability | (1+sin²2)/2 | 1/2 |
| Intermediate collapse | None | None |

The equal Z histogram does not certify equal output rays. Neither does a preparation that happens to lie in a special subspace certify that the two operators commute. Operator equality is stronger than equality of one state, and equality of one state is stronger than equality of one selected histogram.

## 6. Commuting generators align endpoints, not all prefixes

If [GA,GB]=0, exponentiation of their sum equals their ordered product. This is a sufficient condition for endpoint equality of the two schedules and the weighted-average schedule. It is not a claim that noncommuting generators can never yield coincident endpoint exponentials at special durations.

Consider HA=Z, HB=2Z, τA=τB=1 and initial +X. The Hamiltonians commute. At t=1 the forward path has accumulated Z, the reverse path 2Z and the average path 3Z/2. Their relative phases differ. At t=2 they have all accumulated 3Z and coincide.

| Clock time | A→B integrated action | B→A integrated action | Average integrated action |
| --- | --- | --- | --- |
| 0 | 0 | 0 | 0 |
| 1 | Z | 2Z | 3Z/2 |
| 2 | 3Z | 3Z | 3Z |

Identical Hamiltonians are a stronger simple case: all three paths coincide at every common time, even with unequal durations. If one pulse has zero duration, the remaining pulse also aligns all three paths. If both durations vanish, the identity propagator is defined, but the average Hamiltonian would require division by T=0. It is unavailable, represented by null rather than an arbitrarily chosen zero matrix.

## 7. The commutator predicts the first small-action difference

Scale both integrated actions by λ and expand each exponential. Constant and linear terms cancel in the order difference. The first surviving term is quadratic and has a sign determined by the order of the factors.

$$
e^{-i\lambda G_B}e^{-i\lambda G_A}-e^{-i\lambda G_A}e^{-i\lambda G_B}
=\lambda^2[G_A,G_B]+O(\lambda^3),
$$

$$
e^{-i\lambda G_B}e^{-i\lambda G_A}-e^{-i\lambda(G_A+G_B)}
=\tfrac12\lambda^2[G_A,G_B]+O(\lambda^3).
$$

For GA=δX and GB=δZ, the leading order difference is δ²[X,Z]=−2iδ²Y. Halving both small durations approximately divides the squared Frobenius order difference by 16. The factor is approximate because higher terms remain. A large commutator norm is useful structural information, but this truncated series is not an exact finite-pulse identity or a certified global error bound.

The experiment keeps the leading matrix, the full difference and the remainder separate. A visualization should not silently replace the actual propagation by the leading term just because the term has a simpler shape.

## 8. Reordering positive pulses is not undoing evolution

To invert a product, reverse the factor order and invert each factor. For this experiment the inverse is a different operator from the positive-duration reordered schedule.

$$
U_{AB}(T)^{-1}=U_A(-\tau_A)U_B(-\tau_B),\qquad
U_{AB}(T)^{-1}U_{AB}(T)=I.
$$

Changing the schedule to B→A still advances positive physical time under both Hamiltonians. It does not change the signs in their exponentials. The negative-time factors above express a mathematical inverse; implementing that inverse physically would require appropriate control of the generator, not a literal backward-moving laboratory clock.

A norm-preserving output is not sufficient to prove that a proposed undo operation worked. Every schedule here preserves norm. The product with the original propagator must be compared with I, or its action tested on a spanning set.

## 9. Conservation laws and comparisons need labels

Within an interval of constant HA, its expectation is conserved. After switching to HB, the same expectation need not remain constant. The externally changed Hamiltonian can change the energy assigned at the switch, so closed unitary evolution with prescribed driving does not imply conservation of one global energy value across the entire schedule. The state itself remains continuous.

The average path conserves the expectation of its own constant generator G. That is a property of the comparison path, not a reason to enforce the same conservation on the pulse paths. We report expectations of HA and HB separately.

State fidelity ignores a common phase; squared representative difference does not. A process fidelity based on |Tr(U†V)/2|² compares the two unitary actions up to a common phase, while the Frobenius matrix difference includes that phase. Matching one state's fidelity cannot certify the process comparison. Scalar energy shifts can change representatives while leaving rays unchanged.

## 10. How to inspect the computation

Begin with the X/Z example, predict the two final Z bars and then compare +X and +Y probabilities. Next select commuting Z and 2Z and inspect the middle of the schedule before its endpoint. Finally reduce both durations together to study the local commutator scaling. Keep physical time, pulse duration and display sample index distinct.

Energy is measured in E0, time in T0 and ℏ in E0T0; every integrated action is dimensionless. The numerical family uses twenty-seven owned integer fields. Hamiltonians, durations, actions, commutators and exact grid points are complex rationals. General two-dimensional Hermitian exponentials use directed square-root and trigonometric intervals, with 768-bit work arithmetic and 512-bit public observations. No evolving vector is renormalized to hide an error. A zero-containing unresolved observation can retain null; a plotted rounded zero is not proof of structural equality.

Independent SymPy matrices and mpmath general matrix exponentials cover complex inputs, nonorthogonal axes, cancellation, large phases and very small nonzero actions. A separate ODE comparison integrates each constant segment before continuing across the switch. Its tolerance is a numerical comparison, not an enclosure theorem for the ODE solver. Boundary continuity, nonnegative elapsed times and inverse propagation are checked separately.

## Research extension: repeated splitting at fixed total exposure

### Keep the physical budget fixed before refining a schedule

The earlier local commutator expansion did not give a finite-step error certificate. Now divide each of the original integrated actions G_A and G_B into n pieces. Keep their total exposures fixed. Set:

$$
X=-iG_A,\quad Y=-iG_B,\quad h=1/n,\quad V=e^{X+Y},\quad
 L_n=(e^{hY}e^{hX})^n,\quad
 S_n=(e^{hX/2}e^{hY}e^{hX/2})^n.
$$

These matrices X,Y are anti-Hermitian. L_n repeats a small A-then-B action; S_n uses a symmetric A-half/B/A-half cycle. V is the original weighted-average endpoint. It is not n full copies of the original two pulses.

For the example H_A=X_Pauli, H_B=Z, ℏ=1 and original durations1 and1, the total physical duration is2 for every n. One Lie cycle spends1/n under each Hamiltonian. One symmetric cycle spends1/(2n),1/n,1/(2n). The average Hamiltonian is (X_Pauli+Z)/2 over physical time2. Comparing to that sum for time2 would double the intended action. The X in the displayed derivation denotes −iG_A, not the Hermitian Pauli matrix.

### Derive a finite Lie bound using unitarity

Connect the one-cycle ordered product to the one-cycle exponential by an interpolating matrix f(s). Direct differentiation gives:

$$
f(s)=e^{(h-s)(X+Y)}e^{sY}e^{sX},\qquad
 f'(s)=e^{(h-s)(X+Y)}[e^{sY},X]e^{sX},\quad
 [e^{sY},X]=\int_0^s e^{(s-u)Y}[Y,X]e^{uY}du.
$$

Every exponential factor here is unitary. Taking operator norms inside the integral therefore bounds the commutator by s times its generator commutator. Integrating f′ from0 to h gives:

$$
\|e^{hY}e^{hX}-e^{h(X+Y)}\|_{\rm op}
 \le\int_0^h s\|[Y,X]\|_{\rm op}ds
 =\tfrac12h^2\|[G_A,G_B]\|_{\rm op}.
$$

This holds for every finite anti-Hermitian pair and h≥0; it is not only a small-step O statement. To accumulate cycles, use the telescoping identity:

$$
C=\|[G_A,G_B]\|_{\rm op},\quad
 L_n-V=\sum_{k=0}^{n-1}L_h^{n-1-k}(L_h-V_h)V_h^k,\quad
 L_h=e^{hY}e^{hX},\quad V_h=e^{h(X+Y)},\quad
 \|L_n-V\|_{\rm op}\le\min(2,C/(2n)).
$$

All surrounding powers have norm1. There is no exponential growth factor in this Lie estimate. The cap2 uses the distance between any two unitaries. A vanishing commutator makes the products exact, while a positive upper bound alone does not prove the actual error is positive.

### Symmetry removes the quadratic defect

Multiply the three exponential series through degree2:

$$
e^{hX/2}e^{hY}e^{hX/2}
 =I+h(X+Y)+\tfrac12h^2(X+Y)^2+O(h^3).
$$

For instance the XY and YX terms both receive coefficient1/2, unlike the unsymmetrized product. A conservative finite remainder can be bounded without guessing a leading coefficient. Let L be the sum of the two action norms. For each series, dominate every degree-k product coefficient in norm by L^k/k!; this follows from the multinomial expansion using the three factor norms. The target exponential has the same degree bound. Their degree0,1,2 terms cancel, leaving two tails:

$$
L=\|G_A\|_{\rm op}+\|G_B\|_{\rm op},\quad
 R_3(z)=\sum_{k=3}^\infty\frac{z^k}{k!}\le\frac{z^3e^z}{6}\quad(z\ge0),\quad
 \|e^{hX/2}e^{hY}e^{hX/2}-V_h\|_{\rm op}\le2R_3(hL).
$$

The scalar tail inequality follows from (j+3)!≥6j!. Apply the same unitary telescoping argument to n symmetric cycles:

$$
\|S_n-V\|_{\rm op}\le\min\!\left(2,\frac{L^3e^{L/n}}{3n^2}\right).
$$

This bound is intentionally conservative, particularly at small n. For fixed actions it proves an O(n⁻²) bound, whereas the Lie certificate scales as O(n⁻¹). It is a finite-matrix norm argument and gives no analogous guarantee for arbitrary unbounded differential operators.

### Relate the certificate to a detector

For a normalized input and any projective yes/no detector R, expand the difference of the two quadratic forms, add and subtract the cross term, then apply Cauchy–Schwarz:

$$
\left|\|RU\psi\|^2-\|RV\psi\|^2\right|
 \le\min(1,2\|U-V\|_{\rm op}),\qquad
 R=R^\dagger=R^2,\quad \|\psi\|=1.
$$

This is a sufficient probability-error bound, often loose. The matrix distance includes global phase while probabilities do not. Identical norms of propagated vectors alone cannot certify an accurate final state or probability.

The following standalone code uses the Pauli example at total physical duration2. Each number is a rounded floating-point illustration. The operator-distance shortcut in the code applies to these determinant-one 2×2 unitaries; the independent check instead uses a general matrix exponential and singular values. The current two-pulse reader has no new repetition-count control.

```python
from math import cos, sin, sqrt, exp
I = ((1+0j, 0j), (0j, 1+0j))
X = ((0j, 1+0j), (1+0j, 0j))
Z = ((1+0j, 0j), (0j, -1+0j))
def product(a, b):
    return tuple(tuple(sum(a[i][k]*b[k][j] for k in range(2))
                       for j in range(2)) for i in range(2))
def rotation(axis, angle):
    return tuple(tuple(cos(angle)*I[i][j]-1j*sin(angle)*axis[i][j]
                       for j in range(2)) for i in range(2))
def power(a, n):
    result = I
    for _ in range(n):
        result = product(a, result)
    return result
axis = tuple(tuple((X[i][j]+Z[i][j])/sqrt(2) for j in range(2))
             for i in range(2))
target = rotation(axis, sqrt(2))
def error(u):  # operator distance for these determinant-one 2x2 unitaries
    overlap = sum(u[i][j].conjugate()*target[i][j]
                  for i in range(2) for j in range(2))
    return sqrt(max(0, 2-overlap.real))
for n in [1, 2, 4, 8]:
    a, b = rotation(X, 1/n), rotation(Z, 1/n)
    half = rotation(X, 1/(2*n))
    lie = power(product(b, a), n)
    symmetric = power(product(half, product(b, half)), n)
    print(n, f"{error(lie):.6f}", f"{1/n:.6f}",
          f"{error(symmetric):.6f}", f"{min(2, 8*exp(2/n)/(3*n*n)):.6f}")
```

| n | Lie error | Lie bound | symmetric error | symmetric bound |
|---|---|---|---|---|
| 1 | 0.799214 | 1.000000 | 0.313666 | 2.000000 |
| 2 | 0.362410 | 0.500000 | 0.071130 | 1.812188 |
| 4 | 0.176261 | 0.250000 | 0.017273 | 0.274787 |
| 8 | 0.087513 | 0.125000 | 0.004286 | 0.053501 |

![Lie and symmetric endpoint errors decrease as fixed exposures are subdivided, compared with their finite operator-norm upper bounds on logarithmic axes.](/assets/learning/quantum-unitary-research-fixed-exposure-splitting.en.svg)

The symmetric certificate can be much larger than the measured error. It remains an upper bound, not a fitted trend line. The horizontal variable counts subdivisions, not extra copies of the physical experiment.

#### Research check 1: double n without doubling time

For n=4 the Lie schedule uses eight segments of duration1/4 and total time2. Repeating four full unit-duration A/B pairs would instead take time8 and would answer another question. The symmetric cycle also totals2 after all four repetitions.

#### Research check 2: commuting actions and a loose bound

If G_B=2G_A, both schedules equal V for every n. The Lie commutator certificate is exactly0. The conservative symmetric bound need not vanish because it discarded cancellation in the tails. A nonzero bound is not evidence of a nonzero defect.

#### Research check 3: translate an operator error

If a certified operator error is δ=0.003, the displayed argument gives probability error at most0.006 for every normalized input and projective detector. An actual chosen detector can be less sensitive. This mathematical bound is separate from measurement shot noise and from uncertainty in the Hamiltonian itself.

## 11. Exercises with explanations

**1. Which matrix product means A first and B second?**

**Explanation.** UB UA. Applied to a column vector, the rightmost factor acts first. Naming the chronological schedule AB does not change matrix multiplication.

**2. What are the forward elapsed times at t=4/5 when τA=2/3 and τB=4/5?**

**Explanation.** A has run for 2/3 and B for 2/15. Their sum is 4/5. In the reversed schedule B has run for 4/5 and A for 0. Equal physical time does not imply equal pulse exposure.

**3. Does the equal final Z histogram in the X/Z example prove that order did not matter?**

**Explanation.** No. The +X probabilities are approximately 0.913411 and 0.5, and the rays differ. A single measurement can miss a changed relative phase.

**4. If HA and HB commute, must the prefix trajectories agree?**

**Explanation.** No. Z and 2Z at t=1 produce different accumulated actions under the two schedules. Commutation allows the same complete pulse exposures to be combined at the endpoint; it does not equalize unequal partial exposures.

**5. If GA+GB=0, does the pulse path stay at its initial ray?**

**Explanation.** Not generally. With equal durations and HB=−HA, the average path is stationary, while the pulse path can move away and return. The endpoint identity does not erase the intermediate evolution.

**6. Can a half-duration experiment establish a universal finite-pulse error bound?**

**Explanation.** No. Approximate 1/16 scaling of a squared order difference supports the local quadratic leading term. The remainder and its parameter range still require analysis. A few observed ratios cannot replace a bound valid across the full input domain.

## 12. Connections and remaining scope

This is a concrete introduction to time ordering, operator splitting, controlled dynamics and order-sensitive compositions in AI. It explains why exponentiating an average generator may approximate a composition when steps are small, and why a useful approximation requires a scale and an error analysis. Repeated splitting schemes and more general time-ordered expansions need separate derivations.

General continuous driving, open systems, mixed-state dynamics, entanglement and differential-operator domains remain later topics. This model uses no intermediate measurement or collapse. Open this textbook through **Book → Interactive Book**.

The background convention of unitary closed-system dynamics follows [Preskill §2.1, Axiom 4](https://www.preskill.caltech.edu/ph219/chap2_15.pdf). Trigonometric series are given in [NIST DLMF §4.19](https://dlmf.nist.gov/4.19). The pulse examples, prefix comparison, commutator signs and numerical contract are derived here and checked against independent computations.
