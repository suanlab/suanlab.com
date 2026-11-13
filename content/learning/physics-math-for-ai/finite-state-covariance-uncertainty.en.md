## 1. Ask which uncertainty is being measured

Imagine a source that prepares the same quantum state on every trial. Measure A on one collection of copies and B on another. Each collection produces a histogram. The widths of those histograms are preparation uncertainties. They can remain positive even when both apparatuses are ideal.

This experiment differs from measuring A and then B on each copy: the first apparatus can change the state presented to the second. It also differs from estimating a mean from a finite sample. A standard error of a sample mean can decrease with the number of copies while the intrinsic outcome variance of each copy stays fixed. Keeping these three questions separate prevents an uncertainty inequality from becoming an unsupported claim about every device or estimator.

The conceptual picture for this chapter is **state → centered observables → fluctuation vectors → positive Gram matrix → uncertainty bound**. We will derive each arrow, then use examples where a familiar shorthand would give the wrong interpretation.

## 2. Center the observables before comparing their fluctuations

Let ρ be positive with trace one and let A and B be Hermitian. Define their means and centered operators. Variance is a second moment of a centered operator, not a variance of matrix entries.

$$
\mu_A=\operatorname{Tr}(\rho A),\quad \Delta A=A-\mu_A I,\quad
v_A=\operatorname{Tr}(\rho(\Delta A)^2)=\operatorname{Tr}(\rho A^2)-\mu_A^2,\qquad
c=\operatorname{Tr}(\rho\Delta A\Delta B)=s+ik.
$$

The real part s is the symmetric covariance. The imaginary part k equals Tr(ρ[A,B])/(2i); it is real because the commutator of two Hermitian operators is anti-Hermitian. Our sign convention places c in the upper-right entry of the Gram matrix. Reversing the order A,B conjugates c and reverses k, while every squared lower bound stays unchanged.

Adding a scalar multiple of the identity to A changes its mean but leaves its centered operator unchanged. Multiplying A by a real α multiplies its variance by α² and both s and k by α. If A and B have physical units, vA has units A², c has units AB and the variance product has units A²B². The finite calculator uses dimensionless numerical observables; physical units must be restored consistently in an application.

## 3. Turn mixed-state fluctuations into ordinary vectors

For a mixed state, work in the Hilbert space of matrices with inner product Tr(X†Y). The two fluctuation vectors are ΔA√ρ and ΔB√ρ. Their Gram matrix is positive because a squared norm cannot be negative.

$$
X=\Delta A\sqrt\rho,\quad Y=\Delta B\sqrt\rho,\quad
\langle X,Y\rangle=\operatorname{Tr}(X^\dagger Y)=c,\qquad
G=\begin{pmatrix}v_A&c\\c^*&v_B\end{pmatrix}\succeq0,
\quad \det G=v_Av_B-s^2-k^2\ge0.
$$

Cyclicity of the trace moves the two square roots together, giving c. This proves the statement for every finite density operator, including states of deficient rank. No inverse of ρ is needed.

An ensemble gives a second picture. Stack the weighted vectors √pᵢΔAψᵢ over preparation labels, and do the same for B. Their inner products give the same G. These columns are a convenient factorization; their appearance depends on the chosen ensemble, while G depends only on ρ,A,B. They are not amplitudes of simultaneously observed A and B outcomes.

## 4. Keep the covariance term when it matters

The determinant condition gives the Schrödinger bound. Dropping the nonnegative s² term gives the weaker Robertson bound. Standard-deviation versions follow by taking nonnegative square roots.

$$
v_Av_B\ge s^2+k^2\ge k^2,\qquad
\sigma_A\sigma_B\ge\sqrt{s^2+k^2}\ge\frac12|\operatorname{Tr}(\rho[A,B])|,
\qquad \sigma_A=\sqrt{v_A}.
$$

A zero commutator expectation makes the Robertson lower bound zero. It does not imply that A and B commute, that their variances vanish, or that the stronger bound is zero. Conversely, commuting observables can have positive covariance due to the chosen state. The inequality supplies a lower bound for a specified state, rather than one universal positive number for every pair of noncommuting matrices.

The two useful slack quantities are det G and vAvB−k². Their difference is s². A small numerical slack should not automatically be called equality: an exact zero and a small positive number answer different questions about saturation.

## 5. Work a correlated pure-qubit example by hand

Take A=X, B=Y and the pure Bloch vector r=(3/5,4/5,0). The Pauli identity XY=iZ fixes the commutator convention. Since X²=Y²=I, the variances follow immediately from their means.

$$
\mu_X=\frac35,\quad\mu_Y=\frac45,\quad
v_X=\frac{16}{25},\quad v_Y=\frac9{25},\quad
c=-\frac{12}{25},\quad k=0,\quad
v_Xv_Y=s^2=\frac{144}{625}.
$$

The stronger inequality is saturated, but the weaker lower bound is zero and is not saturated. The standard-deviation product is 12/25. Thus a vanishing commutator expectation can hide a substantial constraint carried by covariance.

| State, observables | Variances vA,vB | Complex covariance c | Strong slack det G |
| --- | --- | --- | --- |
| Pure r=(3/5,4/5,0), X,Y | 16/25,9/25 | −12/25 | 0 |
| I/2, X,Y | 1,1 | 0 | 1 |
| Pure e₀ in three dimensions, transitions 0↔1 and 0↔2 | 1,1 | 0 | 1 |

The last row has A=e₀e₁†+e₁e₀† and B=e₀e₂†+e₂e₀†. Its fluctuations are orthogonal unit vectors. A pure state does not generally force saturation; the pure-qubit geometry is special.

## 6. Saturation is linear dependence of fluctuation vectors

Cauchy–Schwarz is an equality exactly when the two fluctuation vectors are linearly dependent, including the case where one is zero. When vA>0, project Y onto X. Completing the square gives both the optimal coefficient and the remaining squared norm.

$$
\min_{\lambda\in\mathbb C}\|Y-\lambda X\|^2
=v_B-\frac{|c|^2}{v_A}=\frac{\det G}{v_A},\qquad
\lambda_* =\frac{c}{v_A},\qquad
\langle X,Y-\lambda_*X\rangle=0.
$$

This is the same geometry used in complex least squares, with a state-dependent inner product. Equality means the minimum residual is zero. It is not a claim that two operators are identical on the whole Hilbert space; they need only agree in the fluctuation directions occupied by the state.

If vA=0, then X=0, positivity forces c=0 and every λ gives the same residual vB. The slope is nonunique, so the calculator returns null for it while retaining a well-defined minimum residual. The equation det G/vA must not be evaluated in that branch. For a Z eigenstate with A=Z,B=X, det G=0 but the minimum residual is one, illustrating why the branch matters.

## 7. A covariance ellipse describes real linear combinations

For real u,v, C=uA+vB is Hermitian. Its variance uses the symmetric covariance. The imaginary cross terms cancel.

$$
\operatorname{Var}_\rho(uA+vB)
=u^2v_A+2uvs+v^2v_B
=\begin{pmatrix}u&v\end{pmatrix}
V\begin{pmatrix}u\\v\end{pmatrix},\qquad
V=\begin{pmatrix}v_A&s\\s&v_B\end{pmatrix}\succeq0.
$$

Visualize a unit vector (u,v) rotating around a circle. Its quadratic value is the variance of the corresponding observable; the principal axes of V give directions of greatest and least variance. A zero eigenvalue means one real linear combination has no fluctuation in this state. An isotropic matrix has no unique axis direction, so a plot should show the whole degenerate eigenspace rather than inventing a preferred vector.

This ellipse is not a joint probability contour for incompatible measurements. The complex G retains k and can have a different determinant and spectrum from V. Even if a classical Gaussian with covariance V can be drawn mathematically, that drawing does not construct a joint apparatus for A and B.

## 8. Separate within-preparation and between-mean variation

For an ensemble ρ=Σpᵢρᵢ, let μA,i and μB,i be component means. Expand centered products around those component means. The cross covariance and both variances split into two contributions.

$$
v_A=\sum_i p_i v_{A,i}+\sum_i p_i(\mu_{A,i}-\mu_A)^2,\qquad
c=\sum_i p_i c_i+\sum_i p_i(\mu_{A,i}-\mu_A)(\mu_{B,i}-\mu_B).
$$

The between-mean term is real. The imaginary covariance therefore averages entirely through the within-component term. For I/2 with A=Z, an equal Z-eigenstate ensemble gives within/between variance 0/1; an equal X-eigenstate ensemble gives 1/0. Both describe the same state-only variance one.

This parallels the law of total variance in statistics, but the preparation label must be specified. It is misleading to assign the two terms permanent “classical” and “quantum” percentages belonging to ρ alone. Retaining an external label changes which conditional questions can be asked.

## 9. Position and momentum require domains as well as symbols

On the real line, X multiplies by x and P=−iℏ∂x. They are unbounded operators: products and integration by parts require suitable common domains and finite relevant moments. For the following smooth Gaussian all boundary terms vanish. Its density has mean x₀ and position variance a².

$$
\psi(x)=(2\pi a^2)^{-1/4}
\exp\!\left[-\frac{(x-x_0)^2}{4a^2}+\frac{ip_0x}{\hbar}\right],\quad a>0,
\qquad \sigma_X=a,\quad\sigma_P=\frac{\hbar}{2a},\quad
\sigma_X\sigma_P=\frac\hbar2.
$$

Differentiating shows (P−p₀)ψ=iℏ(x−x₀)ψ/(2a²), exactly the linear-dependence condition. Making the wave packet narrower increases its momentum spread. This is a property of the prepared wavefunction before describing any apparatus.

Finite matrices cannot satisfy [X,P]=iℏI exactly when ℏ≠0: the trace of every finite commutator is zero, while the proposed right side has trace iℏd. A finite-grid discretization can approximate selected low-energy states with controlled boundary effects, but it cannot erase this obstruction. The calculator's arbitrary Hermitian matrices are not exact canonical position and momentum.

## 10. Preparation width is not estimation error or device noise

For N independent copies, an unbiased sample mean of A has variance vA/N. This conclusion uses independence and a fixed state; correlations or model misspecification change it. It does not reduce vA itself. Estimating a variance also has statistical uncertainty, absent from an exact-probability experiment.

| Quantity | Question answered | Extra assumptions needed |
| --- | --- | --- |
| σA | How broad are ideal A outcomes in this state? | State and observable |
| Standard error | How variable is a statistic across samples? | Sampling and estimator model |
| Device RMS error ε | How does the chosen pointer compare with A? | POVM, pointer values and error definition |
| Disturbance η(B) | How does the channel change the chosen B observable? | Instrument and disturbance definition |

The next chapter builds explicit instruments. Replacing σA by ε(A) and σB by η(B) in Robertson's inequality without a theorem would change the question and can give a false statement.

## 11. Use the finite experiment and its limits deliberately

The experiment accepts d=1…4, four nonnegative ensemble weights, complex raw state representatives, arbitrary Hermitian upper-triangle entries and real combination coefficients. Positive-weight vectors must be nonzero; inactive coordinates must be zero. Up to four pure components can represent any state at these dimensions through an eigen-ensemble, although the experiment does not infer that ensemble from sampled data.

Moments, complex covariance, determinant slacks, decomposition and projection accounts use exact rational arithmetic. Standard deviations, weighted fluctuation coordinates and two-by-two covariance spectra use 768-bit work with outward 512-bit reporting. A tiny positive eigenvalue remains positive; a repeated eigenspace remains merged. Display rounding is separate from an exact equality certificate.

The computation supplies neither a noncommuting joint distribution nor a measurement apparatus. General observable diagonalization, continuous wavefunctions and finite-shot inference are different tasks. A large matrix-entry denominator is not a promise of experimental accuracy.

## Research extension: several fluctuations and a calibrated qubit purity identity

### Keep the entire complex moment matrix

The two-vector proof extends to any finite list of Hermitian observables. Center each one in the same state and form Hilbert–Schmidt fluctuation vectors:

$$
\Delta A_i=A_i-\operatorname{Tr}(\rho A_i)I,\quad Z_i=\Delta A_i\sqrt\rho,\quad
 G_{ij}=\operatorname{Tr}(Z_i^\dagger Z_j)=V_{ij}+iC_{ij},\quad G\succeq0,\quad
 V_{ij}=\tfrac12\operatorname{Tr}\rho\{\Delta A_i,\Delta A_j\},\quad
 C_{ij}=\frac{\operatorname{Tr}\rho[A_i,A_j]}{2i}.
$$

For any complex coefficients z, z†Gz is the squared norm of Σz_i Z_i and cannot be negative. For real coefficients u, uᵀVu is the variance of the Hermitian observable Σu_iA_i. These are different kinds of linear combination. The imaginary antisymmetric part cancels for real u but still constrains the full Gram matrix.

Positivity supplies all principal-minor inequalities, not only a selected two-observable relation. This does not produce a joint distribution or an apparatus for incompatible observables. It is a constraint on moments of a specified state. The preparation-versus-device distinction follows the background discussed in [Ozawa §II](https://arxiv.org/pdf/quant-ph/0510083); the finite Pauli identities below are derived directly here.

### Resolve the three-Pauli spectrum

Use the ordered list X,Y,Z and a qubit Bloch vector r. Pauli multiplication gives the full matrix:

$$
\rho=\tfrac12(I+\mathbf r\cdot\boldsymbol\sigma),\quad s=|\mathbf r|\le1,\quad
 G_{ij}=\delta_{ij}-r_ir_j+i\sum_k\varepsilon_{ijk}r_k,\quad
 V=I_3-\mathbf r\mathbf r^{\mathsf T}.
$$

For example G_xy=−r_xr_y+i r_z. The plus sign of this imaginary part follows from XY=iZ and from the chapter's upper-right covariance convention. Conjugating the wrong inner-product slot would reverse it.

A proper real rotation preserves the orientation tensor and transforms the moment matrices by congruence:

$$
R\in SO(3):\quad G(R\mathbf r)=R\,G(\mathbf r)R^{\mathsf T},\qquad
 V(R\mathbf r)=R\,V(\mathbf r)R^{\mathsf T}.
$$

When s>0, rotate r to (0,0,s). Then G has a 2×2 block with diagonal1 and off-diagonal ±is, plus the entry1−s². V is diagonal in that frame. Hence:

$$
\operatorname{spec}G=\{1+s,1-s,1-s^2\},\qquad
 \operatorname{spec}V=\{1,1,1-s^2\}.
$$

At s=0 all three eigenvalues coincide and there is no distinguished Bloch axis. At s=1, G has rank1 while V has rank2. The complex and real notions of fluctuation dependence cannot be identified.

### An uncertainty slack can reveal purity under specific assumptions

Taking the determinant, or any two-Pauli principal determinant, gives:

$$
\det G=(1-s^2)^2,\quad
 \det G_{\{i,j\}}=1-s^2=2\bigl[1-\operatorname{Tr}(\rho^2)\bigr]\quad(i\ne j).
$$

For this calibrated qubit model the two-observable Schrödinger slack determines purity. It includes the commutator expectation: for X,Y that requires r_z as well as the X and Y means. It is not a reconstruction of purity from only two unrelated outcome histograms.

For more general real Pauli directions, use bilinearity of the centered moments. Choose an orthonormal basis of the plane spanned by the two directions; its 2×2 change of coordinates multiplies the determinant by the squared area. This gives:

$$
A=\mathbf a\cdot\boldsymbol\sigma,\quad B=\mathbf b\cdot\boldsymbol\sigma:\qquad
 v_Av_B-|c_{AB}|^2=(1-|\mathbf r|^2)|\mathbf a\times\mathbf b|^2.
$$

If the directions are parallel, the slack vanishes for every state. Near-parallel directions make it small even for a mixed state. Thus an almost-zero measured slack does not by itself certify almost-purity: dimension, observable calibration, axis separation and statistical errors matter. A generic higher-dimensional pair does not obey this qubit identity.

### Complex dependence need not mean a noiseless real combination

The +z pure state is an instructive endpoint:

$$
\mathbf r=(0,0,1):\quad \Delta Y\psi=i\Delta X\psi,\quad\Delta Z\psi=0,\quad
 \operatorname{rank}_{\mathbb C}G=1,\quad\operatorname{rank}_{\mathbb R}V=2,\quad
 \operatorname{Var}(uX+vY)=u^2+v^2\quad(u,v\in\mathbb R).
$$

Its X,Y covariance submatrix V is the identity, so every nonzero real combination in that plane has positive variance. Yet the complex fluctuation vectors are dependent and the complex Gram determinant vanishes. The coefficient i does not define a Hermitian combination iX−Y whose histogram would be noiseless. This is the same distinction behind using complex coefficients in a least-squares projection but real coefficients for an observable ellipse.

For a rational family that includes the earlier tilted pure example, take:

$$
\rho_s=\tfrac12\left[I+\tfrac s5(3X+4Y)\right],\quad 0\le s\le1,\quad
 v_X=1-9s^2/25,\quad v_Y=1-16s^2/25,\quad c_{XY}=-12s^2/25.
$$

```python
from fractions import Fraction as Q
for s in [Q(0), Q(1, 4), Q(3, 5), Q(1)]:
    purity = (1+s*s)/2
    pair_slack = 1-s*s
    full_determinant = pair_slack*pair_slack
    rank = 1 if s == 1 else 3
    assert pair_slack == 2*(1-purity)
    print(s, purity, pair_slack, full_determinant, rank)
```

| s | purity | pair slack | three determinant | complex rank |
|---|---|---|---|---|
| 0 | 1/2 | 1 | 1 | 3 |
| 1/4 | 17/32 | 15/16 | 225/256 | 3 |
| 3/5 | 17/25 | 16/25 | 256/625 | 3 |
| 1 | 1 | 0 | 0 | 1 |

![The three-Pauli complex Gram eigenvalues and real covariance eigenvalues approach different ranks at a pure-state boundary, despite being built from the same preparation.](/assets/learning/quantum-uncertainty-research-pauli-gram-purity.en.svg)

The table uses exact fractions. The figure shows rotationally invariant eigenvalues, not extra simultaneously measured outcomes. The existing two-observable reader remains a pair calculator; a three-observable moment-matrix control has not been added.

#### Research check 1: the maximally mixed qubit

For r=0, all three variances equal1, all covariances vanish and both G and V equal I₃. Each Pauli pair has strong slack1 and purity is1/2. A zero commutator expectation did not make either operator pair compatible.

#### Research check 2: nearly parallel axes

For a maximally mixed state, choose unit directions separated by angle θ. Their strong slack is sin²θ, which tends to0 as the axes coincide although purity stays1/2. A small slack must be interpreted with the known axis geometry.

#### Research check 3: do not transfer the identity to a qutrit

The chapter's pure e₀ qutrit with two independent transitions has pair slack1. Its fluctuations span two independent orthogonal directions. Using the qubit purity formula there would give the wrong conclusion because the Pauli closure and dimension assumptions have failed.

## 12. Exercises, explanations and research directions

**1. For I/2 with X,Y, does a zero commutator expectation imply compatible operators?**

**Explanation.** No. [X,Y]=2iZ is nonzero but its expectation is zero. Both variances are one, c=0 and the strong slack is one. The state-dependent lower bound can be trivial for incompatible operators.

**2. In the tilted pure-qubit example, which inequality is saturated?**

**Explanation.** Schrödinger: vXvY=s²=144/625. Robertson has lower bound zero and positive slack144/625. Dropping covariance loses the operative constraint.

**3. What is the best complex projection when vA=0 and vB=1?**

**Explanation.** Every coefficient is optimal because the first fluctuation vector vanishes. The coefficient is null/nonunique and the minimum residual is one, despite det G=0.

**4. Why does the pure three-level example fail to saturate?**

**Explanation.** A e₀=e₁ and B e₀=e₂ are orthogonal nonzero fluctuations. Their Gram matrix is the identity. Purity alone does not imply linear dependence in a space with more than one orthogonal fluctuation direction.

**5. Does changing a constant offset or using more samples remove preparation uncertainty?**

**Explanation.** An offset disappears upon centering, so vA is unchanged. Independent sampling reduces the variance of the estimated mean to vA/N, not the outcome variance of a single preparation.

**6. Can a finite-matrix simulation verify the exact canonical commutator?**

**Explanation.** No; the trace obstruction rules it out. It can test a stated approximation on selected states, with discretization and domain errors analyzed separately.

Complex Gram geometry connects this chapter to kernel methods, covariance estimation and least squares. Extensions include many-observable covariance matrices, entropic uncertainty, state discrimination, quantum estimation and energy-dependent speed limits. Each adds its own assumptions: time is a parameter in ordinary Schrödinger evolution, so one must not invent a universal time operator by replacing X with time in the canonical relation. These research topics remain open parts of the curriculum rather than completion claims for this finite experiment.

Open the textbook through **Book → Interactive Book**. The distinction between preparation widths and measurement error is discussed in [Ozawa's original paper, §II](https://arxiv.org/abs/quant-ph/0510083). The Gram derivation, examples, numerical contract and comparisons above are developed for this textbook.
