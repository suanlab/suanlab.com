## 1. A spin label specifies an operator representation

The preceding chapter studied spin 1/2 in two dimensions. More generally, a single finite irreducible angular-momentum representation has a label j and a basis with magnetic labels m. The label j determines the Casimir value and the dimension. The label m determines one chosen component, conventionally Jz. It does not assign simultaneous x and y values.

We derive the general structure before examining finite numerical examples. The experiment supports j=0,1/2,…,4; that bound limits its matrix size, not the mathematical theory. For j=0 the space has dimension 1, not 0. Higher integer j examples describe valid representations without asserting that every label corresponds to an elementary particle found in nature. Orbital and intrinsic angular momentum share an algebra but have different physical realizations.

## 2. Define ladder operators from the commutator algebra

Assume Hermitian generators Jx, Jy and Jz on a finite inner-product space. Their algebra and the combinations called raising and lowering operators are as follows, with cyclic versions of the first commutator.

$$
[J_x,J_y]=i\hbar J_z,\qquad
J_\pm=J_x\pm iJ_y,\qquad
[J_z,J_\pm]=\pm\hbar J_\pm,\qquad
[J_+,J_-]=2\hbar J_z,\qquad J_+^\dagger=J_-.
$$

The adjoint relation uses the space's inner product. It need not look like an ordinary conjugate transpose in arbitrary coordinates. We will make that distinction explicit rather than assuming that every coefficient basis is orthonormal.

If Jzψ=ℏmψ, then commuting Jz through J± shows that a nonzero image has component value ℏ(m±1). The word ladder refers to this step in an eigenvalue label. It does not specify a unitary time evolution or a measurement procedure.

## 3. Positivity determines the ladder coefficients

Because J² commutes with the three generators, work within one irreducible sector where J² has one scalar value. The identities below express squared ladder-image norms using observables that can be diagonalized together.

$$
\mathbf J^2=J_z^2+\frac{J_+J_-+J_-J_+}{2},\qquad
\|J_\pm|j,m\rangle\|^2
=\hbar^2\left[j(j+1)-m(m\pm1)\right].
$$

At this point the second equation anticipates the Casimir label fixed by termination. To derive it, first call the Casimir value c. The squared norms are c−ℏ²m(m+1) and c−ℏ²m(m−1). They must be nonnegative. A finite ladder cannot continue beyond every m, so its top is annihilated by J+ and its bottom by J−.

Writing the highest label as j gives c=ℏ²j(j+1). The lowest label then solves m(m−1)=j(j+1); the root compatible with lying below the top is m=−j. Unit steps between these ends require 2j to be a nonnegative integer. Finite-dimensional Hermiticity and irreducibility are assumptions of this argument; it is not a domain proof for unbounded differential operators.

## 4. Read a general finite ladder

With phases of the basis chosen consistently, the normalized matrix elements have positive square-root coefficients.

$$
J_\pm|j,m\rangle=\hbar\sqrt{(j\mp m)(j\pm m+1)}\,|j,m\pm1\rangle,\qquad
m=-j,-j+1,\ldots,j,\qquad d=2j+1.
$$

At an endpoint the coefficient vanishes and the whole image is the zero vector. There is no normalized state at a label outside the ladder. An arbitrary state can be a superposition of several m values, so applying J+ need not produce a single basis state.

For j=3/2, the Casimir is 15ℏ²/4 and the three links have magnitudes √3ℏ, 2ℏ and √3ℏ. The following table lists operator coefficients between normalized basis states; a zero means annihilation.

| m | Raising coefficient | Lowering coefficient |
| --- | --- | --- |
| 3/2 | 0 | √3ℏ |
| 1/2 | √3ℏ | 2ℏ |
| −1/2 | 2ℏ | √3ℏ |
| −3/2 | √3ℏ | 0 |

A ladder diagram should label edges with these operator coefficients and vertices with m and Jz=ℏm. Edge size is not a transition probability. Repeated application eventually annihilates every vector; the operator itself has nilpotence order d for a nontrivial irreducible ladder.

## 5. Exact polynomial coordinates need a metric

Square roots can be moved from the operator coefficients into the basis normalization. Let n=2j and use homogeneous polynomials in two auxiliary variables. These variables are not particle positions. The chosen inner product gives the polynomial basis the following diagonal metric.

$$
b_k=x^{n-k}y^k,\quad C_k=\binom nk,\quad G_{kk}=\frac1{C_k},\quad
e_k=\sqrt{C_k}\,b_k,\qquad
N=z^\dagger Gz,\quad u_k=\frac{z_k}{\sqrt{C_k}},\quad u^\dagger u=N.
$$

The b basis is orthogonal in this metric but generally is not normalized. The e basis is orthonormal. The raw coefficients z and u describe the same abstract vector in these two bases. The normalized physical amplitude is u/√N.

Squaring z entries and dividing by their Euclidean sum computes a coefficient histogram, not generally the physical Jz law. Even equal coefficients can have unequal physical probabilities. This is an example of why a coordinate array and a vector's geometry must be specified together.

## 6. Restore the physical adjoint and matrix

Polynomial differentiation gives rational ladder entries and the diagonal component directly.

$$
J_+=\hbar x\partial_y,\quad J_-=\hbar y\partial_x,\quad
J_z=\frac\hbar2(x\partial_x-y\partial_y),\qquad
J_+b_k=\hbar k b_{k-1},\quad J_-b_k=\hbar(n-k)b_{k+1},\quad
J_zb_k=\hbar(n/2-k)b_k.
$$

These formulas preserve the homogeneous degree n. They therefore act inside the selected finite representation. The coefficient k is different from the normalized square-root coefficient because the b vectors have different lengths.

Let B map polynomial coefficients to orthonormal coordinates. An operator and its adjoint transform with the corresponding metric, as follows.

$$
B=\operatorname{diag}(C_k^{-1/2}),\quad G=B^\dagger B,\qquad
A_{\mathrm{phys}}=B A_b B^{-1},\quad
(A_{\mathrm{phys}})_{ik}=(A_b)_{ik}\sqrt{C_k/C_i},\qquad
A^{\dagger_G}=G^{-1}A_b^\dagger G.
$$

Consequently Jx, Jy and Jz are Hermitian in the weighted coordinates and become ordinary Hermitian matrices in the orthonormal coordinates. For J+, the weighted adjoint is J−. Testing a plain conjugate transpose on the polynomial matrix can fail without indicating a failure of physical Hermiticity. The metric and basis map must accompany the matrix.

## 7. Work through unequal weights in spin one

Take j=1, ℏ=1 and polynomial coefficients z=(1,1,1). The binomial factors are (1,2,1), so N=1+1/2+1=5/2. The physical normalized vector and Jx matrix are the following.

$$
\psi=\begin{pmatrix}\sqrt{2/5}\\1/\sqrt5\\\sqrt{2/5}\end{pmatrix},\qquad
J_x=\frac1{\sqrt2}\begin{pmatrix}0&1&0\\1&0&1\\0&1&0\end{pmatrix},\qquad
(p_{m=1},p_{m=0},p_{m=-1})=(2/5,1/5,2/5).
$$

In polynomial coordinates Jx has rows (0,1/2,0), (1,0,1) and (0,1/2,0). Acting on z gives (1/2,2,1/2). The weighted inner product with z is 2, so the mean is 2/(5/2)=4/5. The weighted squared norm of that image is 5/2, giving second moment 1 and variance 1−16/25=9/25.

| Component | Mean | Second moment | Variance |
| --- | --- | --- | --- |
| Jx | 4/5 | 1 | 9/25 |
| Jy | 0 | 1/5 | 1/5 |
| Jz | 0 | 4/5 | 4/5 |

The naive coefficient histogram would be (1/3,1/3,1/3). The physical law instead includes the inverse binomial weights. The three second moments sum to 2, agreeing with j(j+1), while the variances sum to 34/25. This calculation checks geometry, normalization and moment interpretation at the same time.

## 8. An operator image norm is not a probability

For the same state, raising once gives polynomial image (1,2,0). Its weighted squared norm is 3. Relative to the original input norm 5/2, the action norm squared is 6/5. Normalizing that image gives a different vector for comparison.

$$
\|J_+\psi\|^2=\frac65,\qquad
\frac{J_+\psi}{\|J_+\psi\|}=\begin{pmatrix}1/\sqrt3\\\sqrt{2/3}\\0\end{pmatrix},\qquad
\frac{\|J_+^k z\|_G^2}{N}=1,\frac65,\frac85,0\quad(k=0,1,2,3).
$$

A value exceeding 1 is not an invalid probability because no probability was defined. J+ is not unitary, and this experiment supplies no measurement instrument whose outcome it represents. One can design operational models involving suitably scaled operators, but that would be an additional specification.

The raw chain is not normalized after each application. A k-fold action has angular-momentum units to the power k, and its squared norm has action units to the power 2k. The initial record is dimensionless; the next has action² units. Multiplying the numerical ℏ by c multiplies the k-th squared norm by c^(2k), while the normalized image direction stays fixed for positive c. A zero image has neither a normalized vector nor a probability law; its reported value is null.

## 9. The mean vector does not specify a higher-spin state

The Casimir fixes a sum of second moments. It is not generally the squared length of the mean vector. The relation and a bound on that length are useful diagnostics.

$$
\sum_{i=x,y,z}\operatorname{Var}(J_i)
=\hbar^2j(j+1)-|\langle\mathbf J\rangle|^2,\qquad
|\langle\mathbf J\rangle|\le\hbar j.
$$

To read the bound, choose an axis along the nonzero mean vector. Its expectation is that vector’s length and cannot exceed the largest projection eigenvalue ℏj. For a zero mean the bound is immediate.

For a spin-coherent state pointing along +x, the mean Jx is ℏj and its variance is 0. The remaining two variances sum to ℏ²j. In spin one, an example has orthonormal coordinates (1/2,1/√2,1/2), corresponding to polynomial coefficients proportional to (1,2,1). Its Casimir is still 2ℏ², while its mean-vector length is ℏ.

Not every pure higher-spin state is coherent. The spin-one state with m=0 and the equal superposition of m=+1 and m=−1 both have zero mean vector, yet their Jz laws are (0,1,0) and (1/2,0,1/2). Thus a three-dimensional spin-mean arrow is not a full state description. The pure-qubit Bloch-sphere picture must not be promoted to every j.

## 10. Numerical coordinates and conditioning

The preparation uses twenty-one owned integers: twice the spin, the ℏ ratio and nine possible complex raw coefficients. Active coordinates end at k=2j; all later entries must be zero. Reducing j does not silently discard an occupied component. The selected matrix dimension is at most 9, and its limits are separate from the general derivation above.

Weighted matrices, probabilities, moments and raw ladder chains use exact complex rationals. Orthonormal matrices and normalized amplitudes can contain square roots. They use directed 768-bit work arithmetic and outward 512-bit observations. Exact algebraic zeros differ from numerical residual intervals that include zero.

A very small ℏ can make a repeated raw image tiny. For its normalized display, divide all coordinates by the same positive exact rational scale before taking a square root. This algebraically preserves the representative, including phase, and keeps the root denominator well-conditioned. The equality a/√M=(a/L)/√(M/L²) explains why this changes the arithmetic expression without changing the normalized image. With L chosen from the largest absolute real or imaginary component, the scaled weighted mass lies between 1/70 and 18 for the selected dimensions. Raw chain values, dimensionful norm accounts and the precision criterion stay unchanged. No per-step norm correction is applied. Independent references construct orthonormal square-root ladders first and transform them back; they do not reuse the production polynomial differentiation matrices.

## 11. Exercises with explanations

**1. How many states are in the j=3/2 basis?**

**Explanation.** Four: m=3/2,1/2,−1/2,−3/2. The dimension is 2j+1. The corresponding Casimir value is 15ℏ²/4, not the square of the highest component alone.

**2. Why is the weighted adjoint of a polynomial matrix needed?**

**Explanation.** Its coefficient basis has metric G. The adjoint defined by that inner product is G⁻¹A†G. Only after the basis map B makes the metric I can we use ordinary conjugate transpose as the coordinate formula for the physical adjoint.

**3. What is the Jz law of polynomial z=(1,1,1) in spin one?**

**Explanation.** The weights are (1,1/2,1) and N=5/2, giving (2/5,1/5,2/5). Equal raw coefficients are not equal physical probabilities in this basis.

**4. Is the raising norm 6/5 a probability error?**

**Explanation.** No. It is an operator-image norm relative to the input normalization. The ladder action is not a specified measurement event. Its normalized image has probabilities (1/3,2/3,0), a different mathematical object.

**5. What happens when J+ annihilates the highest state?**

**Explanation.** The result is the zero vector and its action norm is 0. There is no normalized image outside the ladder; the calculation returns null rather than normalizing with an epsilon or choosing a replacement state.

**6. Does a zero spin-mean vector identify the state?**

**Explanation.** No. The spin-one examples in section 9 are different pure states with the same three means. Their Jz distributions distinguish them. Higher moments or the full state are needed beyond a mean arrow.

## 12. Connections and remaining scope

This chapter connects representation theory, metric-aware linear algebra, positive quadratic forms and numerical conditioning. For AI work, a coordinate-dependent matrix should be interpreted with its inner product and basis map. A matrix that looks asymmetric in one coefficient basis can represent a self-adjoint operator, while a well-behaved norm on one input cannot certify its operator structure. Preserving an exact accounting identity and maintaining a stable display are different requirements.

The experiments illustrate arbitrary-j formulas through bounded finite matrices. They do not implement orbital spherical harmonics, differential-operator domains, a tensor-product decomposition, general mixed states or angular-momentum coupling. Those topics and the remaining research depth continue later. Open the textbook through **Book → Interactive Book**.

The angular-momentum algebra and termination background were checked against [MIT8.321 Lecture19](https://ocw.mit.edu/courses/8-321-quantum-theory-i-fall-2017/c491464fb065aa4cc8a99cc562475183_MIT8_321F17_lec19.pdf). [Preskill §2.2.1](https://www.preskill.caltech.edu/ph219/chap2_15.pdf) supplies the spin representation context. The polynomial metric, worked states, conditioning contract and explained exercises are developed for this textbook.
