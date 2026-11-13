## 1. A tensor network stores a rule for amplitudes

An n-qubit vector has one complex amplitude for each binary string. For three qubits the strings run from 000 to 111; for n qubits there are 2^n. Writing a Hamiltonian as a sum of local terms does not automatically make its state vector small. We need a separate statement about the structure of that vector and the representation used to store it.

A matrix product state, MPS, represents amplitudes by contracting a chain of small tensors. Think of a physical bit as selecting a matrix at each site. Multiplying the selected matrices gives the amplitude of that whole bit string. The internal matrix index is summed; it is not another measured particle. Its size is the bond dimension.

$$
\psi_{s_0\cdots s_{n-1}}=\sum_{a_1,\ldots,a_{n-1}}A_0^{s_0}[1,a_1]A_1^{s_1}[a_1,a_2]\cdots A_{n-1}^{s_{n-1}}[a_{n-1},1],\qquad D_0=D_n=1.
$$

We use open boundaries, so the first left index and final right index have size one. An amplitude is therefore a scalar, not the trace of a periodic matrix product. Site0 is the most significant bit in the dense output. This chapter develops the representation, exact contraction, environments and changes of virtual coordinates. Finding an optimal state or discarding Schmidt components is a separate task.

## 2. Draw the legs before filling arrays

Each site has a physical leg s, a left virtual leg a and a right virtual leg b. An internal line joins two legs with the same dimension and means summing the corresponding index. An external physical leg remains open until a bit value is specified. A drawing becomes an unambiguous calculation only after these index roles and ordering rules are fixed.

$$
A_i^s\in\mathbb C^{D_i\times D_{i+1}},\qquad P=\sum_{i=0}^{n-1}d_iD_iD_{i+1},\qquad P_{\mathrm{qubit},D}=4D+2(n-2)D^2\quad(n\ge2).
$$

Here d_i is the local physical dimension, equal to two in the calculator. P counts active complex entries before removing gauge redundancy; it is not the dimension of the manifold of distinct physical states. For n=3 and both bonds2, the site contributions are4,8,4, giving16 entries. There are only8 dense amplitudes. Small examples need not be compressed.

| Item | Meaning | Three-qubit example |
| --- | --- | --- |
| Physical leg | A local basis choice | s₀,s₁,s₂ each have size2 |
| Virtual leg | A summed representation index | D₁=D₂=2 |
| Site tensor | Two matrices selected by s | Shapes1×2,2×2,2×1 |
| Output vector | All physical amplitudes | 8 complex entries |

The output convention is A[site][physical][left][right]. An input key such as t1Imag203 instead encodes site1,left2,physical0,right3. Treat the documented layout as part of the interface; an array transpose can silently produce a different state even when every dimension still matches.

## 3. Product states and a GHZ path diagram

With every bond dimension one, the selected matrices are scalars and every amplitude factors into local amplitudes. This gives product states. An entangled state needs more than one independent virtual path across at least one cut, although allocating extra paths alone does not create entanglement.

For a concrete entangled example, use the following three-site tensors. All entries are zero or one, and the physical label chooses the superscripted matrix.

$$
A_0^0=(1,0),\quad A_0^1=(0,1),\quad A_1^0=\begin{pmatrix}1&0\\0&0\end{pmatrix},\quad A_1^1=\begin{pmatrix}0&0\\0&1\end{pmatrix},\quad A_2^0=\binom10,\quad A_2^1=\binom01.
$$

For 000, the path through virtual label0 survives all three sites and contributes1. For 111, the label1 path contributes1. Every mixed string is killed by an incompatible projector or boundary vector. The resulting raw vector is |000〉+|111〉, with squared norm2. Its normalized state is GHZ₃. We keep this normalization separate from the tensor definition.

| Bit string | Surviving virtual path | Raw amplitude |
| --- | --- | --- |
| 000 | 0→0 | 1 |
| 111 | 1→1 | 1 |
| 001 | No complete path | 0 |
| 010 | No complete path | 0 |

Try following 101 yourself. The middle physical0 matrix accepts label0, while the first physical1 boundary selects label1, so the product vanishes. This is a useful manual check before asking a visual diagram to animate the contraction.

## 4. Cut the chain into two coefficient maps

Cut after k sites. Collect all physical bits to the left into x and all bits to the right into y. The left contraction L maps a physical prefix to the virtual cut index; the right contraction B maps that virtual index to a physical suffix. Their ordinary matrix product is the dense coefficient matrix at that physical bipartition.

$$
C_{xy}=\sum_aL_{xa}B_{ay},\qquad C=LB,\qquad\operatorname{rank}C\le D_k,\qquad N=\sum_{x,y}|C_{xy}|^2.
$$

The rank inequality follows because every column of C lies in the column span of L. For a nonzero pure state, coefficient rank is Schmidt rank, so the virtual dimension is an upper bound on the rank actually used. Normalizing the state changes singular values by a common factor and leaves the rank unchanged.

For GHZ₃ at the first cut, C has nonzero entries only at(row0,column00) and(row1,column11), both1. Its rank is2, and dimension one cannot represent this state exactly at that cut. A dimension4 representation may still have rank2 or1. Distinguish capacity, actual rank and the sizes of the physical spaces on both sides.

## 5. Environments are Gram matrices with a convention

For each cut, form the left and right environments by summing out their physical strings. Both are positive semidefinite virtual Gram matrices. They summarize everything needed from those blocks for a norm closure, but their virtual entries depend on the chosen representation.

$$
E_{ab}=\sum_x\overline{L_{xa}}L_{xb},\qquad R_{ab}=\sum_y\overline{B_{ay}}B_{by},\qquad N=\sum_{a,b}E_{ab}R_{ab}=\operatorname{Tr}(ER^{\mathsf T}).
$$

Notice the right definition: R_ab sums conjugate(B_ay) times B_by. Expanding the squared norm of C gives E_ab R_ab with the same index order. It is Tr(E Rᵀ), not generally Tr(E R). This transpose is easy to miss if every test uses real diagonal examples. Positivity of both matrices does not make every possible contraction between their entries correct.

As a phase-sensitive example, take L=(1,i) and B=(1,i)ᵀ, using only the physical00 output of a two-site bond2 representation. The sole amplitude is1+i²=0. Both E and R are [[1,i],[-i,1]]. Their elementwise closure is0, while Tr(ER)=4. This example is algebraically valid but defines no normalized physical state. An incorrect closure would invent a positive norm and conceal the cancellation.

## 6. Transfer contraction reuses partial sums

Explicitly listing every virtual path is a clear reference calculation. To compute environments efficiently, contract one site at a time and reuse the previous result. Start with scalar boundary environments and sweep in the appropriate direction.

$$
E_{i+1}=\sum_s(A_i^s)^\dagger E_i A_i^s,\qquad R_i=\sum_s\overline{A_i^s}R_{i+1}(A_i^s)^{\mathsf T},\qquad E_0=R_n=(1).
$$

The left update uses an adjoint, while the right update uses conjugation and transpose in the order required by our R convention. At every cut, closing the two environments must give the same dense norm N. This provides multiple independent places to locate an index or conjugation error; checking only the final scalar can hide compensating mistakes.

The recurrence also explains the diagram of a double-layer network: one layer carries ket tensors and the other their conjugates, with each physical leg summed locally. A prefix amplitude sweep keeps physical strings open; an environment sweep sums them out. These are different calculations with different storage requirements, despite using the same site tensors.

## 7. Insert an observable without constructing every amplitude

An operator can replace the identity on a selected physical leg in the double-layer contraction. Sum its bra/ket matrix entries with the two physical indices, then close the surrounding environments and divide by the global squared norm. The denominator must be positive; a zero contraction cannot be repaired by choosing a convenient normalization.

$$
\langle O_i\rangle=\frac1N\sum_{a,a',b,b',s,t} E_i[a,a']\overline{A_i^s[a,b]}\,O_{st}A_i^t[a',b']R_{i+1}[b,b'],\qquad\langle Y\rangle=-2\operatorname{Im}\rho_{01}.
$$

A local reduced density matrix can be obtained by leaving the bra/ket physical indices open instead. Its entries refer to the physical qubit basis, whereas E and R refer to virtual coordinates. They are not interchangeable density matrices. Compute the dense partial trace as a separate reference for the local result.

For |0〉⊗(|0〉+i|1〉), the squared norm is2. The right density is [[1/2,−i/2],[i/2,1/2]], giving X=0,Y=+1,Z=0 and purity1. Losing the conjugate reverses Y. A real GHZ example cannot detect that sign error, which is why a phase-sensitive product state belongs beside the entangled example.

## 8. Gauge transformations change virtual coordinates

Insert an invertible matrix G on one side of a shared bond and its inverse on the other side. Every adjacent product is unchanged for every pair of physical labels, so every complete amplitude is unchanged. No normalization assumption and no unitary restriction on G is needed.

$$
A_i^s\longmapsto A_i^sG,\qquad A_{i+1}^t\longmapsto G^{-1}A_{i+1}^t,\qquad (A_i^sG)(G^{-1}A_{i+1}^t)=A_i^sA_{i+1}^t.
$$

A gauge is a coordinated change to two neighboring tensors. Multiplying only one tensor by G is usually a physical state change. Multiplying both by G is also generally wrong: the cancellation requires an inverse in the correct position. A singular matrix has no inverse and is not an allowed exact gauge of this form.

This is representation redundancy. It resembles the freedom to factor a matrix as(U G)(G⁻¹ V), rather than time evolution applied to a physical subsystem. The demonstration checks the full complex vector, not just its norm or a few local expectations. Equal norms alone would leave many physically different states indistinguishable.

## 9. A numerical gauge example changes parameter length

Use GHZ₃ and transform its first bond with scale2 and shear1/2. The chosen order is diagonal scale times shear, giving the following G and inverse. Define K as the sum of the squared absolute values of every active tensor entry.

$$
G=\begin{pmatrix}2&1\\0&1\end{pmatrix},\quad G^{-1}=\begin{pmatrix}1/2&-1/2\\0&1\end{pmatrix},\qquad N'=N=2,\quad K=6,\quad K'=\frac{19}{2}.
$$

The first site's entry length changes from2 to6. The middle site changes from2 to3/2, and the last remains2, so K′=19/2. The physical squared norm remains2 and every amplitude remains identical. A Euclidean penalty on tensor coordinates therefore depends on gauge even when the represented state is fixed.

| Quantity for GHZ₃ | Before gauge | After gauge |
| --- | --- | --- |
| Physical squared norm N | 2 | 2 |
| Tensor coordinate length K | 6 | 19/2 |
| First-cut coefficient rank | 2 | 2 |
| First left environment | I₂ | [[4,2],[2,2]] |

The parameter-length change is not evidence of increased entanglement, a changed energy or a better approximation. It is evidence that coordinate geometry and physical-state geometry are different. In optimization research, first specify which of those spaces a regularizer or distance is meant to describe.

## 10. Environment covariance preserves the closure

At the transformed cut, L becomes L G and B becomes G⁻¹ B. Substitution into their Gram definitions gives their transformation laws. The left and right environments usually change, even though the physical state does not.

$$
E'=G^\dagger EG,\qquad R'=\overline{G^{-1}}R(G^{-1})^{\mathsf T},\qquad\sum_{ab}E'_{ab}R'_{ab}=\sum_{ab}E_{ab}R_{ab}.
$$

For the real GHZ gauge above, E′=[[4,2],[2,2]] and R′=[[1/2,−1/2],[−1/2,1]]. The elementwise products sum to2. Neither transformed environment equals the identity, yet closure still agrees with the original norm. These matrices are coordinate-dependent block metrics, not separately preserved physical observables.

With complex shear the conjugation in R′ is essential. A transpose-only test using real G is insufficient. The calculator checks both covariance residual matrices, the inverse residual and the complete state difference. Zero residuals certify the stated finite algebra; they do not establish convergence of an optimization algorithm that has not been run.

## 11. Isometries explain canonical forms

A left-isometric site has orthonormal columns when its left virtual and physical indices are grouped. A right-isometric site instead has orthonormal rows when its physical and right indices are grouped. Their identity matrices have different dimensions.

$$
\sum_s(A_i^s)^\dagger A_i^s=I_{D_{i+1}}\quad\text{(left isometry)},\qquad\sum_s A_i^s(A_i^s)^\dagger=I_{D_i}\quad\text{(right isometry)}.
$$

For the raw GHZ tensors, the first and middle sites are left isometries but the last site's left Gram is the scalar2. This agrees with squared norm2: the earlier identity environments propagate until the final scalar. Dividing the last site's two matrices by sqrt(2) would normalize the vector and make the entire chain left-normalized. This analytic step is separate from the calculator's rational input interface.

Checking one site is not a normalization proof for the chain. A left-isometric tensor is possible only if its column dimension fits within the combined left/physical row dimension; a redundant bond may prevent that condition. Canonical diagnostics report actual Grams and residuals, not an automatic promise that any chosen dimensions can be whitened by an invertible gauge.

## 12. Canonicalization is an algorithm, not a checkbox

To obtain a left-isometric site, reshape its tensor into a matrix and factor it with a rank-aware QR or SVD. The orthonormal factor becomes the new site tensor, while the remaining factor is absorbed into the next site. Keeping the full supported factorization preserves the contracted vector.

$$
M_{(a,s),b}=A_i^s[a,b]=\sum_c Q_{(a,s),c}T_{cb},\qquad\sum_s(Q^s)^\dagger Q^s=I,\qquad A_{i+1}^t\longmapsto T A_{i+1}^t.
$$

If the matrix has deficient column rank, an economy factorization may reduce the virtual dimension. The factor T can then be rectangular and cannot be called an invertible square gauge. Keeping an arbitrary completion instead can carry unused directions. State preservation, numerical stability and dimension reduction must be checked separately.

The current calculator measures canonicality and performs the specified invertible scale/shear change; it does not run QR, SVD or a sweep. The equations here explain the future algorithm and its assumptions. A diagram turning green because a Gram residual is zero would be a diagnostic, not proof that a canonicalization routine has been implemented.

## 13. Correlations distinguish local summaries from global information

For two sites, insert two Pauli operators in the transfer network. Subtract the product of their one-site means to obtain a connected correlation. Pauli X/Y/Z are dimensionless; physical spin components would require the appropriate hbar/2 factors.

$$
C_{ij}^{\alpha\beta}=\langle\sigma_i^\alpha\sigma_j^\beta\rangle-\langle\sigma_i^\alpha\rangle\langle\sigma_j^\beta\rangle,\qquad C_{01}^{ZZ}(\mathrm{GHZ}_3)=1.
$$

Normalized GHZ₃ has every one-site Z mean0 and every pair ZZ mean1. Its connected ZZ is therefore1. A bond1 product state cannot reproduce that nonzero connected correlation. Conversely, giving a product state a redundant bond4 does not change its vanishing connected correlations or make its Schmidt rank4.

Local and pair summaries need not determine a global state. The incoherent mixture of |000〉 and |111〉 has the same one-site states and all two-site Pauli expectations as GHZ₃, yet the three-site XXX expectation is0 instead of1. This mixture is an analytic comparison outside the pure-MPS input mode. Do not infer global coherence merely from a local purity or a large ZZ pair correlation.

## 14. Bond capacity, contraction cost and compression claims

Across each cut, the coefficient rank cannot exceed the virtual dimension. The physical left and right dimensions impose additional bounds. These facts constrain exact representations without identifying the singular values or the error from reducing a bond.

$$
\mathrm{GHZ}_3:\ D_1=D_2=2,\quad\operatorname{rank}C_1=\operatorname{rank}C_2=2;\qquad |000\rangle:\ D_1=D_2=4,\quad\operatorname{rank}C_1=\operatorname{rank}C_2=1.
$$

For a three-qubit product state carried on bonds4, the active-entry count is8+32+8=48, while the dense vector has8 entries. Most directions may be redundant, but the allocated array still costs space. For larger chains with a modest fixed D, tensor storage can grow linearly in site count. An arbitrary highly entangled state may require D itself to grow exponentially, so the qualifier on D matters.

| Three-site representation | Bonds | Active complex entries | Dense amplitudes |
| --- | --- | --- | --- |
| Product state | (1,1) | 6 | 8 |
| GHZ state | (2,2) | 16 | 8 |
| Redundant product | (4,4) | 48 | 8 |

A well-ordered norm contraction with dense matrix multiplication has familiar polynomial costs in n,d,D. The current reference-friendly implementation also reports dense summand slots for its chosen loops. Those slots are not measured runtime, minimal operations or bit complexity. Exact rational numerators can grow. Asking the reader to display every dense amplitude necessarily asks for exponentially many outputs regardless of how efficiently a single norm was contracted.

## Research extension: transfer spectra, correlation length and gauge conditioning

### Build an exactly normalized bond-two family

Choose a physical bit at each site and use the following nonnegative matrices and open boundaries:

$$
0\le p\le1,\quad T=\begin{pmatrix}p&1-p\\1-p&p\end{pmatrix},\quad
 A^0=\begin{pmatrix}\sqrt p&\sqrt{1-p}\\0&0\end{pmatrix},\quad
 A^1=\begin{pmatrix}0&0\\\sqrt{1-p}&\sqrt p\end{pmatrix},\quad
 L=(1,1)/\sqrt2,\quad R=\frac{(1,1)^{\mathsf T}}{\sqrt p+\sqrt{1-p}}.
$$

Each A has only one nonzero row. Multiplying a physical string fixes that row path, so for N sites:

$$
\psi_{s_1\cdots s_N}=L A^{s_1}\cdots A^{s_N}R
 =\frac1{\sqrt2}\prod_{j=1}^{N-1}\sqrt{T_{s_j,s_{j+1}}},\qquad
 |\psi_{s_1\cdots s_N}|^2=\frac12\prod_{j=1}^{N-1}T_{s_j,s_{j+1}},\quad\|\psi\|^2=1.
$$

Summing over the last bit repeatedly uses the row sums of T and proves normalization. The Z-basis probabilities equal those of a stationary two-state Markov chain. The quantum state is their coherent square-root superposition, not the diagonal classical mixture. Other measurement bases can distinguish these objects.

### A virtual transfer map explains a physical correlation

Define the right-to-left transfer map in the convention below. Its fixed matrix I, fixed dual functional ell and spectrum can all be checked by multiplying two-by-two matrices:

$$
\mathcal E(X)=\sum_s A^sX(A^s)^\dagger,\quad
 \mathcal E(I)=I,\quad\mathcal E(Z)=\lambda Z,\quad\lambda=2p-1,\quad
 \operatorname{spec}\mathcal E=\{1,\lambda,0,0\},\quad
 \ell=I/2+\sqrt{p(1-p)}X,\quad\mathcal E^\dagger(\ell)=\ell,\quad\operatorname{Tr}\ell=1.
$$

Here X and Z in the transfer formulas are virtual Pauli matrices. Ell is a virtual trace functional, not automatically the reduced density of a physical site. With the specified finite boundaries, contracting an unmeasured boundary site reaches these stationary environments. Alternatively the finite Z correlations follow directly from the Markov probability law.

Insert the physical Z by subtracting the two Kraus-like terms. The resulting connected correlation for separation r>=1, wherever both sites exist, is:

$$
\mathcal E_Z(X)=A^0X(A^0)^\dagger-A^1X(A^1)^\dagger,\quad
 \mathcal E_Z(I)=Z,\quad\mathcal E_Z(Z)=\lambda I,\qquad
 \langle Z_i\rangle=0,\quad C_{ZZ}(r)=\lambda^r,\quad
 \xi=-\frac1{\ln|\lambda|}\quad(0<|\lambda|<1).
$$

The first insertion produces virtual Z, propagation multiplies it by lambda for each step, and the second insertion returns lambda I. The zero local means make the raw and connected ZZ correlations equal. Negative lambda gives alternating signs, while its absolute value controls the envelope. Distances and xi are measured in lattice spacings, not physical time. These transfer ideas are discussed in [Schollwöck, §4.2.2](https://arxiv.org/pdf/1008.3477); this particular family and its finite-boundary proof are developed here.

A fixed bond capacity does not impose a uniform upper bound on this correlation length:

$$
p=1-\varepsilon,\quad\varepsilon\downarrow0:\qquad
 \lambda=1-2\varepsilon,\quad\xi\sim\frac1{2\varepsilon},\quad
 D=2\ \text{throughout}.
$$

The limiting p=1 state has long-range ZZ order with bond dimension2. At p=1/2 correlations vanish beyond the same site, so the logarithmic expression with lambda=0 should be read via that direct statement, not a division by a guessed small eigenvalue.

### Preserve the physical state when changing virtual coordinates

Use an invertible complex G at every virtual bond, including the boundary transformations. For column-vectorized virtual matrices:

$$
A'^s=G^{-1}A^sG,\quad L'=LG,\quad R'=G^{-1}R,\quad
 S=\overline G\otimes G,\quad E'=S^{-1}ES,\quad\kappa_2(S)=\kappa_2(G)^2,\qquad
 \|E'^r-P'\|_2\le\kappa_2(S)\|E^r-P\|_2,\quad
 P'=S^{-1}PS\quad(0<p<1).
$$

E is the matrix of the transfer superoperator and P is its stationary spectral projector in the interior0<p<1. Similarity preserves its eigenvalues, but a nonunitary gauge can enlarge norms and numerical prefactors. The condition-number factor follows by submultiplicativity; it is a bound, not necessarily attained. Gauge-invariant physical contractions transform their boundaries and operator insertions consistently. An enlarged raw transfer norm does not by itself mean a larger physical correlation.

The two endpoints of the simplest comparison illustrate different rank usage:

$$
p=1:\quad\psi=(|0\cdots0\rangle+|1\cdots1\rangle)/\sqrt2,\quad C_{ZZ}(r)=1;\qquad
 p=\tfrac12:\quad\psi=|+\rangle^{\otimes N},\quad C_{ZZ}(r)=0\quad(r\ge1).
$$

At p=1/2 the allocated bond2 is redundant for a product state; a minimal bond1 representation exists. At p=1 the GHZ correlations do not decay and there are multiple peripheral fixed directions. A generic statement that every finite-D MPS must have finite correlation length would fail on this example. No parent Hamiltonian or its excitation gap has been specified here.

Choose N>=5 for the distance-four column; this analytic family does not expand the reader's size limit.


```python
from fractions import Fraction as Q
for p in [Q(1, 4), Q(1, 2), Q(3, 4), Q(1)]:
    second = 2*p-1
    print(p, second, second, second**2, second**4)
```

| parameter p | lambda | CZZ(1) | CZZ(2) | CZZ(4) |
|---|---|---|---|---|
| 1/4 | -1/2 | -1/2 | 1/4 | 1/16 |
| 1/2 | 0 | 0 | 0 | 0 |
| 3/4 | 1/2 | 1/2 | 1/4 | 1/16 |
| 1 | 1 | 1 | 1 | 1 |

For a more general transfer operator, diagonalizability is another assumption to check. A size-two Jordan block provides the elementary warning:

$$
J_\lambda=\lambda I+N,\quad N^2=0,\quad\lambda\ne0:\qquad
 J_\lambda^r=\lambda^rI+r\lambda^{r-1}N\quad(r\ge1).
$$

Its powers carry a polynomial prefactor as well as the exponential. This is a conditional linear-algebra comparison; our explicit family above does not need such a prefactor. In addition, an observable may have zero overlap with a subleading mode. A fitted decay length therefore requires specifying the transfer spectrum, insertions, boundaries and asymptotic regime, rather than reading one matrix eigenvalue as every possible observable's decay.

![A bond-two family has alternating, decaying or long-range connected ZZ correlations; its correlation length grows without increasing the allocated bond dimension.](/assets/learning/quantum-representation-research-transfer-correlation-gauge.en.svg)

The table is exact. These are analytic transfer-family and conditioning examples, not an implementation of an infinite-MPS solver or a canonicalization/truncation sweep. The existing finite reader keeps its general complex rational tensors, environments and scale/shear comparison.

#### Research check 1: the sign survives odd distances

At p=1/4, lambda=−1/2 and xi=1/ln2. The correlations at distances1,2,4 are−1/2,1/4,1/16. At p=3/4 they have the same magnitudes but opposite signs at odd distance. Fitting only even separations loses that information.

#### Research check 2: long range with a small bond

At p=1, only all-zero and all-one strings survive. For N>=2, one-site Z means vanish but every distinct pair has ZZ=1. Bond dimension2 is enough for this nondecaying correlation. Its transfer spectrum has an additional eigenvalue1, so the isolated-leading-mode exponential argument does not apply.

#### Research check 3: a large coordinate prefactor

For G=diag(g,1) with g>0, kappa2(G)=max(g,1/g), and the transfer similarity can have condition number kappa2(G)². The physical state and exact ZZ table stay fixed when both boundaries transform. A penalty on virtual matrices or an intermediate norm can change without changing the physical target.

## 15. Eight exercises with worked explanations

**1. Trace a path.** Evaluate the raw amplitudes000,111 and101 for the displayed GHZ tensors, and obtain the squared norm.

**Explanation.** The compatible label0 path gives amplitude1 for000 and label1 gives1 for111. The projector chosen by middle0 kills the path selected by first1, so101 has amplitude0. The other mixed strings vanish similarly, giving N=2. Normalized nonzero amplitudes are1/sqrt(2), not1/2.

**2. Separate capacity and rank.** Can bond1 represent normalized GHZ₃ exactly? Does placing |000〉 on bond4 give Schmidt rank4?

**Explanation.** The GHZ coefficient matrix has rank2 at both cuts, exceeding bond1. The product coefficient matrix has just one nonzero entry and rank1. Extra virtual coordinates create capacity and redundancy; they do not force that capacity to be used by the state.

**3. Check the closure orientation.** For L=(1,i) and B=(1,i)ᵀ, compute the physical amplitude, E,R and both proposed closures.

**Explanation.** The amplitude is1−1=0. Both Grams are [[1,i],[-i,1]]. Summing E_ab R_ab gives1−1−1+1=0. Tr(ER)=4 is wrong for this convention. No normalized state or local physical density may be reported from this contraction.

**4. Check a phase sign.** Find the right-qubit Bloch vector of |0〉⊗(|0〉+i|1〉).

**Explanation.** Normalize by sqrt(2). The right density has upper-right entry−i/2, so X=2Reρ₀₁=0,Y=−2Imρ₀₁=1,Z=0. This catches a conjugation error that real states cannot expose. The global state's first qubit remains |0〉.

**5. Insert a gauge.** Apply the displayed G to the first GHZ bond. What happens to N,K and the rank?

**Explanation.** G G⁻¹ cancels before any physical-string sum, preserving all amplitudes, N=2 and both ranks2. The separate site entry lengths become6,3/2,2, so K changes from6 to19/2. A tensor norm penalty can change along a single physical-state representation orbit.

**6. Close transformed environments.** Use E′=[[4,2],[2,2]] and R′=[[1/2,−1/2],[−1/2,1]]. Why need neither matrix remain the identity?

**Explanation.** The four products are2,−1,−1,2 and sum to2. The two matrices carry opposite coordinate changes. Gauge invariance applies to their physical closure, not to each virtual metric entry independently. Complex G additionally requires the stated conjugations.

**7. Locate normalization.** The first and middle raw GHZ sites are left isometries. Is the raw state normalized, and what is an analytic normalization choice?

**Explanation.** The final left Gram is2, so the full squared norm is2. Dividing only the last site by sqrt(2) produces unit norm while retaining the earlier isometries. The rational calculator can instead keep raw tensors and normalize physical outputs; those are distinct representation choices.

**8. Find a global ambiguity.** Why do the GHZ state and its equal incoherent00…0/11…1 mixture defeat a conclusion based only on one-site and pair expectations?

**Explanation.** For three sites, tracing out any one qubit removes the GHZ cross terms, giving the same pair density as the mixture. Their XXX expectations differ: it exchanges the two GHZ components coherently but has zero diagonal expectation in the mixture. A larger observable set is required to distinguish them.

## 16. What has been calculated and what comes next

The finite model accepts one to four ordered qubits, bonds1…4 and general complex rational tensors. It contracts the full vector, computes all-cut environments and closures, physical local densities, pair Pauli correlations, exact coefficient ranks and canonicality diagnostics. A selected invertible scale/shear gauge has an exact inverse and covariance checks. Zero contraction preserves raw algebra but returns no normalized physical state.

The formulas and examples are independently checked by explicit virtual-path sums and dense physical calculations. The calculation does not solve for a ground state, evolve in time, canonicalize automatically or choose a bond truncation. The next chapter will add Schmidt weights, entropy, optimal low-rank error and the distinction between discarding and renormalizing. DMRG, MPO, long-chain scaling and research convergence claims require their own derivations and evidence.

For context, see [Schollwöck's MPS review](https://arxiv.org/html/1008.3477v2), specifically its gauge discussion, contraction section and canonical-form construction. Our index convention and worked examples are derived explicitly above. Read a diagram by naming its open legs, summed legs, conjugates and normalization before interpreting its colors or motion.
