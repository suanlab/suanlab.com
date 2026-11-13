## 1. A mixed qubit occupies the Bloch ball

A pure qubit lies on the surface of the Bloch sphere. A general qubit density operator can lie inside the ball. Its vector r describes three Pauli means, not a particle position or a hidden classical spin direction. The matrix, its purity and its determinant are linked as follows.

$$
\rho=\frac{I+\mathbf r\cdot\boldsymbol\sigma}{2}
=\frac12\begin{pmatrix}1+r_z&r_x-ir_y\\r_x+ir_y&1-r_z\end{pmatrix},\qquad
|\mathbf r|\le1,\quad
\operatorname{Tr}(\rho^2)=\frac{1+|\mathbf r|^2}{2},\quad
\det\rho=\frac{1-|\mathbf r|^2}{4}.
$$

The ball condition is a positivity condition. Surface points are pure, the center I/2 is maximally mixed, and interior points have rank two. A vector outside the ball gives a negative eigenvalue. Renormalizing a drawn arrow to length one would change a mixed state into a different state rather than correct a plotting issue.

This three-mean description is special to dimension two. For general d, a density operator has d²−1 real parameters. Higher-spin examples with identical means already showed why an ordinary three-dimensional arrow cannot represent every state.

## 2. Resolve the spectrum without inventing an eigenbasis

Since (r·σ)²=|r|²I, the eigenvalues follow from a quadratic identity. Write s=|r|. For s>0 the projectors select the two directions parallel and antiparallel to r.

$$
\lambda_\pm=\frac{1\pm s}{2},\qquad
\lambda_- =\frac{1-s^2}{2(1+s)},\qquad
P_\pm=\frac{I\pm\mathbf r\cdot\boldsymbol\sigma/s}{2}\quad(s>0),\qquad
\rho=\lambda_-P_-+\lambda_+P_+.
$$

The second expression for λ− follows by multiplying 1−s by 1+s. It avoids subtracting two nearby numbers when the state is almost pure. A tiny positive determinant still means rank two; a displayed eigenvalue rounded to zero is not proof of purity.

At s=0 there is no distinguished direction r/s. The eigenvalue 1/2 has multiplicity two and its full spectral projector is I. Merge that eigenspace before interpreting the result. A numerical routine that returns two arbitrary basis vectors at the center has not found a physically preferred axis.

## 3. Entropy is a function of the spectrum in any dimension

For a general finite density operator, von Neumann entropy is the Shannon entropy of its eigenvalue distribution. It measures a property of the state, not the entropy of one particular preparation label. We use natural logarithms for nats and divide by ln2 for bits.

$$
S(\rho)=-\sum_k\lambda_k\ln\lambda_k,\qquad
S_{\mathrm{bits}}(\rho)=\frac{S(\rho)}{\ln2},\qquad
0\ln0:=\lim_{x\downarrow0}x\ln x=0,\qquad
0\le S(\rho)\le\ln d.
$$

The notation 0ln0 defines the continuous entropy term. It does not define ln0 as a finite number. The experiment reports a zero eigenvalue's logarithm as null and its entropy contribution as exactly zero. A matrix logarithm of a singular state needs its support understood; entropy itself remains well defined through the spectral expression.

Each term is nonnegative for eigenvalues in [0,1], so S=0 exactly for a rank-one state. Concavity of −xlnx gives Σf(λk)≤df(1/d)=lnd, with equality only at I/d. Replacing d by the rank gives a sharper support bound. Unitary conjugation preserves eigenvalues and therefore entropy.

## 4. Preparation entropy, measurement entropy and state entropy differ

An ensemble's label probabilities pᵢ have classical entropy H(p). If every label names the same ray, ρ is pure even when H(p)>0. For orthogonal occupied pure states, the probabilities are eigenvalues and H(p)=S(ρ). For a nonorthogonal pure-state ensemble, S(ρ)≤H(p).

The Gram construction explains the inequality: ρ and G have the same nonzero spectrum, while the diagonal of G is p. Diagonal pinching cannot reduce spectral entropy, as justified in the following section. A label record can thus contain information that is absent from the system state.

Let h(x)=−xlnx−(1−x)ln(1−x). The final column describes the entropy of Z-basis outcomes, which can exceed the state's entropy.

| State | Purity | State entropy, nats | Z outcome entropy, nats |
| --- | --- | --- | --- |
| e₀ | 1 | 0 | 0 |
| Positive X pure state | 1 | 0 | ln2 |
| I/2 | 1/2 | ln2 | ln2 |
| Unequal mixture from the preceding chapter | 19/25 | h((5−√13)/10) | h(3/10) |

A random Z outcome from a pure X state is not evidence that the state is mixed. Conversely, the same purity does not identify a state: purity is one spectral moment, and many matrices share it. In dimensions above two, even the full entropy is only one summary of a multi-parameter spectrum.

## 5. Basis-relative coherence compares two entropies

A passive basis change transforms the coordinate matrix by U†ρU. The physical state is unchanged, but which entries are diagonal changes. Define Δbasis as complete dephasing in the specified basis.

$$
\rho_{\mathrm{basis}}=U^\dagger\rho U,\qquad
C_{\mathrm{basis}}(\rho)=S(\Delta_{\mathrm{basis}}\rho)-S(\rho)
=D(\rho\Vert\Delta_{\mathrm{basis}}\rho)\ge0.
$$

To see nonnegativity in finite dimension, diagonalize ρ in another basis. The probabilities on the chosen diagonal are qᵢ=Σk|Uik|²λk. The matrix of squared unitary magnitudes has nonnegative entries and row and column sums one. Apply concavity of −xlnx in each row, then sum over rows; H(q)≥H(λ). This also justifies the diagonal-pinching entropy claim used for G.

The relative-entropy identity follows because ln(Δρ) is diagonal: its trace against ρ depends only on ρ's diagonal. If a diagonal entry vanishes, positivity forces the corresponding row and column of ρ to vanish, so support causes no infinite contribution here. This reasoning is specific to the state and its own dephased version.

A pure positive X state has Z-basis coherence ln2 but zero coherence in its X eigenbasis. Its state entropy is zero in either coordinate system. Coherence is therefore not another name for mixedness. The experiment computes the transformed matrix and its probabilities explicitly, while preserving purity and spectral quantities.

## 6. Partial trace describes a subsystem without choosing a hidden label

A second reason to use density operators is restricted access to a composite system. In an orthonormal product basis, the partial trace sums a matched index of the inaccessible subsystem. It is characterized by agreement of every local observable.

$$
(\rho_A)_{ij}=\sum_\mu(\rho_{AB})_{i\mu,j\mu},\qquad
\operatorname{Tr}[\rho_{AB}(M_A\otimes I_B)]
=\operatorname{Tr}(\rho_A M_A).
$$

For the Bell vector (e₀⊗e₀+e₁⊗e₁)/√2, use the product-basis order 00,01,10,11. The global matrix and the reduced matrix are concrete hand calculations.

$$
\rho_{AB}=\frac12\begin{pmatrix}1&0&0&1\\0&0&0&0\\0&0&0&0\\1&0&0&1\end{pmatrix},\qquad
\rho_A=\operatorname{Tr}_B\rho_{AB}=\frac12\begin{pmatrix}1&0\\0&1\end{pmatrix}.
$$

Cross terms vanish because the traced labels are orthogonal. The global state has purity one and entropy zero; A alone has purity 1/2 and entropy ln2. This is not a unitary basis change on A. Removing access to B is a different operation.

The classical joint mixture of 00 and11 also has reduced state I/2 but global entropy ln2. Equal marginals do not identify the joint state or establish entanglement. These are analytic composite-system examples; the current single-qubit calculator does not accept a joint register. Tensor products and entanglement receive their own subsequent experiments.

## 7. Purification connects local mixing with a larger pure state

Given a spectral decomposition, append orthogonal environment labels and use square roots of the eigenvalues as amplitudes. Tracing those labels removes cross terms and recovers the original density.

$$
|\Psi\rangle_{AE}=\sum_{k:\lambda_k>0}\sqrt{\lambda_k}\,|k\rangle_A\otimes|k\rangle_E,\qquad
\operatorname{Tr}_E|\Psi\rangle\langle\Psi|=\rho_A.
$$

The state is normalized because Σλk=1. An environment of dimension at least rankρ suffices for this construction; a smaller one cannot supply the required Schmidt rank. A unitary acting only on the environment produces another purification with the same reduced state.

An ensemble gives another construction, Σᵢ√pᵢψᵢ⊗eᵢ, using orthogonal label states even when the ψᵢ are not orthogonal. Discarding the label again yields ρ. This relates the weighted-column picture to purification without saying that a particular environment or label interpretation is recoverable from local data.

A locally mixed state can arise from a classical selection or from a subsystem of an entangled pure state. Local measurements alone do not reveal which description produced the same ρ. General Schmidt decomposition, ensemble transformations and joint-system dynamics extend this reasoning later; this chapter supplies the definitions and worked local example.

## 8. A dephasing channel preserves populations while changing coherence

Now consider an operation on a qubit, rather than a basis change. Mix the identity action with conjugation by Pauli Z. The signed parameter η and the phase-flip probability are related but are not the same input.

$$
\mathcal E_\eta(\rho)=\frac{1+\eta}{2}\rho+\frac{1-\eta}{2}Z\rho Z,\quad -1\le\eta\le1,\qquad
K_0=\sqrt{\frac{1+\eta}{2}}I,\quad K_1=\sqrt{\frac{1-\eta}{2}}Z,\qquad
(r_x,r_y,r_z)\mapsto(\eta r_x,\eta r_y,r_z).
$$

Both mixture weights are nonnegative, including for negative η. The Kraus sum reproduces the operation and ΣK†K=I proves trace preservation. Here ΣKK†=I also holds, so the operation is unital. The Z populations are unchanged; their relative off-diagonal amplitude is scaled by η.

At η=1 the channel is identity, at η=0 it completely removes Z-basis coherence, and at η=−1 it is the unitary phase flip Z. Negative η is not a negative probability. A diagonal state can remain unchanged under all these channels, so one invariant input does not establish that the entire operation is identity.

## 9. Check complete positivity using the whole channel

An operation should work on all relevant inputs, including when a system is correlated with an external one. Trace preservation on one tested density is insufficient. The finite family supplies its Choi certificate with an explicit convention: output⊗input and index 2·output+input.

$$
J(\mathcal E_\eta)=\begin{pmatrix}1&0&0&\eta\\0&0&0&0\\0&0&0&0\\\eta&0&0&1\end{pmatrix},\qquad
\operatorname{spec}J=\{0,0,1-\eta,1+\eta\},\qquad
\operatorname{Tr}_{\mathrm{output}}J=I,\quad\operatorname{Tr}J=2.
$$

The eigenvalues are nonnegative exactly on the allowed interval. This is an unnormalized Choi matrix, not a trace-one state. Its positive rank is one at the two unitary endpoints and two otherwise, agreeing with the minimal number of Kraus operators. At η=0 the eigenvalue one is itself degenerate.

The Choi input/output indices certify a map; they do not mean that the experiment accepts an arbitrary two-qubit preparation. General channel representations, extensions and entanglement are broader topics. Here the certificate prevents the particular error of inferring an operator property from normalization on one state.

## 10. A signed parameter path is not a physical clock

The experiment starts from η=1 and interpolates to a chosen endpoint. Its parameter u labels a family of channels applied to the same initial state, not successive measurements or consecutive steps of a time law.

$$
\eta(u)=1+(\eta_{\mathrm{end}}-1)u,\quad0\le u\le1,\qquad
u_0=\frac1{1-\eta_{\mathrm{end}}}\quad(\eta_{\mathrm{end}}<0).
$$

The exact η=0 point is inserted if a display grid misses it. For a positive X initial state, the following values follow directly from its two eigenvalues. Here h₂ is binary entropy in bits and h₂(1/4)≈0.811278.

| η | Bloch rx | Purity | Entropy, bits | Squared fidelity to initial | Trace distance to initial |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 0 | 1 | 0 |
| 1/2 | 1/2 | 5/8 | h₂(1/4) | 3/4 | 1/4 |
| 0 | 0 | 1/2 | 1 | 1/2 | 1/2 |
| −1/2 | −1/2 | 5/8 | h₂(1/4) | 1/4 | 3/4 |
| −1 | −1 | 1 | 0 | 0 | 1 |

Equal absolute η gives equal spectra, but the states can differ. Crossing zero increases and then decreases entropy for a state with transverse coherence. Each individual channel still has output entropy at least that of the chosen initial state, because this is a unital random-unitary family. Those two statements compare different pairs of states. No Markovian time law is supplied by this interpolation.

## 11. State distance and fidelity answer questions entropy cannot

Entropy depends only on eigenvalues. To compare particular states, use quantities sensitive to their relative geometry. Our fidelity convention is squared Uhlmann fidelity, explicitly distinguished from its square root.

$$
T(\rho,\sigma)=\frac12\|\rho-\sigma\|_1=\frac12|\mathbf r-\mathbf s|,\qquad
F_{\mathrm{sq}}(\rho,\sigma)=\left[\operatorname{Tr}\sqrt{\sqrt\rho\,\sigma\sqrt\rho}\right]^2
=\operatorname{Tr}(\rho\sigma)+2\sqrt{\det\rho\det\sigma},\qquad
\|\rho-\sigma\|_{\mathrm{HS}}^2=\frac12|\mathbf r-\mathbf s|^2.
$$

The Bloch-vector equalities and determinant formula apply to qubits. The trace-norm and matrix-square-root definitions extend to general dimension. The determinant formula follows because the square of the sum of two square-root eigenvalues equals their sum plus twice the square root of their product.

For identical I/2 states, Tr(ρσ)=1/2 but squared fidelity is one. The Hilbert–Schmidt overlap alone is therefore not mixed-state fidelity. For orthogonal pure positive and negative X states, both entropies are zero, while trace distance is one and squared fidelity zero. A common sign on a pure-state vector leaves its density unchanged and has distance zero instead.

## 12. Read numerical boundaries without changing the state

The eleven bounded integer fields specify rational Bloch coordinates, a nonzero raw probe, signed η and one to sixteen sampling intervals. Positivity is checked exactly before calculation. At a tiny positive radius the two eigenspaces remain distinct; only an exactly zero radius merges them. At a tiny positive determinant the smaller eigenvalue remains positive.

Density entries, purity, determinant, Pauli statistics, passive-basis coordinates and Choi accounts are exact rationals. Spectral projectors, entropy, distances and Kraus coordinates use directed arithmetic. The work precision is 768 bits and the outward reported precision is 512 bits; logarithms use 800-bit work and 768-bit internal bounds. Neither near-zero classification nor output normalization is substituted for a certificate.

The entropy calculation branches on exact zero before logarithms and uses monotonic log bounds for positive intervals. Unresolved numerical observations can remain null. That status differs from the null logarithm of a proven zero eigenvalue. The independent reference uses generic eigensolvers, matrix square roots and singular values rather than the production scalar formulas.

## Research extension: valid channels need not form a divisible time evolution

### A family of physical maps needs an additional composition test

The reader's u parameter is not a clock. To define a new time-dependent model, explicitly choose η(t), start with η(0)=1 and compare the entire family E_η(t). Composition of the existing dephasing maps is simple:

$$
\mathcal E_a\circ\mathcal E_b=\mathcal E_{ab},\qquad
 \mathcal E_1=\operatorname{id},\qquad \mathcal E_0(\rho)=\operatorname{diag}(\rho).
$$

CP-divisibility asks whether, for every t≥s, there exists a completely positive trace-preserving map V_t,s such that E_η(t)=V_t,s E_η(s). This is a statement about an intermediate map on every input, not the survival of one chosen state. [Rivas, Huelga and Plenio](https://arxiv.org/pdf/0911.4270) discuss this propagator composition criterion.

When η(s) is nonzero, E_η(s) is invertible as a linear map on matrices. The intermediate map is therefore uniquely fixed. The earlier Choi calculation gives its complete-positivity condition:

$$
\eta(s)\ne0:\quad V_{t,s}=\mathcal E_{\kappa},\quad
 \kappa=\frac{\eta(t)}{\eta(s)},\quad
 \operatorname{spec}J(V_{t,s})=\{0,0,1-\kappa,1+\kappa\},\quad
 V_{t,s}\ \text{CPTP}\ \Longleftrightarrow\ |\kappa|\le1.
$$

The inverse of a channel need not itself be a channel. In particular, increasing |η| requires a transverse amplification with a negative intermediate Choi eigenvalue. Validity of both endpoint maps does not rescue that intermediate map.

### What if coherence has already vanished exactly?

At η(s)=0 the inverse formula is unavailable. Instead compare the two initial X eigenstates:

$$
\eta(s)=0:\quad[\exists V_{t,s}\ \text{CPTP}:\ \mathcal E_{\eta(t)}=V_{t,s}\mathcal E_{\eta(s)}]
 \Longleftrightarrow\eta(t)=0,\qquad
 \mathcal E_0(\rho_{+x})=\mathcal E_0(\rho_{-x})=I/2.
$$

If η(t)≠0, the later outputs differ although both earlier inputs to V_t,s were I/2. No single linear map could do that. If η(t)=0, the identity intermediate map suffices. This argument treats the noninvertible point directly rather than dividing by a small epsilon.

A useful operational witness is the trace distance of that pair:

$$
T(t)=\tfrac12\|\mathcal E_{\eta(t)}(\rho_{+x})-\mathcal E_{\eta(t)}(\rho_{-x})\|_1=|\eta(t)|,\qquad
 T(V\rho,V\sigma)\le T(\rho,\sigma)\quad(V\ \text{CPTP}).
$$

For any CPTP map the inequality follows by optimizing the difference of an outcome probability over effects0≤M≤I: its adjoint maps such effects back into that same interval. Thus an increase of T forbids CPTP postprocessing of the earlier outputs. [Breuer, Laine and Piilo](https://arxiv.org/pdf/0908.0238) develop this distinguishability perspective. The equivalence with the scalar |η| test here belongs to this particular dephasing family; one pair does not characterize every general quantum process.

### A nonnegative rate produces a simple time law

On a differentiable interval where η is nonzero, differentiating the off-diagonal entries gives:

$$
\dot\rho=\gamma(t)(Z\rho Z-\rho),\quad
 \gamma(t)=-\frac{\dot\eta(t)}{2\eta(t)}\quad(\eta(t)\ne0),\qquad
 \eta(t)=e^{-2\gamma t}\quad(\gamma\ge0\ \text{constant},\ \eta(0)=1).
$$

The factor2 comes from ZρZ−ρ, which multiplies an off-diagonal entry by−2. Rate γ has inverse-time units. A constant nonnegative rate produces a semigroup with decreasing positive η. A time-dependent nonnegative rate similarly gives decreasing |η| on an invertible interval. The local rate formula cannot be carried through a zero without a separate analysis.

### Static random frequencies give a physical revival example

Choose the sign of a frequency once at preparation and keep it fixed during the run. Average the two corresponding unitary rotations:

$$
U_\pm(t)=e^{\mp i\Omega tZ/2},\qquad
 \mathcal E_t(\rho)=\tfrac12 U_+(t)\rho U_+(t)^\dagger+\tfrac12 U_-(t)\rho U_-(t)^\dagger,
 \qquad\eta(t)=\cos(\Omega t),\quad\Omega>0.
$$

Each time map is a convex mixture of unitaries, so it is CPTP. It is a consistent time-dependent experiment with correlated noise across intervals, not a claim that the existing interpolation slider is physical time. Averaging afresh after every short segment would instead resample the sign and define another process.

The formal rate and two landmarks are:

$$
\gamma(t)=\frac\Omega2\tan(\Omega t)\quad(\cos(\Omega t)\ne0),\qquad
 \eta(\pi/(2\Omega))=0,\quad\eta(\pi/\Omega)=-1,\quad
 T(\pi/(2\Omega))=0,\quad T(\pi/\Omega)=1.
$$

Between the zero and the later negative endpoint, |η| increases. The rate is negative wherever defined on that interval, and no CPTP intermediate map can restore the erased distinction. The singular rate does not make the finite channel at the zero ill-defined. At the endpoint η=−1 the individual map is simply the valid unitary Z channel.

For the initial +x state, the later +x test and spectral entropy are:

$$
p_{+x}(t)=\tfrac12(1+\eta(t)),\quad
 S_{\rm bits}(t)=h_2\!\left(\frac{1+|\eta(t)|}{2}\right),\quad
 h_2(p)=-p\log_2p-(1-p)\log_2(1-p),\quad 0\log_2 0:=0.
$$

Use q=tan(Ωt/2), so η=(1−q²)/(1+q²), on0≤Ωt<π. The next code uses exact fractions for η and probabilities, with rounded entropy in bits. It adds no time-rate or noise-environment control to the current dephasing reader.

```python
from fractions import Fraction as Q
from math import log2
for q in [Q(0), Q(1, 2), Q(1), Q(2)]:
    eta = (1-q*q)/(1+q*q)  # q = tan(Omega*t/2), 0 <= Omega*t < pi
    distance, probability = abs(eta), (1+eta)/2
    eigenvalues = [(1+distance)/2, (1-distance)/2]
    entropy = sum(-float(p)*log2(float(p)) for p in eigenvalues if 0 < p < 1)
    print(q, eta, distance, probability, f"{entropy:.6f}")
```

| q | eta | distance | p_plus_x | entropy bits |
|---|---|---|---|---|
| 0 | 1 | 1 | 1 | 0.000000 |
| 1/2 | 3/5 | 3/5 | 4/5 | 0.721928 |
| 1 | 0 | 0 | 1/2 | 1.000000 |
| 2 | -3/5 | 3/5 | 1/5 | 0.721928 |

![A static-frequency mixture has eta equal to cos of time: distinguishability decreases to zero and then revives, unlike exponential semigroup dephasing; the entropy of one probe follows a different curve.](/assets/learning/quantum-density-research-dephasing-divisibility.en.svg)

The exponential comparison uses γ=Ω/2. One probe's entropy alone is not a complete map diagnostic. A diagonal initial state would remain unchanged throughout both families. Here the two-state distance and intermediate-map calculation establish the relevant distinction. A revival does not violate trace-distance contraction of an individual CPTP map: it prevents the later map from being a CPTP continuation of the already reduced earlier system.

#### Research check 1: compose two endpoints

E_(1/2) followed by E_(−1/2) gives E_(−1/4), which is CPTP. But reaching E_(−1/2) from E_(1/4) would require the unique intermediate E_(−2), whose Choi matrix has a negative eigenvalue. Check the ratio, not only the later endpoint.

#### Research check 2: try to continue after complete erasure

At Ωs=π/2 both initial X eigenstates have become I/2. At Ωt=π they become distinct again. A map applied only to the earlier reduced system receives the same input in both cases, so it cannot reproduce both later outputs. Correlated information outside that reduced description matters.

#### Research check 3: a constant diagonal probe

For an initial Z eigenstate, every E_η leaves the state pure and unchanged. Its entropy and fidelity curves would hide the entire transverse effect. Constancy of this one preparation cannot certify divisibility or identity of the process.

## 13. Exercises with explanations

**1. Why use the alternate formula for the smaller eigenvalue?**

**Explanation.** It avoids cancellation in 1−s near purity. The exact numerator 1−s² retains positivity, and the denominator stays away from zero. It does not turn an almost-pure state into a pure one.

**2. Is assigning zero entropy to a zero eigenvalue an evaluation of ln0?**

**Explanation.** No. The continuous term xlnx has limit zero, while ln0 remains undefined. The calculation keeps a null log and a structurally zero contribution.

**3. Why is Choi trace two when state trace must be one?**

**Explanation.** We use the unnormalized channel convention. Its output partial trace is I on a two-dimensional input space, so the full trace is two. Dividing it by two would change the stated certificate convention.

**4. Is η=−1 an invalid dephasing input?**

**Explanation.** No. Its phase-flip probability is one and its identity weight zero. It is unitary Z, which sends positive X to negative X while preserving entropy.

**5. Can two zero-entropy states be perfectly distinguishable?**

**Explanation.** Yes: the two X eigenstates are orthogonal. Their trace distance is one and squared fidelity zero. Entropy alone does not compare their relative directions.

**6. What are the Bell state's global and reduced purities?**

**Explanation.** The global projector has purity one; either reduced state I/2 has purity 1/2. Equal reduced matrices do not determine whether a different joint state was classically correlated or entangled.

**7. What is the coherence of a pure positive X state in its own eigenbasis?**

**Explanation.** Zero. Its dephased matrix in that basis is already pure and diagonal. In the Z basis the coherence is ln2, while the state's entropy remains zero in both bases.

**8. Must the preparation-label entropy equal the density entropy?**

**Explanation.** No. Repeated copies of one ray can have many random labels but density entropy zero. Equality holds for an ensemble of occupied orthogonal pure states; nonorthogonal ensembles can have strictly larger label entropy.

## 14. Research connections and the limits of the example

Spectral entropy, positive matrix parameterizations, distinguishability and basis-aware coherence connect to regularization, matrix learning and uncertainty summaries. Their meanings must be specified before using them as interchangeable objectives. A low entropy does not identify a target state, and a high purity does not certify a learned channel's action on other inputs.

Entropy need not increase under every quantum channel. An analytic reset example sends every input to e₀e₀†:

$$
K_0=|0\rangle\langle0|,\qquad K_1=|0\rangle\langle1|,\qquad
\sum_aK_a^\dagger K_a=I,\qquad
\sum_aK_a\rho K_a^\dagger=|0\rangle\langle0|.
$$

It reduces I/2 entropy from ln2 to zero and is not unital. This additional hand example is not another channel offered by the dephasing engine. General channels, entropy inequalities, finite-shot estimation, composite dynamics and research depth continue later. Open the textbook through **Book → Interactive Book**.

Background links are [Preskill Chapter2](https://www.preskill.caltech.edu/ph219/chap2_15.pdf), [Chapter3](https://www.preskill.caltech.edu/ph229/notes/chap3.pdf), [Chapter10 §10.2](https://www.preskill.caltech.edu/ph219/chap10_6A_2022.pdf) and [IBM channel representations](https://quantum.cloud.ibm.com/learning/en/courses/general-formulation-of-quantum-information/quantum-channels/representations-of-channels). The worked states, partial-trace calculation, signed path and numerical contracts are developed here independently for this textbook.
