## 1. Ask which modes are occupied instead of naming particles

In the previous chapter, spin sites represented distinguishable degrees of freedom. For identical fermions, labels saying that particle1 is here and particle2 is there cannot serve as physical identifying information. Instead, choose a set of **one-particle modes** and ask which are occupied. A mode might be a spatial orbital, a momentum state or a spin-orbital. Distinguishing a mode label from a particle label is the starting point of the Fock representation.

Suppose the mode functions phi0 and phi1 are orthogonal. A state with one identical fermion in each mode has an antisymmetric two-particle wavefunction. Here x denotes the **complete one-particle coordinate**, including spin when needed. We are not claiming that the spatial part alone is always antisymmetric.

$$
\Psi(x_1,x_2)=\frac{\phi_0(x_1)\phi_1(x_2)-\phi_1(x_1)\phi_0(x_2)}{\sqrt2},\qquad
\Psi(x_2,x_1)=-\Psi(x_1,x_2).
$$

Rather than repeatedly writing this exchange sign in a long wavefunction, we can place it in the algebra of creation and annihilation operators. That is a more consequential change than simply compressing a bit string. An electron's two spin values and the0/1 occupancy of a fermion mode refer to different degrees of freedom. If the up and down spin-orbitals of one spatial orbital are separate modes, both modes may be occupied simultaneously.

Our route is **modes → Fock basis → operator order and signs → Hamiltonian → conservation and physical assumptions → limits of observables**. We assume complex inner products and Pauli tensor products, without fixing the particle number in advance. Calculations use at most four modes; this is neither a bound on the theory nor a limit on the material to learn.

## 2. The vacuum is different from the zero vector

Assume that the chosen one-particle modes are orthonormal. The normalized vacuum has every mode empty. The creator c_i† occupies mode i, and the annihilator c_i empties it. Fix the order of modes as0,1,… and define the following basis. The rightmost operator still acts first.

$$
|n_0\cdots n_{m-1}\rangle=(c_0^\dagger)^{n_0}\cdots(c_{m-1}^\dagger)^{n_{m-1}}|\mathrm{vac}\rangle,\qquad
n_i\in\{0,1\},\qquad \dim\mathcal F=2^m.
$$

For two modes, |11〉=c₀†c₁†|vac〉. This instructs us to apply c₁† first and c₀† second. It does not attach physical time-order labels to the particles. The chosen operator-product convention determines the signs of coordinates.

The fixed-particle-number N subspace has dimension equal to the number of ways of choosing N occupied modes from m. The full Fock space includes all of these number subspaces. The vacuum has particle number0 but norm1. In contrast, the result0 in c_i|vac〉=0 is a zero vector and cannot be normalized into another state. Replacing that result by the vacuum would change the operator algebra.

## 3. Anticommutation produces exclusion and order signs

Fermion creators and annihilators satisfy the canonical anticommutation relations, or CAR. Braces denote the **sum** of two products, rather than the difference used for a commutator. These are matrix identities covering both equal and distinct modes.

$$
\{c_i,c_j^\dagger\}=\delta_{ij}I,\qquad
\{c_i,c_j\}=\{c_i^\dagger,c_j^\dagger\}=0,\qquad
n_i=c_i^\dagger c_i,\qquad n_i^2=n_i.
$$

Setting i=j in the second relation gives c_i²=0 and a vanishing square for the creator. This is the Pauli restriction against occupying the same mode twice. To calculate n_i², insert c_i c_i†=I−n_i and obtain n_i²=n_i, so occupation eigenvalues are0/1. Distinct number operators commute and admit a joint occupation basis.

Inspect the complete action of the two annihilators below. Basis order is00,01,10,11, with n₀ the most significant bit. Every0 in the table denotes a zero vector.

| Input | Output of c₀ | Output of c₁ |
| --- | --- | --- |
| \|00〉 | 0 | 0 |
| \|01〉 | 0 | \|00〉 |
| \|10〉 | \|00〉 | 0 |
| \|11〉 | \|01〉 | −\|10〉 |

The final minus sign separates this algebra from independent bit ladders. Moving c₁ past the preceding c₀† requires one anticommutation. Reversing the creation order gives the same issue: c₁†c₀†|vac〉=−|11〉. Removing the relative sign is a change of model, not a cosmetic simplification of its display.

## 4. Count the parity of preceding occupations

An annihilator at mode i gives0 if that mode is empty. Otherwise, count the occupied modes preceding it. A creator uses the same prefix sign when its opposite occupation condition permits creation.

$$
c_i|n_0\cdots n_i\cdots\rangle=
\begin{cases}
0,&n_i=0,\\
(-1)^{\sum_{j<i}n_j}|n_0\cdots0_i\cdots\rangle,&n_i=1.
\end{cases}
$$

For example, c₂|011〉=−|010〉 because the sum of the first two occupations is1. Thus c₀†c₂|011〉=−|110〉, whereas c₂c₀†|011〉=+|110〉, and the two actions sum to zero. The sign follows the selected mode order. It does not require a picture of particles physically passing each other along a trajectory.

The Jordan–Wigner representation implements this rule with qubit matrices. The matrix a lowers one bit, while the preceding Z factors transmit occupation parity.

$$
a=\frac{X+iY}{2}=\begin{pmatrix}0&1\\0&0\end{pmatrix},\qquad
c_i=\left(\prod_{j<i}Z_j\right)a_i,\qquad
c_0=a\otimes I,\quad c_1=Z\otimes a.
$$

Dropping the Z string makes distinct-mode matrices commute. Apply the naive anticommutator of annihilator0 and creator1 to |10〉: it produces2|01〉. The correct paths instead give−|01〉 and+|01〉 and cancel. A matrix size of2^m does not establish that the fermion algebra has been implemented. Check both the bit encoding and the correspondence for physical operators.

## 5. Taking an adjoint reverses the order of a product

Use the following even Hamiltonian. The scalar epsilon0 sets an energy origin, mu_i is the onsite **energy coefficient in this chapter**, h describes hopping, Delta describes pairing, and U couples densities. The symbol mu does not automatically imply a particular thermodynamic chemical-potential sign convention. h and Delta may be complex; the remaining coefficients are real.

$$
\begin{aligned}
H={}&\epsilon_0I+\sum_i\mu_i n_i+\sum_{i<j}U_{ij}n_in_j\\
&+\sum_{i<j}\left(h_{ij}c_i^\dagger c_j+h_{ij}^*c_j^\dagger c_i
+\Delta_{ij}c_i^\dagger c_j^\dagger+\Delta_{ij}^*c_jc_i\right).
\end{aligned}
$$

Each term contains an even number of creators or annihilators. Hopping empties one mode and occupies another, preserving particle number. Pairing creates or removes two particles together. A density interaction assigns energy according to joint occupation. Specifying a coupling is different from evolving the prepared state.

In particular, (c_i†c_j†)†=c_jc_i. Writing c_ic_j instead introduces another anticommutation sign. Conjugating a complex coefficient while leaving the operator order unchanged may fail to give a Hermitian Hamiltonian. Apply the general identity (AB)†=B†A† first, and only then use CAR to reorder the operators if required.

General four-mode scattering terms and odd sources are outside this finite experiment's inputs. Effective pairing models can arise through pair exchange with a reservoir or a mean-field description, but they do not by themselves assert a fundamental failure of particle-number conservation in a closed universe. Deciding what is included in the system is part of specifying a physical model.

## 6. Calculate hopping phases and signs across an occupied mode

For two modes, take H=c₀†c₁+c₁†c₀. It exchanges the one-particle basis states |01〉 and |10〉 with coefficient1 and annihilates the vacuum and11. Two states with equal occupation probabilities consequently have opposite energies.

$$
|\psi_\pm\rangle=\frac{|01\rangle\pm|10\rangle}{\sqrt2},\qquad
E_\pm=\pm1,\qquad
H=\frac{X_0X_1+Y_0Y_1}{2}.
$$

The factor1/2 differs from the XX+YY Hamiltonian in the preceding chapter. Seeing the same exchange pattern does not justify copying an energy without checking the coefficient. With h=i, the one-particle block becomes the Y matrix, and the normalized state |01〉+i|10〉 has energy+1. Relative phase again controls the energy.

With three modes, hopping between0 and2 reveals the effect of the intervening occupation. Keep the mode order0,1,2 and define H₀₂=c₀†c₂+c₂†c₀.

$$
c_0^\dagger c_2|011\rangle=-|110\rangle,\qquad
|\chi\rangle=\frac{|011\rangle+|110\rangle}{\sqrt2},\qquad
\langle\chi|H_{02}|\chi\rangle=-1.
$$

A model without parity strings gives+1 on the same coefficient array. The operators differ even though the state labels look identical. A basis redefinition can move signs between representations, but it must transform the state and every Hamiltonian coefficient consistently. Renaming a state cannot justify incorrect CAR.

## 7. Pairing connects even states with different particle numbers

For two modes, T=c₀†c₁† takes the vacuum to11. Its adjoint T†=c₁c₀ takes11 back to the vacuum. In the even block spanned by these two states, the pairing Hamiltonian with real Delta=1 is X.

$$
H_{\rm pair}=T+T^\dagger,\qquad
|e_\pm\rangle=\frac{|00\rangle\pm|11\rangle}{\sqrt2},\qquad
E_\pm=\pm1.
$$

Both states have even parity but superpose particle numbers0 and2. Their mean number is1 and number variance is1. Thus parity can be sharp even when number is not. The energy variance of e_+ is zero, but its energy exceeds that of e_−. Zero variance alone is not a ground-state test. This analytic conclusion for a small block should be separated from a solver for arbitrary off-diagonal inputs.

When Delta=i, the normalized vacuum+i|11〉 state has energy+1. Its term contribution checks conjugation and order through actual complex values.

$$
|e_i\rangle=\frac{|00\rangle+i|11\rangle}{\sqrt2},\qquad
\langle c_0^\dagger c_1^\dagger\rangle=-\frac{i}{2},\qquad
2\operatorname{Re}\!\left(i\langle c_0^\dagger c_1^\dagger\rangle\right)=1.
$$

Some real examples can accidentally give plausible values even when a pair mean is confused with the opposite ordering. Check a complex Delta and a complex state together. Reversing the order when taking T† and conjugating an expectation are separate rules, and each deserves verification.

## 8. Number conservation, parity conservation and sharpness are three questions

Define total number and parity. Since each n_i is0 or1, parity is+1 for even occupation and−1 for odd occupation. Conservation in quantum dynamics concerns a commutator with the entire Hamiltonian, rather than a variance in the current state.

$$
\widehat N=\sum_i n_i,\qquad P=(-1)^{\widehat N},\qquad
[\widehat N,c_i^\dagger c_j^\dagger]=2c_i^\dagger c_j^\dagger,\qquad
[H,P]=0\quad\text{for the even Hamiltonian above}.
$$

| Model / prepared state | Is N conserved? | Current Var(N) | Current parity |
| --- | --- | --- | --- |
| Pairing / vacuum | No | 0 | Even |
| Pairing / e_+ | No | 1 | Even |
| Onsite and hopping / e_+ | Yes | 1 | Even |

The vacuum being an N=0 eigenstate at one instant does not make a pairing Hamiltonian conserve N. Conversely, a number-conserving model can start from a prepared superposition of different even number components. Conservation says that the number distribution is unchanged by Hamiltonian evolution, not that the distribution must contain only one value.

The current Hamiltonian contains only even terms and conserves parity. That fact does not establish that every nonzero input vector is an allowed physical pure state. We next separate a dynamical conservation law from assumptions about physical states and the observable algebra.

## 9. State the parity-superselection assumption

This chapter adopts fermion parity superselection as a physical assumption. Physical density matrices commute with P, and allowed physical observables belong to the parity-even algebra. A pure state must then have definite even or odd parity. This does not fix every state's particle number to one value.

$$
[\rho,P]=0,\qquad
\Pi_e=\frac{I+P}{2},\quad\Pi_o=\frac{I-P}{2},\qquad
\rho=|\psi\rangle\langle\psi|\ \Longrightarrow
\Pi_e|\psi\rangle=|\psi\rangle\ \text{or}\ \Pi_o|\psi\rangle=|\psi\rangle.
$$

A vacuum+11 superposition lies within the even sector and is compatible with this parity condition. The one-mode state (|0〉+|1〉)/sqrt(2), by contrast, contains pure coherence between even and odd sectors and is not presented as an allowed physical pure preparation under the assumption. A block-diagonal mixed density can still contain probabilities for both parities. Do not extend the pure-state definite-parity condition to all mixed states.

The operators c_i and c_i+c_i† are parity-odd. Their means can be computed formally without making them directly allowed observables in this physical model. This distinction avoids labelling arbitrary vectors as physical states before specifying the system, its operations and its observable algebra. Particle-number superselection is an additional assumption one could choose; it is not another name for the parity assumption used here.

## 10. A twirl is a comparison state, not an automatic repair

Starting from a formal density, we can remove even–odd coherence to construct a comparison state. Call this operation the parity twirl, and inspect the original and comparison side by side.

$$
\bar\rho=\frac{\rho+P\rho P}{2}=\Pi_e\rho\Pi_e+\Pi_o\rho\Pi_o,\qquad
\|\rho-\bar\rho\|_F^2=2p_ep_o\quad\text{for a pure input},
\qquad p_e=\langle\Pi_e\rangle,\quad p_o=\langle\Pi_o\rangle.
$$

Split a pure vector into even and odd components. Its two off-diagonal blocks are adjoints, each with squared Frobenius size p_ep_o. The removed coherence therefore has total squared size2p_ep_o. Only a zero weight in one component makes the pure density block-diagonal.

For a formal one-mode plus state, p_e=p_o=1/2 and the squared Frobenius size of the removed coherence is1/2. Original purity is1, while twirled purity is1/2. The mean energy of H=n is1/2 in both states, and the odd mean changes from〈c〉=1/2 to0. Every even observable A satisfies Tr(ρA)=Tr(ρ̄A). Cyclicity of trace and PAP=A give this equality directly. It does not make the two densities the same pure state.

Nor do vanishing first odd means establish [ρ,P]=0. In the three-mode state (|000〉+|111〉)/sqrt(2), the components differ in number by3, so one annihilation cannot connect them and every〈c_i〉vanishes. Even–odd coherence nevertheless remains, with2p_ep_o=1/2. Checking a few zero means is different from checking the physical assumption.

A formal-counterexample mode permits these calculations for learning. It does not delete forbidden amplitudes from a selected even/odd pure sector and renormalize the rest, or silently replace a formal input with its twirl. Preparations with different assumptions should be explicitly represented as different states.

## 11. What one-body and anomalous means retain and miss

Write the index convention before calculating. We use gamma_ij=〈c_i†c_j〉; a definition with the opposite indices in another reference may be its transpose. F is the anomalous mean of two annihilators.

$$
\gamma_{ij}=\langle c_i^\dagger c_j\rangle,\qquad F_{ij}=\langle c_ic_j\rangle,
\qquad \gamma^\dagger=\gamma,\quad F^T=-F,\quad
\langle c_i^\dagger c_j^\dagger\rangle=F_{ji}^*.
$$

Tr(gamma) is the mean particle number. CAR implies positivity of gamma and I−gamma, so one-body occupation eigenvalues lie between0 and1. To see this, form d=Σvᵢcᵢ. Then v†gamma v is〈d†d〉and, using CAR, v†(I−gamma)v is〈dd†〉; both are nonnegative. A correlated state's gamma need not be a projector. The i<j hopping term contributes **2Re(h_ij gamma_ij)**. Writing Tr(h gamma) with those same matrix indices can miss a transpose. In the h=i one-particle example, gamma01=−i/2 gives energy+1.

Even gamma and F together do not determine a general global state or number distribution. Compare these two pure even states of four modes.

$$
|u\rangle=\frac{|0011\rangle+|1100\rangle}{\sqrt2},\qquad
|v\rangle=\frac{|0000\rangle+|1111\rangle}{\sqrt2},\qquad
\gamma_u=\gamma_v=\frac I2,\quad F_u=F_v=0.
$$

| State | Mean N | Var(N) | Number distribution |
| --- | --- | --- | --- |
| u | 2 | 0 | Probability1 at N=2 |
| v | 2 | 4 | Probability1/2 at each of N=0 and4 |

Both preparations have occupation1/2 in every mode and vanishing two-annihilator means, yet their total-number second moments differ. A learned model matching lower-order observables exactly has not thereby reconstructed the full state. Determine separately whether the chosen set of observables is sufficient for the research question.

## 12. Reordering modes also changes subsystem interpretation

Mode order is not a physical law, but it enters the definition of the Fock basis. Relabelling modes changes the signs needed to restore occupied creators to the chosen order. Exchanging the same two occupied modes replaces c₀†c₁† by c₁†c₀† and introduces a minus sign. This is different from merely permuting bit positions.

For a superposition of the vacuum and two occupied modes, the change may appear as a relative sign between components. Transforming the pairing coefficients and the state together preserves physical energy. Transforming only one and observing an energy change does not mean that mode order generated a physical force. Distinguish a coordinate transformation from a physical operation on the state.

Also, qubit coordinates do not automatically make an ordinary qubit partial trace a measure of fermionic mode entanglement. We need an embedding for the mode-subset algebra and a definition of its parity-even observables. This unit supplies the full density and specified operator means, without a general fermion-subsystem entanglement calculator. The ordinary tensor-product results in the later MPS and Schmidt units require this additional structure when applied to fermions.

## 13. Distinguish a full-Fock minimum from a sector minimum

Consider H=−n for one mode. The vacuum is even and has energy0. The occupied state is odd and has energy−1. The space over which we minimize changes the answer.

$$
E_{\min}^{\mathcal F}=-1,\qquad
E_{\min}^{e}=0,\qquad E_{\min}^{o}=-1.
$$

| Comparison space | Minimum energy | Is the vacuum a ground state there? |
| --- | --- | --- |
| Full Fock space | −1 | No |
| Even sector | 0 | Yes |
| Odd sector | −1 | The vacuum is outside this space |

A parity constraint or a fixed-number constraint changes the feasible set of an optimization problem. We cannot calculate a full-space minimum and relabel it as a selected-sector ground result. State whether a projector weight is a probability in the entire input or a conditional probability renormalized within a sector.

The finite experiment enumerates energies and parity/fixed-N minima only when H is diagonal in the occupation basis. A general off-diagonal ground classification remains uncomputed. Separate an analytic solution of the two-mode pairing block from a solver for arbitrary input. A constant Hamiltonian has no next distinct level, so a positive excitation gap is undefined there.

## 14. Check energy residuals together with physical assumptions

Retain a nonzero raw complex vector z and divide expectations by its norm. Variance tests the eigenstate condition for the current H; it does not replace the parity assumption or identify the appropriate ground subspace.

$$
N_z=z^\dagger z>0,\qquad E=\frac{z^\dagger Hz}{N_z},\qquad
\operatorname{Var}(H)=\frac{\|(H-E)z\|^2}{N_z}.
$$

The experiment handles1…4 ordered modes, onsite/density terms, general complex hopping/pairing and pure input states. It checks every CAR, actual N/P commutators, gamma/F and termwise energy accounts, plus formal parity coherence and an explicit twirl comparison. A zero input is rejected as a state, while the zero **output** of Pauli exclusion remains a valid operator result. An output norm ratio is not called a measurement probability without a specified instrument.

For new AI research, matching the algebra, conservation laws, sector and observable set matters more than reusing an occupation table. When approximating states with neural networks or tensors, verify signs, complex phase, number distributions and physical assumptions as well as energy and residuals. Matching one-body training statistics and matching a global fermionic state are different achievements. Small exact matrices supply reference evidence, without automatically proving large-system convergence, complexity or optimization success.

## Research extension: Majorana covariance, Gaussian assumptions and finite zero modes

### Repackage CAR without changing the mode ordering

Use the same ordered annihilators as the original chapter. Their Hermitian combinations and a real antisymmetric covariance are:

$$
m_{2j}=c_j+c_j^\dagger,\quad m_{2j+1}=-i(c_j-c_j^\dagger),\quad
 \{m_a,m_b\}=2\delta_{ab}I,\quad
 \Gamma_{ab}=\frac i2\operatorname{Tr}(\rho[m_a,m_b]),\quad
 \Gamma^{\mathsf T}=-\Gamma\in\mathbb R^{2m\times2m},\quad
 \langle m_am_b\rangle=\delta_{ab}-i\Gamma_{ab},\quad I-i\Gamma\succeq0.
$$

For a complex coefficient vector z, positivity of ⟨(Σz_a m_a)†(Σz_b m_b)⟩ proves the Gram condition. Antisymmetry then bounds covariance singular values by1. Our sign convention gives Γ01=−1 for the one-mode vacuum. Check that example before comparing another convention. The m operators are parity-odd; their bilinear covariances are even and compatible with the stated observable algebra.

A real antisymmetric coefficient matrix encodes a quadratic Hamiltonian:

$$
H=E_cI+\frac i4\sum_{a,b}A_{ab}m_am_b,\quad A^{\mathsf T}=-A\in\mathbb R^{2m\times2m},\qquad
 \langle H\rangle=E_c-\tfrac14\operatorname{Tr}(A\Gamma),\qquad
 \mu(n_j-\tfrac12)=\frac{i\mu}{2}m_{2j}m_{2j+1}.
$$

Sum both index orders; omitting that convention changes a factor of two. Since Γab=i⟨m_a m_b⟩ off the diagonal, direct substitution proves the trace formula. The constant Ec includes any occupation-to-centered-number shifts. The Majorana form and quadratic setting are described in [Kitaev's original paper](https://arxiv.org/pdf/cond-mat/0010440).

The Heisenberg equation closes linearly for any initial density:

$$
\hbar\dot{\mathbf m}=A\mathbf m,\quad O(t)=e^{At/\hbar},\quad O^{\mathsf T}O=I,\qquad
 \Gamma(t)=O(t)\Gamma(0)O(t)^{\mathsf T}.
$$

The matrix exponential is orthogonal because A is real antisymmetric. Expanding CAR in the commutator derives the linear equation, and substituting the transformed operators derives the covariance congruence. This is an exact statement for quadratic dynamics, not a claim that every initial state is Gaussian.

### Two-point reconstruction needs a state-class assumption

An even fermionic Gaussian state obeys Wick factorization: all higher even moments are determined by its two-point data. This includes thermal quadratic states and appropriate limiting pure states. Within that class:

$$
\rho\text{ Gaussian}:\quad
 \rho\text{ pure}\ \Longleftrightarrow\ \Gamma^2=-I,\qquad
 \langle n_in_j\rangle=\gamma_{ii}\gamma_{jj}-|\gamma_{ij}|^2+|F_{ij}|^2\quad(i\ne j),\quad
 \gamma_{ij}=\langle c_i^\dagger c_j\rangle,\ F_{ij}=\langle c_ic_j\rangle.
$$

The occupation identity follows by applying Wick to c_i†c_i c_j†c_j with the signs from CAR. The pure-state covariance criterion is conditional on the Gaussian description; purity of an arbitrary physical state does not imply that equation. See [Bravyi's Gaussian-state definition and Wick relation](https://arxiv.org/pdf/quant-ph/0404180).

A physical even four-mode pure state supplies a direct counterexample to unrestricted reconstruction:

$$
|v\rangle=(|0000\rangle+|1111\rangle)/\sqrt2,\quad\rho_*=I_{16}/16,\quad
 \Gamma_v=\Gamma_*=0,\quad \gamma_v=\gamma_*=I_4/2,\quad F_v=F_*=0,\qquad
 \langle n_0n_1\rangle_v=1/2,\quad\langle n_0n_1\rangle_*=1/4,\quad
 \operatorname{Var}_v(N)=4,\quad\operatorname{Var}_*(N)=1.
$$

The cat's two occupied components differ by four particles, so no quadratic operator connects them. Both states have identical two-point data and every quadratic mean energy agrees, yet a density correlation and the number variance distinguish them. The maximally mixed state is Gaussian; the pure cat violates the displayed Wick prediction1/4 and is not. In particular, Γ=0 does not certify that the source has entropy4 ln2. These are analytic mixed/pure comparisons, not new mixed-input controls in the pure reader.

### Exact odd zero operators do not prove a protected physical qubit

For two modes, couple only the middle Majoranas:

$$
H_e=\frac{it}{2}m_1m_2,\quad t>0,\quad
 [H_e,m_0]=[H_e,m_3]=0,\quad\{P,m_0\}=\{P,m_3\}=0,\qquad
 H_e=-\frac t2(c_0^\dagger c_1+c_1^\dagger c_0+c_0^\dagger c_1^\dagger+c_1c_0),\quad
 \operatorname{spec}H_e=\{-t/2,-t/2,t/2,t/2\}.
$$

The two outer odd operators commute with He because each anticommutes with both factors in its bilinear. They connect opposite-parity energy eigenstates. This explains the twofold mathematical degeneracy, but a coherent superposition of its two parity sectors is not an allowed physical pure preparation under this chapter's superselection assumption. The hopping and pairing signs in the displayed c expansion follow from the original creation-order convention.

A permitted even coupling of the outer pair removes the degeneracy:

$$
H_{e,u}=\frac{it}{2}m_1m_2+\frac{iu}{2}m_0m_3,\quad[H_{e,u},P]=0,\qquad
 E_{s,r}=\tfrac12(st+ru),\quad s,r\in\{-1,+1\},\quad
 E_0=-\tfrac12(t+u),\quad\Delta_{\rm ord}=u\quad(0\le u\le t).
$$

The two bilinears commute and have independently selectable eigenvalues±1 after multiplication by i. Both sectors are still preserved by P. For u>0 there is a unique full-Fock ground state, even though parity conservation remains exact. Thus parity conservation alone does not prohibit this splitting. In a longer wire, locality and spatial separation become essential additional conditions; the two-mode example establishes no robust topological phase or exponential protection law.

The exact table sets t=1 and keeps u in[0,1]. Its ordered gap includes eigenvalue multiplicity:


```python
from fractions import Fraction as Q
for ratio in [Q(0), Q(1, 4), Q(1, 2), Q(1)]:
    energies = sorted((s+r*ratio)/2 for s in [-1, 1] for r in [-1, 1])
    gap = energies[1]-energies[0]  # ordered, with multiplicity
    multiplicity = energies.count(energies[0])
    zero_majoranas = 2 if ratio == 0 else 0
    assert gap == ratio
    print(ratio, energies[0], gap, zero_majoranas, multiplicity)
```

| outer coupling u/t | ground energy/t | ordered first gap/t | commuting linear Majoranas | ground multiplicity |
|---|---|---|---|---|
| 0 | -1/2 | 0 | 2 | 2 |
| 1/4 | -5/8 | 1/4 | 0 | 1 |
| 1/2 | -3/4 | 1/2 | 0 | 1 |
| 1 | -1 | 1 | 0 | 1 |

Finally, density interactions leave the quadratic class:

$$
H_{\rm int}=U n_0n_1:\qquad
 \hbar\dot m_0=i[H_{\rm int},m_0]=U n_1m_1.
$$

This contains a number operator times a Majorana, not a constant-coefficient linear combination of the m vector. A closed covariance-only evolution rule therefore needs additional structure or approximation in an interacting model. A learned two-point simulator should state this restriction instead of inferring closure from a few matching initial moments.

![A four-mode pure cat and maximally mixed state share zero Majorana covariance but different density correlations; a permitted even outer coupling splits the exact two-mode Majorana degeneracy.](/assets/learning/quantum-many-body-research-majorana-covariance-limits.en.svg)

The table is exact. Quadratic covariance evolution and the two-mode spectral calculation are standalone derivations. The current reader keeps its general complex hopping/pairing, density interactions and parity tests; it has no new Gaussian reconstruction, time evolution or arbitrary off-diagonal ground solver.

#### Research check 1: a single-mode sign check

For H=μ(n−1/2), A01=μ and vacuum Γ01=−1. The trace formula gives−μ/2. An occupied mode gives+μ/2. Reversing the covariance convention while leaving the energy formula untouched would reverse these predictions.

#### Research check 2: covariance matching is not a Gaussianity test

For the four-mode cat, gamma=I/2 and F=0 would predict ⟨n0n1⟩=1/4 under Wick. The actual value is1/2. The missing information is in higher moments, not a failure of CAR or parity. Quadratic time evolution still propagates its covariance exactly while leaving this reconstruction gap unresolved.

#### Research check 3: a split state still conserves parity

At t=1,u=1/4 the energies are−5/8,−3/8,3/8,5/8. Their first separation is1/4 and the ground is unique. The Hamiltonian remains even. Exact degeneracy at u=0 and the existence of odd zero operators do not imply immunity to every even perturbation or permission to ignore parity sectors.

## 15. Eight problems about order and assumptions

**1. Vacuum and exclusion.** Compare c₀†c₁†|vac〉, c₁†c₀†|vac〉and (c₀†)²|vac〉. Is the final result a normalized vacuum?

**Explanation.** The results are|11〉,−|11〉and0 in the chosen basis. The final0 is a zero-norm vector, unlike the unit-norm vacuum. It records that a second creator cannot occupy the same mode again.

**2. Actual CAR paths.** Apply c₀c₁† and c₁†c₀ separately to |10〉. Compare the naive model with no parity string.

**Explanation.** The first path creates−|11〉and then annihilates to−|01〉. The second annihilates to the vacuum and then creates+|01〉. Their sum is zero. Both naive paths give+|01〉, producing2|01〉from the anticommutator.

**3. Long-range hopping energy.** Derive the energy−1 of chi under H₀₂ in section6. Also determine current number and parity.

**Explanation.** Each transition between the two occupied basis components has a minus sign, so H₀₂chi=−chi. Energy variance is zero. Both components have two particles: N=2, Var(N)=0 and even parity. This hand calculation does not assert that a general off-diagonal spectrum has been supplied.

**4. Complex pairing adjoint.** For Delta=i and e_i, calculate the creation-pair mean and pairing energy. Why must the adjoint reverse operator order?

**Explanation.** T moves the vacuum amplitude into11, giving〈T〉=conj(i)/2=−i/2. Therefore2Re(i〈T〉)=1. Its adjoint is c₁c₀, whereas c₀c₁=−c₁c₀. Missing this minus sign assembles the wrong Hermitian-conjugate term.

**5. Conservation and sharpness.** Why does zero number variance in the vacuum not imply number conservation under H_pair? What are the parity and number variance of e_+?

**Explanation.** The vacuum is an N=0 eigenstate, but H_pair can send it into the two-particle component, so its commutator with N is nonzero. The state e_+ is an even-parity eigenstate with mean number1 and number variance1. Number conservation, parity conservation and a sharp current distribution are different questions.

**6. Comparing formal coherence.** Compare purity, mean H=n and〈c〉for a one-mode plus state and its parity twirl. Does the three-mode cat satisfy parity superselection because all〈c_i〉vanish?

**Explanation.** Purities are1 and1/2; both energies are1/2; odd means are1/2 and0. Twirling gives a different mixed state. The three-mode cat retains squared even–odd coherence1/2 despite zero first odd means, so it fails the assumption. These formal odd means are not being promoted to allowed physical observables.

**7. Equal gamma and F.** Calculate the second number moment and variance for u and v. Why is one-body reconstruction different from reconstructing the full state?

**Explanation.** State u has definite N=2, second moment4 and variance0. State v has equal probabilities at N=0 and4, second moment8 and variance4. Both have mean2, gamma=I/2 and F=0, yet higher occupation statistics differ. The specified means do not determine every observable or the state.

**8. Which ground state was found?** The vacuum has zero variance under H=−n. Classify it in the full Fock space and in the even sector.

**Explanation.** Energy0 and variance0 establish an eigenstate. An odd occupied state has energy−1, so the vacuum is not a full-space ground state. The even sector contains only the vacuum, making it the sector ground state. The feasible space determines the conclusion.

## 16. Carry the sign algebra into the next representation

When explaining a result, first identify the modes, the creation-order convention and the state's sector. Then distinguish operator-path signs, complex adjoints, N/P commutators, input parity assumptions and the scope of observables. Identical bit arrays, energies or gamma matrices do not establish that two calculations address the same physical problem.

The following boson unit removes the0/1 restriction and develops square-root ladder factors and cutoffs of an infinite space. MPS and Schmidt units then address contraction, gauge freedom and truncation. Applying tensor networks to fermions requires retaining the ordering, algebra and physical-observable structure developed here. A faster representation does not remove these conditions.

For further study, consult the [official Jordan–Wigner explanation](https://quantumai.google/openfermion/tutorials/jordan_wigner_and_bravyi_kitaev_transforms) and [Szalay and colleagues' fermionic-information review](https://arxiv.org/html/2006.03087v4). General subsystem embeddings, fermionic partial traces, allowed operations and entanglement require additional definitions and verification beyond this introductory algebra.
