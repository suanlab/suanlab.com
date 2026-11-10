## 1. Why combining small degrees of freedom becomes difficult

The preceding unit compared energies and variational states for a given matrix. We now ask where that matrix comes from. A single spin one-half needs two complex amplitudes, but a general collection of spins cannot be specified by listing two amplitudes at each site independently. Coherent correlations can connect the outcomes at different sites. The difficulty of a many-body system begins with the number of possible **joint states**, rather than with the size of an individual component.

Think first of four light switches. There are sixteen on/off configurations. A quantum state does more than select one configuration: it assigns a complex amplitude to every configuration. This analogy explains the count of basis states; it does not turn a spin into a classical switch. Measurements along X and Y use different bases, and relative phases affect their outcomes.

Our route is **space → basis and local action → interactions → energy and residual → correlations and symmetry → verification**. We assume complex inner products, tensor products, Pauli matrices and energy variance. Examples use at most four spins so that hand calculations can be checked against full matrices. The theory is not restricted to four particles. Understanding the small system gives us a foundation for asking when an MPS can compress its representation.

## 2. Fix the tensor product and basis order first

Let the space at site i be H_i=C², with |0〉 and |1〉 the eigenstates of Z. The labels0/1 mean Z eigenvalues+1/−1. The full space is the tensor product of these local spaces. Products of local states are special members of that full space.

$$
\mathcal H=\bigotimes_{i=0}^{n-1}\mathbb C^2,\qquad
\dim\mathcal H=2^n,\qquad
|b_0\cdots b_{n-1}\rangle=|b_0\rangle\otimes\cdots\otimes|b_{n-1}\rangle.
$$

We use b₀ as the most significant bit. The coordinate order for two spins is |00〉,|01〉,|10〉,|11〉. For three spins, |101〉 occupies integer index5. Some libraries use the reverse endian convention. Before comparing answers, establish that a state name refers to the same coordinate in both calculations. A different basis order is valid, provided the state and every operator are transformed together.

A general state has2ⁿ amplitudes, whereas a product state can be generated from two amplitudes at each site. For instance, the product (a|0〉+b|1〉)⊗(c|0〉+d|1〉) has coefficients ac,ad,bc,bd. Four arbitrary complex numbers do not always factor this way. For two sites, the rank of the coefficient matrix exposes the distinction. The later Schmidt unit extends this observation to every cut in a chain.

## 3. Embed a local operator in the full space

To apply X only at site i, put an identity at every other site. The subscript identifies a **physical site**, rather than an entry of a matrix. Thus X₀=X⊗I and X₁=I⊗X for two spins. Operators at different sites commute, while X and Z at the same site do not.

$$
X=\begin{pmatrix}0&1\\1&0\end{pmatrix},\quad
Y=\begin{pmatrix}0&-i\\i&0\end{pmatrix},\quad
Z=\begin{pmatrix}1&0\\0&-1\end{pmatrix},\qquad
\sigma_i^a=I^{\otimes i}\otimes\sigma^a\otimes I^{\otimes(n-i-1)}.
$$

X flips the selected bit. Z multiplies a bit1 by−1. Y combines a flip with a phase: Y|0〉=i|1〉 and Y|1〉=−i|0〉. Check both signs when testing the direction of a complex observable. As usual, the rightmost operator in a matrix product acts first.

| Input basis state | Output of X₀ | Output of Y₁ | Output of Z₀Z₁ |
| --- | --- | --- | --- |
| \|00〉 | \|10〉 | i\|01〉 | \|00〉 |
| \|01〉 | \|11〉 | −i\|00〉 | −\|01〉 |
| \|10〉 | \|00〉 | i\|11〉 | −\|10〉 |
| \|11〉 | \|01〉 | −i\|10〉 | \|11〉 |

Each column describes the action of a larger matrix. A product of Pauli matrices maps each computational basis vector to one other basis vector and multiplies it by1,−1,i or−i. This sparse action can be used before storing every entry of a2ⁿ-by-2ⁿ matrix. It does not eliminate the exponential number of amplitudes if we choose to output the whole state.

Physical angular momentum is S^a=ℏσ^a/2. Consequently, J_S S₀·S₁ and J_σ(X₀X₁+Y₀Y₁+Z₀Z₁) agree when J_σ=J_Sℏ²/4. Confusing Pauli eigenvalues±1 with physical S_z eigenvalues±ℏ/2 introduces a factor of four in the interaction energy. All Hamiltonian coefficients below use the Pauli convention and carry units of energy.

## 4. A graph and an interaction specify different things

Local fields and two-site interactions give the following finite Hamiltonian. Here c sets the energy origin, h contains local field coefficients, and J contains interaction coefficients. Every coefficient is real and has energy units. The axes a,b range over X,Y,Z.

$$
H=cI+\sum_{i,a}h_i^a\sigma_i^a+
\sum_{(i,j)\in E}\sum_{a,b\in\{X,Y,Z\}}J_{ij}^{ab}\sigma_i^a\sigma_j^b.
$$

E is a set of distinct undirected site pairs, written with i<j. An undirected pair does not identify XᵢYⱼ with YᵢXⱼ: each axis is assigned to a particular site. Because operators at different sites commute, all nine axis combinations are Hermitian for real coefficients. This is a general two-body model on the graph, rather than an arbitrary many-body matrix.

Picture four sites on a line. An open chain has edges01,12,23; a ring adds03. A custom graph can also connect sites such as02. The graph identifies allowed terms, while a zero coefficient removes an actual interaction on an allowed edge. Conversely, correlations in the prepared state can connect a pair with no direct interaction.

A two-site ring has one edge01 under our distinct-pair convention. Some periodic sums count both01 and10, adding the same interaction twice. If that convention is intended, its combined coefficient must be stated. When the same symbol J produces different energies in two implementations, check edge multiplicities and coefficient definitions before suspecting a different physical law.

## 5. Read the two-spin Ising model as a matrix

Start with H=−JZ₀Z₁ and J>0. Aligned Z bits have lower energy, which defines the ferromagnetic sign here. Since Z is diagonal, the computational basis is also an energy eigenbasis.

$$
H=-JZ_0Z_1=\operatorname{diag}(-J,J,J,-J),\qquad
\mathcal G=\operatorname{span}\{|00\rangle,|11\rangle\},\qquad \Delta=2J.
$$

| Basis state | Z₀Z₁ | Energy | Subspace |
| --- | --- | --- | --- |
| \|00〉 | +1 | −J | Ground subspace |
| \|01〉 | −1 | +J | Excited subspace |
| \|10〉 | −1 | +J | Excited subspace |
| \|11〉 | +1 | −J | Ground subspace |

The ground subspace is two-dimensional. It would be wrong to accept only |00〉 as the answer: |11〉 and any normalized superposition within their span are also ground states. The gap is the separation2J to the next **distinct** energy. The zero separation between two degenerate ground vectors is not the gap used here.

Changing the sign of J favors opposite bits. Adding cI shifts every energy by c without changing the eigenspaces or gap. When every energy is equal, as for H=0, the whole space is a ground subspace and there is no next distinct level. Do not force a positive-gap formula onto that case.

## 6. Off-diagonal interactions cannot be evaluated from probabilities alone

Now consider H=J(X₀X₁+Y₀Y₁). XX flips both bits. YY also flips them, but its phase depends on the input. The two actions cancel on |00〉 and |11〉 and add between |01〉 and |10〉.

$$
(X_0X_1+Y_0Y_1)|01\rangle=2|10\rangle,\qquad
(X_0X_1+Y_0Y_1)|00\rangle=0,\qquad
E_{\psi_\pm}=\pm2J,\quad
|\psi_\pm\rangle=\frac{|01\rangle\pm|10\rangle}{\sqrt2}.
$$

The two states have identical probabilities in the Z basis but different energies. An off-diagonal Hamiltonian detects relative signs and phases. Replacing amplitudes by probabilities before calculating energy would erase precisely the information needed to distinguish them. Probabilities were sufficient for the diagonal Ising example because of that Hamiltonian's special structure.

For the Heisenberg interaction H=J(X₀X₁+Y₀Y₁+Z₀Z₁), the antisymmetric singlet ψ₋ has energy−3J, while the three triplets have energy+J. We can obtain these results from a4-by-4 matrix or from the square of total spin. Match the Pauli/ℏ convention before comparing the derivations. Knowing this small analytic spectrum does not mean that we have solved the ground state of a general off-diagonal chain.

## 7. Mean energy and variance answer different questions

A nonzero raw vector z lets us retain convenient integer coordinates for a physical state. Calculate its norm N first and divide every expectation by N. Using z†Hz alone would make the reported energy depend on the arbitrary size of the representative.

$$
N=z^\dagger z>0,\qquad
E=\frac{z^\dagger Hz}{N},\qquad
r=(H-E)z,\qquad
\operatorname{Var}(H)=\frac{\|r\|^2}{N}
=\frac{z^\dagger H^2z}{N}-E^2.
$$

E has energy units and variance has energy-squared units. A zero residual identifies an eigenstate. To identify a ground eigenstate, we also need information about other energies. The two-spin Ising state |01〉 is an exact excited eigenstate with zero variance and zero ground probability. Thus zero variance is not a substitute for a ground-state certificate or evidence that an optimization has succeeded.

Term contributions should sum to E. Variance obtained from the squared residual should agree with variance obtained from the second moment. We can also check z†r=0. These separate accounts are stronger evidence than one plausible-looking energy value. A tiny positive variance must remain positive when classifying the state. An energy-origin shift leaves variance unchanged, while multiplying H by a real number b multiplies variance by b².

## 8. A GHZ state in a transverse field reveals action hidden by the mean

A four-spin GHZ state coherently combines the all-zero and all-one configurations. Add a transverse X field to an open ferromagnetic Ising chain. The following example chooses J=1 and transverse coefficient1/2 in the selected energy unit.

$$
|g\rangle=\frac{|0000\rangle+|1111\rangle}{\sqrt2},\qquad
H=-Z_0Z_1-Z_1Z_2-Z_2Z_3-\frac12\sum_{i=0}^3X_i.
$$

Each ZZ term multiplies the GHZ state by−1. Xᵢ flips a single bit and takes the state outside the two basis directions supporting GHZ. Every local X mean is therefore zero and the mean energy is−3. The mean alone looks as though no transverse field had been added, but the full Hamiltonian action contains additional directions.

$$
H|g\rangle=-3|g\rangle-\frac12\sum_iX_i|g\rangle,\qquad
\langle H\rangle=-3,\qquad
\operatorname{Var}(H)=1.
$$

The four states Xᵢg are mutually orthogonal and have unit norm. The squared residual is consequently4×(1/2)²=1. GHZ is not an eigenstate of this transverse-field Hamiltonian. Setting the transverse coefficient back to zero gives zero variance and places it in the ground subspace of the diagonal chain. The same prepared vector can be a ground state for one Hamiltonian and fail to be an eigenstate for another.

This is one evaluation of an operator on a supplied state. Entering a transverse field has not advanced time or automatically evolved the state. Physical evolution additionally needs U(t)=exp(−iHt/ℏ), a time and consistent units. Changing a graph or field control is a different operation from following a physical trajectory.

## 9. Interpret local mixedness and connected correlations

Even for a pure global state, observing only one site requires tracing out the rest. Its density matrix is a2-by-2 Hermitian matrix that admits Bloch coordinates. In GHZ, each site has equal0/1 probabilities and no surviving local coherence after the other sites are removed.

$$
\rho_i=\frac12\left(I+x_iX+y_iY+z_iZ\right),\qquad
(\rho_i)_{01}=\frac{x_i-iy_i}{2},\qquad
\operatorname{Tr}\rho_i^2=\frac{1+x_i^2+y_i^2+z_i^2}{2}.
$$

Every GHZ site therefore has a zero Bloch vector and purity1/2. In contrast, a pure global product state has pure local states. **With global purity as an assumption**, local mixedness diagnoses entanglement between that site and the rest. The same inference does not follow from local purity alone when the global state is mixed.

Connected correlations measure the part of a joint observable that is not explained by the separate means. They can be evaluated even when the two sites have no direct coupling.

$$
C_{ij}^{ab}=\langle\sigma_i^a\sigma_j^b\rangle
-\langle\sigma_i^a\rangle\langle\sigma_j^b\rangle,\qquad i\ne j.
$$

GHZ has zero local Z means and unit pairwise ZZ means, giving connected ZZ=1. A classical equal mixture of |0000〉 and |1111〉 has the same Z statistics. The two preparations differ in the expectation of X₀X₁X₂X₃, which is1 for pure GHZ and0 for the mixture. One connected ZZ value is neither an entanglement measure nor a causal direction. Similarly, an interaction in H does not automatically entangle a product state that we have merely supplied for evaluation.

## 10. Distinguish a conservation law from a sharp prepared state

For an observable A without explicit time dependence, [H,A]=0 ensures that its expectation is conserved for every state under closed Hamiltonian evolution. In contrast, Var(A)=0 in the current state says that a measurement is definite at that moment. These statements cannot replace one another.

$$
M=\sum_iZ_i,\qquad P=\prod_iZ_i,\qquad
[H,A]=0\quad\not\Longleftrightarrow\quad
\operatorname{Var}_\psi(A)=0.
$$

| Hamiltonian / prepared state | Is total M conserved? | Is current M sharp? |
| --- | --- | --- |
| X₀ / \|00〉 | No | Yes, M=2 |
| X₀X₁+Y₀Y₁ / \|00〉 | Yes | Yes, M=2 |
| Z₀Z₁ / (\|00〉+\|11〉)/sqrt(2) | Yes | No, values±2 |

XX alone does not commute with M, but its failure cancels against that of YY in their sum. Explicitly, [M,XX]=2i(YX+XY) and [M,YY]=−2i(YX+XY). Combining separate conservation flags for individual terms would miss this cancellation. Calculate the commutator of the full Hamiltonian instead.

XX flips two Z bits and commutes with P. X₀ flips one bit and does not. The transverse-field Ising chain also conserves Q=∏Xᵢ: both Z factors in each ZZ interaction change sign together, and the X fields commute with Q. This is a different operator from P=∏Zᵢ. Failure of the M and P diagnostics does not establish the absence of all symmetry.

Spin Z parity is not by itself fermionic statistics or a superselection assumption. The following Fock unit explains the additional structure needed for that connection.

## 11. Frustration means that local preferences cannot all be satisfied

Consider the antiferromagnetic Ising triangle H=Z₀Z₁+Z₁Z₂+Z₀Z₂. Each edge prefers opposite bits, with energy−1. Trying to alternate0 and1 around three sites leaves one edge unsatisfied. This is frustration in a small classical diagonal example.

$$
H_\triangle=Z_0Z_1+Z_1Z_2+Z_0Z_2,\qquad
E=-1\ \text{with multiplicity }6,\qquad
E=3\ \text{with multiplicity }2.
$$

The configurations000 and111 contribute+1 on all three edges and have energy3. Each of the other six configurations has two−1 edges and one+1 edge, giving energy−1. Adding the individual term minima would give−3, but no configuration achieves it. Listing local preferences and showing that they can coexist are different tasks.

This supplies intuition for graph constraints and combinatorial optimization. It does not prove a complexity result for large optimization problems or an advantage for a quantum algorithm. Introducing noncommuting terms also raises questions beyond satisfying classical bit constraints. Establish the diagonal triangle exactly before making a broader claim.

## 12. Check what a spectral certificate actually covers

For a diagonal Hamiltonian, enumerate computational basis energies and group equal values into degenerate levels. The ground probability sums over the whole ground subspace. A known positive gap converts excess mean energy into an upper bound on weight outside that subspace.

$$
p_0=\langle\psi|\Pi_0|\psi\rangle,\qquad
1-p_0\le\frac{E-E_0}{\Delta},\qquad
\Delta>0.
$$

If the ratio exceeds1, the probability bound may be capped at1, while retaining the original ratio clarifies its meaning. A constant Hamiltonian has no excitation gap and every state already belongs to its ground subspace. For a general off-diagonal input, this finite experiment does not calculate a spectrum, so the ground classification is **unknown**. That differs from false. The analytic XX+YY and singlet examples illustrate principles without supplying a ground solver for every off-diagonal input.

The experiment covers1…4 sites, local XYZ and all nine two-site axes, with open, distinct-edge ring and custom graphs. It does not replace general three-body interactions, an infinite system, optimization or time evolution. A zero input is not a physical state and is rejected. Complex amplitudes and tiny positive variances are retained. Distinguishing exact rational or interval evidence from display coordinates prevents a smooth picture from becoming a substitute for valid mathematics.

For AI research, assess state expressivity, optimization performance and physical-observable accuracy separately. A neural state with low energy should be accompanied by its Hamiltonian, graph, units, comparison bounds, variance, correlations and size analysis. Enforcing symmetry in a representation differs from obtaining a sharp measurement in one state. Later MPS material explains conditions for using smaller tensors and the freedom in representation coordinates.

## 13. Six problems to predict and explain

**1. Basis and phase.** Apply X₀ to the raw state z=|00〉+i|01〉. After normalization, what is the Y mean of the second spin?

**Explanation.** The output is |10〉+i|11〉. The original norm is N=2, with the first spin in |0〉 and the second in (|0〉+i|1〉)/sqrt(2). The01 entry of its density is−i/2, so Y=+1. Removing a global phase and removing this relative phase i are different operations.

**2. Raw normalization.** For H=−Z₀Z₁ and z=2|00〉+2i|01〉, calculate N, E, Var(H) and the ground-subspace probability.

**Explanation.** N=8. The occupied basis states each have probability1/2 and energies−1,+1, so E=0. Because H²=I, the variance is1. The ground subspace contains00 and11, giving probability1/2. Treating the unnormalized numerator as an expectation would produce a result depending on the arbitrary scale of z.

**3. Transverse GHZ.** Replace the field−(1/2)ΣXᵢ in section8 by−hΣXᵢ. How do the mean and variance change? Distinguish h=0 from h≠0.

**Explanation.** The local X means remain zero, so E=−3. Four mutually orthogonal Xᵢg components have residual coefficient−h, giving variance4h². At h=0, GHZ belongs to the ground eigenspace of the diagonal chain. At h≠0, positive variance shows that it is not an eigenstate. Here h is the transverse energy coefficient.

**4. A sum and its conservation law.** Which of XX, YY and XX+YY commutes with M=Z₀+Z₁? Can the zero M variance of |00〉 decide this?

**Explanation.** The separate commutators are2i(YX+XY) and−2i(YX+XY), both nonzero. They cancel in the sum. The fact that |00〉 is an M eigenstate describes the preparation and cannot determine an operator identity on the entire space.

**5. Frustration and degeneracy.** Find the triangle's ground energy, degeneracy and gap. Explain why adding three local minima does not give its ground energy.

**Explanation.** Three bits cannot make every neighbor pair opposite, so at least one edge contributes+1. Six configurations have energy−1, while the two uniform configurations have energy3. The ground energy is−1, its multiplicity is6 and the gap to the next distinct energy is4. The separate optimal conditions are incompatible.

**6. Does correlation alone prove entanglement?** Compare connected ZZ and the full X product for pure GHZ and an equal mixture of the all-zero and all-one states. How should we interpret this comparison when the experiment accepts pure inputs only?

**Explanation.** Both preparations have local Z mean0 and joint ZZ mean1, hence connected ZZ=1. The full X product exchanges the two amplitudes of pure GHZ and has expectation1; its diagonal expectation in the mixture is0. ZZ alone cannot distinguish the preparations or certify their entanglement. The mixture is a separate analytic comparison, not a mixed state encoded by different amplitudes in a pure-state calculator.

## 14. Explain the model before changing its representation

After a calculation, state the basis order, the Pauli versus physical-spin coefficient convention, and the distinction between permitted graph edges and nonzero interactions. Then explain the current state's energy, variance, local purity and connected correlations separately from the Hamiltonian's conservation laws. Keeping an uncomputed ground energy or gap out of the conclusion is also part of understanding the result.

Different responses to the growing state space address different questions. Identical-particle statistics requires Fock operators and their sign structure. Compression requires an MPS representation and a suitable distribution of Schmidt weights. New AI research must bring the physical model, state representation, numerical method and verification evidence together. This small chain makes these distinctions accessible to hand calculation; it does not finish the subject of many-body research.

For further study, consult [MIT 8.05 on tensor products and angular momentum](https://ocw.mit.edu/courses/8-05-quantum-physics-ii-fall-2013/resources/lecture-videos/) and [Sandvik's lectures on quantum-spin computation](https://physics.bu.edu/py502/lectures6/qspins.pdf). Exact diagonalization, Monte Carlo and finite-size scaling for larger systems each require their own assumptions and additional verification.
