## 1. A circuit is an ordered computation

A circuit specifies which operation acts, on which wire, and when. Listing the gates is insufficient: an order change can change the output even when every gate is valid. This is useful preparation for quantum machine learning, where a trainable circuit is a composition of operations, and for debugging simulators, where a wire-order mistake can look like a physics mistake.

Assume complex vectors, tensor products and unitary matrices. We first use ideal pure states. Noise and measurement require the channel formulation in the following units. The finite experiment uses one to three qubits so every amplitude can be inspected. The general definitions have no three-qubit restriction.

For gates A then B, the state is B A|ψ〉. Circuit diagrams usually draw time from left to right; matrix products act from right to left. Always translate the verbal instruction before multiplying.

$$
|\psi_k\rangle=U_k|\psi_{k-1}\rangle,\qquad
|\psi_m\rangle=U_m\cdots U_2U_1|\psi_0\rangle.
$$

An analogy is successive rotations of a physical object. Each rotation is reversible, yet exchanging two rotations can change the final orientation. The analogy explains composition order; quantum amplitudes additionally carry relative phases.

## 2. Interference requires bringing paths back together

The Hadamard gate splits a basis state into two amplitudes. Applying a phase gate changes how those amplitudes recombine under another Hadamard. A probability-only description of the intermediate state misses this information.

$$
H=\frac1{\sqrt2}\begin{pmatrix}1&1\\1&-1\end{pmatrix},\quad
Z=\begin{pmatrix}1&0\\0&-1\end{pmatrix},\quad
H|0\rangle=\frac{|0\rangle+|1\rangle}{\sqrt2}.
$$

Follow H, Z, H on |0〉. The first H gives the plus state. Z changes it to the minus state. The last H gives |1〉. In contrast, Z, H, H returns |0〉. The same gate multiset produces different outputs. Check the identity HZH=X, not merely the final probabilities for one input.

For T=diag(1,e^{iπ/4}), HTH leaves a continuous-looking interference probability even though this particular phase is fixed:

$$
HTH|0\rangle=\frac{1+e^{i\pi/4}}2|0\rangle+
\frac{1-e^{i\pi/4}}2|1\rangle,\qquad
p(1)=\frac{2-\sqrt2}{4}.
$$

Both intermediate computational-basis probabilities after H and TH are1/2. The final probabilities differ because the last H converts relative phase into a population difference. Predict the result before calculating it. Removing the last H removes this particular readout of phase, not the phase itself.

## 3. Wire labels determine tensor factors

We use |b₀b₁b₂〉 with site0 the most significant bit. Thus |101〉 has coordinate index5, X₀=X⊗I⊗I, and X₂=I⊗I⊗X. A library with a different endian convention can be correct; its coordinates must be converted before comparison.

CX with control0 and target1 maps |10〉 to |11〉 and leaves |01〉 fixed. Reversing control and target changes both facts. CZ multiplies |11〉 by−1 and leaves the other two basis vectors fixed. SWAP exchanges two wire labels; it does not introduce the antisymmetric sign of a fermionic mode exchange.

Starting from |00〉, H on wire0 followed by CX₀→₁ produces (|00〉+|11〉)/√2. Each local computational outcome is uniform, while the two outcomes are perfectly correlated. A product of local marginals cannot reconstruct that joint state. Inspect joint amplitudes rather than interpreting two independent probability bars as the full result.

## 4. Global phase and controlled implementations

Multiplying a whole state by e^{iα} leaves its density matrix unchanged. Multiplying only one branch generally changes interference. A circuit simulator should distinguish exact amplitude equality from equality up to a common phase.

There is a further subtlety: U and e^{iα}U induce the same channel on a system, but controlled-U and controlled-(e^{iα}U) can differ on the control. For U=I and α=π, applying the latter to |+〉⊗|0〉 changes the control to |−〉. Therefore a system channel alone does not specify its coherent controlled implementation. This matters when comparing circuit transformations and proposed learning architectures.

## 5. Verification and cost

The reverse circuit applies adjoints in reverse chronological order. H, X, Y, Z, CX, CZ and SWAP are self-inverse; S and T require S† and T†. Undoing gates in their original order is generally wrong. Compare the recovered complex amplitudes as well as the norm.

The calculation kernel keeps coefficients in Q(√2,i), including H and T, and returns raw amplitudes with their squared norm. If the input is unnormalized, probabilities require division by that norm. A zero vector has no normalized state and must be rejected. Exact algebra avoids declaring tiny residuals zero, but does not avoid exponential state size: n qubits have2ⁿ amplitudes. Small simulation success is not a scalability claim.

The independent reference instead constructs full embedded matrices. Agreement of two different calculation paths, inverse recovery, and every computational basis action are stronger evidence than checking norm preservation alone: applying the wrong unitary also preserves norm.

## 6. Read probabilities, local states and joint correlations together

Start with any nonzero raw vector z. Its squared norm N determines how to interpret the entries. Raw amplitudes are not already normalized merely because they were produced by a circuit:

$$
N=z^\dagger z,\qquad p_b=\frac{|z_b|^2}{N},\qquad
\rho=\frac{zz^\dagger}{N},\qquad \rho_A=\operatorname{Tr}_{\bar A}\rho.
$$

For z=(1,i), N=2 and the off-diagonal entries of the density matrix are −i/2 and i/2. Its computational probabilities are1/2 and1/2, but it is not the maximally mixed state. The off-diagonal entries carry information that those two probabilities omit.

A single-qubit reduced state can be written in Bloch coordinates. These are expectations, not three independent probabilities:

$$
\rho_i=\frac{I+r_xX+r_yY+r_zZ}{2},\qquad
r_a=\operatorname{Tr}(\rho_i\sigma_a),\qquad
\operatorname{Tr}(\rho_i^2)=\frac{1+\|r\|^2}{2}.
$$

The z=(1,i) example has r=(0,1,0). Checking the sign of Y is a useful way to catch an incorrect complex conjugation. A point on the Bloch sphere is pure; a reduced state inside the ball can be mixed even when the full state is pure. In a Bell state, each Bloch vector is0 and local purity is1/2, while joint purity is1. Treating those local vectors as independent pure states would lose the entanglement.

Two-site expectations show some of the missing joint information. Connected correlations subtract the part explained by the individual means:

$$
C_{ab}^{ij}=\langle\sigma_i^a\sigma_j^b\rangle,
\qquad C_{ab}^{ij,\mathrm{conn}}=
C_{ab}^{ij}-\langle\sigma_i^a\rangle\langle\sigma_j^b\rangle.
$$

For the plus Bell state, XX=1, YY=−1 and ZZ=1, with zero individual means. A product input can instead have a nonzero two-site expectation and zero connected correlation. For example, |+i〉⊗|+〉 has Y₀X₁=1, entirely explained by Y₀=1 and X₁=1. Nonzero raw correlation alone is not an entanglement certificate.

Even all two-site correlations need not specify a three-qubit state. The GHZ state and the equal classical mixture of |000〉 and |111〉 share their one- and two-qubit reduced states. Their three-site XXX expectation differs:1 for GHZ,0 for the mixture. Retain the full density matrix when deciding whether two joint states agree.

Finally, specify the fidelity convention. Here the squared pure-state overlap is

$$
F_2(z,w)=\frac{|z^\dagger w|^2}{(z^\dagger z)(w^\dagger w)}.
$$

It equals1 for states differing only by a global phase, and1/2 between |0〉 and |+〉. Some texts call the square root of this quantity fidelity. Compare formulas before comparing reported numbers. Background on reduced states and the root-fidelity convention: [IBM density matrices](https://learning.quantum.ibm.com/course/general-formulation-of-quantum-information/density-matrices), [IBM purifications and fidelity](https://learning.quantum.ibm.com/course/general-formulation-of-quantum-information/purifications-and-fidelity).

## 7. Exercises with explanations

**Predict:** Does Z change the computational probabilities of |+〉? No. It changes the relative sign, which an X-basis measurement can detect. An H before a computational measurement implements that readout.

**Calculate:** Apply CX with control1 and target0 to |01〉. The control is1, so the output is |11〉. With control0 and target1 it would remain |01〉. This checks both bit order and the meaning of control.

**Debug:** A circuit returns the correct norm and the wrong Bell correlations. Is normalization the likely fix? No. Check the initial tensor ordering, target/control indices and gate chronology. A wrong unitary can preserve every norm while computing the wrong state.

## 8. Connections and sources

Parameterized circuits add differentiable gates and objectives; noise changes state vectors into density operators. The next unit asks which noisy updates are physically meaningful. Read the earlier unitary and entanglement lessons before treating local statistics as a complete circuit description. Representation background: [IBM Quantum Learning](https://learning.quantum.ibm.com/course/general-formulation-of-quantum-information/quantum-channels). The exact HTH and wire-order calculations here are local worked examples.
