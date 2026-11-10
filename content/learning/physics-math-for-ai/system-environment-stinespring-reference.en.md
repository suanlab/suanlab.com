## 1. Reversibility depends on what we retain

Suppose a system loses excitation to an environment initially in a known state. If we retain both outputs, we can describe the transfer coherently. If we retain only the system, information about the transfer is missing. The resulting local evolution can be irreversible even when the joint operation is reversible.

This unit connects the channel lesson to an explicit environment. Assume density matrices, partial trace and Kraus operators. Our environment is a qubit initially in |0〉. This is a finite illustrative dilation, not a model of every thermal bath or a proof of Markovian dynamics. An initially correlated environment can require a different description; a fixed CPTP rule on all system states must not be assumed without checking the preparation procedure.

## 2. Build an isometry before choosing a full unitary

Use output order system B then environment E. Define V from a two-dimensional input into a four-dimensional output by

$$
V|0\rangle=|00\rangle,\qquad
V|1\rangle=\frac35|10\rangle+\frac45|01\rangle.
$$

The images are orthogonal and have norm 1, so V†V=I. This makes V an isometry. It is rectangular, so VV† is a projection rather than the four-dimensional identity. Confusing these two products would incorrectly reject the construction.

A full joint unitary can extend V, but its action on other environment inputs requires extra choices. For example, complete the action on joint input basis states by U|00〉=|00〉, U|10〉=(3|10〉+4|01〉)/5, U|01〉=(−4|10〉+3|01〉)/5 and U|11〉=|11〉. The four images are orthonormal. Changing the completion outside the prepared-environment subspace need not change the observed channel with E initially |0〉.

The environment carries a record of whether an excitation moved. This is an interpretation of the chosen basis and physical implementation. A different environment basis gives different records without necessarily changing the unobserved system output.

## 3. Trace out the environment explicitly

For system input ρ, form VρV† and sum matrix elements with matching environment indices:

$$
\Phi(\rho)=\operatorname{Tr}_E(V\rho V^\dagger),\qquad
\Phi(\rho)_{ab}=\sum_{e=0}^1(V\rho V^\dagger)_{ae,be}.
$$

For input |1〉, the joint state is (3|10〉+4|01〉)/5. Its system output is diag(16/25,9/25). The cross term |10〉〈01| vanishes under the environment trace because its two environment indices differ. This is not an instruction to delete all off-diagonal system entries: coherences between terms with the same environment label can survive.

Take |+i〉=(|0〉+i|1〉)/√2. The reduced output is

$$
\Phi(|+i\rangle\langle+i|)=
\begin{pmatrix}41/50&-3i/10\\3i/10&9/50\end{pmatrix}.
$$

Its determinant is36/625, so it is mixed. The full output V|+i〉 remains pure. The difference is which degrees of freedom we keep, not random numerical error.

## 4. Recover Kraus operators and understand their freedom

Projecting the environment onto basis label e gives K_e=(I_B⊗〈e|)V. In the computational basis these are the two damping Kraus operators from the previous unit. Inserting the environment identity Σ|e〉〈e| into the partial trace gives the operator sum.

Now use the plus/minus environment basis. The Kraus operators become L_+=(K₀+K₁)/√2 and L_−=(K₀−K₁)/√2. Expanding their two output terms cancels the cross terms and recovers the same channel. The individual terms generally differ. For input |0〉, computational-environment records have probabilities 1 and0, while plus/minus records each have probability 1/2. Both forgotten-record outputs are |0〉〈0|.

Thus Kraus operators are not unique physical event names attached to the channel. A particular measured environment defines an instrument. An environment unitary followed by a forgotten environment leaves the system channel unchanged; reading the environment can distinguish the implementations.

## 5. Attach a reference to test a map on correlations

Let R be a reference and apply the channel only to system A. The extended action is id_R⊗Φ. For the normalized Bell input |Ω〉/√2, the output is J(Φ)/2 in reference⊗system order. The environment dilation orders the combined output R,B,E; keep that order when taking partial traces.

For every TP channel, the reference marginal stays the same:

$$
\operatorname{Tr}_B[(\mathrm{id}_R\otimes\Phi)(\rho_{RA})]
=\operatorname{Tr}_A\rho_{RA}.
$$

One proof uses the adjoint: for any reference observable F, its expectation after the map involves F⊗Φ*(I)=F⊗I. This does not mean joint correlations are unchanged. Damping can alter Bell correlations while leaving the reference alone maximally mixed. A product state and a Bell state can share the same local marginals; inspecting marginals alone cannot certify the joint process.

## 6. Follow one input from no damping to complete transfer

The lab uses a rational parameter t with survival amplitude c=(1−t²)/(1+t²) and transfer amplitude s=2t/(1+t²). They obey c²+s²=1; the decay probability is γ=s². At t=1/2 they are3/5 and4/5. Label t and γ separately, since they are not the same number.

For input |+i〉 and environment |0〉, the joint pure state at that point is

$$
|\Psi\rangle_{BE}=\frac1{\sqrt2}
\left(|00\rangle+\frac{4i}{5}|01\rangle+\frac{3i}{5}|10\rangle\right).
$$

Taking the two partial traces gives

$$
\rho_B=\begin{pmatrix}41/50&-3i/10\\3i/10&9/50\end{pmatrix},\qquad
\rho_E=\begin{pmatrix}17/25&-2i/5\\2i/5&8/25\end{pmatrix}.
$$

The matrices differ, but both purities are553/625. Equal reduced spectra follow from a pure bipartite joint state; they do not imply equal reduced matrices. At t=0 the two outputs are |+i〉 and |0〉. At t=1 they are |0〉 and |+i〉: the input has transferred to the environment. A pure reset system output therefore does not establish that information vanished from the joint system.

Change the input while keeping t=1/2. Input |1〉 yields an entangled joint state with reduced purity 337/625, while input |0〉 remains the product |00〉 with reduced purity 1. The same interaction need not entangle every input. With mixed input, first verify whether the joint state is pure before using the pure-state equality of reduced spectra. The prepared mixed-input example has joint purity 11/18, system purity 1011/1250 and environment purity 3937/5625, which are distinct quantities.

Finally, inspect environment basis freedom on input |0〉. Computational-basis Kraus first-column norms give record probabilities 1 and0; plus/minus first-column norms give1/2 and1/2. Both unconditioned system outputs are |0〉〈0|. The next unit retains these labels and compares the full instruments.

## 7. Exercises with explanations

**Check the shape:** Is VV†=I necessary? No. V maps dimension 2 into dimension 4. Its columns must be orthonormal; VV† projects onto their two-dimensional image. V†V=I is the required isometry condition.

**Trace carefully:** Why does |10〉〈01| disappear? Its environment factor is |0〉〈1| with trace 0. The system term |0〉〈1| can survive when paired with environment |0〉〈0|, so “remove all off-diagonals” is not a partial trace algorithm.

**Compare records:** Is measuring E in the plus/minus basis the same experiment as computational-basis measurement? Their unconditioned system channels agree, but their records and conditional states can differ. Specify which data are retained before claiming equivalence.

## 8. Verification and research connections

Verify V†V=I and preserve tensor order in both partial traces. Compare original and mixed Kraus representations on all four matrix units; agreement on one input alone can hide different channels. Check the reference marginal and the joint output separately. Further study includes minimal dilations, environment dimension versus Choi rank, initially correlated states, memory and physical resource costs.

For the representation theorem see [IBM Quantum Learning](https://learning.quantum.ibm.com/course/general-formulation-of-quantum-information/quantum-channels) and [Watrous, chapter2](https://jhwatrous.github.io/TQI.pdf). The finite V and its explicit basis order make the present calculation independently reproducible.
