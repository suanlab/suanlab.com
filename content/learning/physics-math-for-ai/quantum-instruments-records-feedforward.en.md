## 1. A measurement has an output record as well as a state

A POVM specifies outcome probabilities, while an instrument also specifies the corresponding quantum outputs. A channel obtained by forgetting the record contains less information. To reason about adaptive circuits, specify all three objects: the outcome probability, the conditional state, and what downstream operations can read.

We assume the previous POVM and channel lessons. Here we connect those notions to a complete computation. The difference matters for error correction, adaptive inference and feedback: a correction can depend on an available classical result, but cannot depend on a result that was discarded.

## 2. Keep branches unnormalized until conditioning is needed

For outcome x, let A_x be a completely positive, trace-nonincreasing map. The sum over outcomes must be TP. Multiple Kraus operators may belong to one coarse-grained outcome:

$$
A_x(\rho)=\sum_\alpha K_{x\alpha}\rho K_{x\alpha}^\dagger,\quad
p_x=\operatorname{Tr}A_x(\rho),\quad
E_x=\sum_\alpha K_{x\alpha}^\dagger K_{x\alpha}.
$$

Then p_x=Tr(E_xρ) and ΣE_x=I. If p_x>0, the conditional state is ρ_x=A_x(ρ)/p_x. If p_x=0, report an impossible branch and leave the conditional state unavailable. The unnormalized zero branch is still a perfectly meaningful operator in the sum.

The distinction also protects linearity. A_x is linear, while dividing by p_x generally is not. Averaging already-normalized branch states without their probabilities produces the wrong channel.

## 3. A classical–quantum state makes the record explicit

Represent the classical label by orthogonal basis states in register C:

$$
\omega_{CB}=\sum_x |x\rangle\langle x|_C\otimes A_x(\rho).
$$

This block-diagonal state has trace 1. Its blocks contain both probability and output state. Tr_B ω is the classical outcome distribution, and Tr_C ω=ΣA_x(ρ) is the forgotten-record output. The absence of off-diagonal record blocks matters: a classical record is not automatically a coherent superposition of outcome labels.

For a Z measurement of |+〉, the two blocks are |0〉〈0|/2 and |1〉〈1|/2. Forgetting the record gives I/2. Keeping the record lets an observer predict the measured system's Z value exactly. Both descriptions have the same system marginal but support different downstream tasks.

## 4. Same probabilities can hide different disturbances

Consider two instruments with effects E₀=|0〉〈0| and E₁=|1〉〈1|. The projective instrument uses K₀=|0〉〈0|, K₁=|1〉〈1|. A measure-and-reset instrument uses L₀=|0〉〈0|, L₁=|0〉〈1|. Their effects coincide because K†K=L†L outcome by outcome.

For |+〉, both report probabilities 1/2 and1/2. The projective instrument forgets to I/2; the reset instrument forgets to |0〉〈0|. Thus identical outcome histograms do not certify identical state updates. Conversely, changing the environment basis in the preceding unit can preserve the forgotten channel while changing the records. Neither the POVM alone nor the forgotten channel alone specifies the full instrument.

## 5. Feedforward uses the retained result

If the controller applies a channel F_x after reading x, the final output is ΣF_x(A_x(ρ)). The argument is unnormalized, so F_x must act linearly there. For a projective Z measurement, apply I after0 and X after1. Both branches then end in |0〉; for |+〉 the final state is |0〉〈0| instead of I/2.

If the result was forgotten before the controller acts, it cannot choose I or X according to that missing result. A fixed later operation might coincide on some inputs, but it does not reconstruct the absent classical correlation. This is why information-flow diagrams should show record edges alongside quantum wires.

## 6. Coherent control and deferred measurement need conditions

A measurement interaction can first correlate a system with an ancilla coherently. For a Z readout, CX from system B to record C maps |+〉_B|0〉_C to a Bell state. Controlled-X from C back to B then produces |0〉_B|+〉_C. Measuring C at the end gives the same final classical record and system state as immediate measurement followed by the conditional correction.

Before that final measurement the coherent version has |+〉 on C, while the classical version has a diagonal record. A phase-sensitive measurement on C can distinguish them. Thus the equivalence concerns specified final observations after the appropriate dephasing/measurement; it is not equality of every intermediate joint state.

Deferral requires preserving the record coherently, implementing its controlled later operations, and not performing incompatible uses of that record. Reset, leakage, inaccessible environments or restrictions on controls can obstruct this replacement. A channel alone does not specify the coherent phase of a controlled implementation. State these assumptions before using deferred measurement to simplify an algorithm.

## 7. A reported outcome can hide several physical alternatives

A useful counterexample to “a known outcome gives a pure state” groups more than one Kraus term into one reported result. Let outcome 0 contain √(1/2)I and √(1/4)Z, and outcome 1 contain √(1/4)X. The effects are3I/4 and I/4, so they sum to I. On |+i〉 the unnormalized branches are

$$
A_0(\rho)=\frac12\rho+\frac14Z\rho Z,\qquad
A_1(\rho)=\frac14X\rho X.
$$

Outcome0 has probability 3/4. After dividing by that probability, its Y expectation is(1/2−1/4)/(3/4)=1/3 and its purity is(1+1/9)/2=5/9. Outcome1 has probability 1/4 and conditional Y=−1. Forgetting the record gives total Y=0 and output I/2. Coarse reporting hides which operation occurred inside outcome 0, even though the original input was pure.

Compare this with Z projection of |+〉. Its two probabilities are1/2,1/2 and its conditional outputs are |0〉 and |1〉. A measure-reset instrument has identical effects but both conditional outputs are |0〉. Applying X after record1 also resets the projected system to |0〉, using a retained record. Probability agreement alone does not identify the state update or the information available to the controller.

For the deferred example, prepare B in |+i〉 and C in |0〉. A CX from B to C followed by a CX from C to B produces |0〉_B|+i〉_C. Before the final record readout, C has Y=1. Dephasing C gives I/2 with Y=0, while B remains |0〉. The specified final system and classical readout can agree with immediate measurement and feedforward, even though intermediate coherent and classical records differ. An X or Y readout of the preserved record is a different task and must not be silently included in that equivalence.

## 8. Exercises with explanations

**Zero branch:** Measure Z on |0〉 and ask for the outcome 1 conditional state. Its probability is0, so the conditional state is undefined. The branch operator is0 and contributes0 to all unconditioned sums.

**Disturbance:** Can equal outcome probabilities prove two instruments equivalent? No. The projective and reset instruments above have identical effects for every input but different conditional and forgotten outputs.

**Deferral:** Why can the coherent and classical correction circuits agree at their final readout? After measuring/dephasing C, their observed classical–quantum outputs coincide. They can differ before that operation, so equality of a selected final task does not prove equality of all states.

## 9. Verification and further questions

Compare the full classical–quantum block state as well as the branch probabilities and system marginal. Normalize only positive-probability branches. For coherent deferral, specify the record basis, available controls and final observation before comparing results. Further research includes adaptive protocols, quantum trajectories and feedback with finite memory; the finite examples provide a starting point for those questions.

For channels, dilation and measurement background see [Watrous, The Theory of Quantum Information](https://jhwatrous.github.io/TQI.pdf) and [IBM Quantum Learning](https://learning.quantum.ibm.com/course/general-formulation-of-quantum-information/quantum-channels). Record retention and comparison conditions are explicit parts of our worked example.
