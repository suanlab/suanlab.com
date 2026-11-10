# Quantum circuit·channel: authoring and verification contract

2026-11-09. Four new units extend the existing unitary, density-matrix and POVM lessons. No week or length limit applies. These are preparation artifacts; registration, Worker/UI integration and reader verification remain pending.

| Draft unit | Mathematical question | Planned experiment |
|---|---|---|
| `finite-qubit-circuit-order-interference` | How do gate order, relative phase and wire order determine a circuit? | Compare HZH, ZHH, HTH and Bell preparation, inspecting every prefix and inverse |
| `qubit-channel-kraus-choi-verification` | Which linear transformations remain physical with a reference system? | Compare damping, dephasing, transpose and trace-decreasing branches; inspect TP/CP/unitality separately |
| `system-environment-stinespring-reference` | How can reversible joint dynamics give irreversible local dynamics? | Trace an environment, mix its basis, and compare the channel on half of a Bell state |
| `quantum-instruments-records-feedforward` | What information disappears when a measurement record is forgotten? | Switch record retention, conditional correction and coherent control; expose zero-probability branches |

## Boundaries and conventions

- Circuit numerical foundation: 1–3 qubits, at most16 chronological gates; I/X/Y/Z/H/S/S†/T/T†/CX/CZ/SWAP. Site0 is the most significant bit. Exact amplitudes live in Q(√2,i). Input normalization is not silently changed; the kernel returns the raw squared norm. A future display must divide probabilities by this norm and use certified intervals when normalized amplitudes leave the field.
- Choi convention: unnormalized J on reference⊗output, J=Σᵢⱼ|i〉〈j|⊗Φ(|i〉〈j|), trace J=2 for a TP qubit map, Tr_output J=I. J/2 is the normalized Bell-output state. State the convention in every visualization.
- Damping hand check uses survival amplitude3/5 and decay amplitude4/5, so γ=16/25. This is a worked example, not an arbitrary-parameter channel implementation.
- Existing POVM lessons already explain effects and disturbance. New material must connect records, Kraus label changes, channel equivalence and reference-system tests rather than repeat those lessons.
- No generic controlled-channel operation: a channel does not determine the coherent phase of a particular unitary implementation. Deferred measurement requires retained records, available controlled operations and a specified final observation.

## Verification gates

The independent Python generator builds dense SymPy matrices; the TypeScript circuit kernel propagates amplitudes. Compare every prefix, complex signs, both control orientations, all three wire positions, exact norms and inverse recovery. Also test invalid sites/dimensions, zero vectors, unsupported gates and gate budgets. Keep algebraic results exact; a small residual is not an exact-zero proof.

Channel/environment/instrument examples are independently derived in the generator and checked against explicit identities. These checks do not yet constitute product channel validators. Implement and cross-check the channel numerical engine, Choi construction/PSD and partial traces next. Then add detailed guided experiments, questions, diagrams, responsive accessible controls, calculation and independent-validation Workers, interruption/retry and persistence. Heavy algebra stays outside reader main-thread imports.

Registration requires bilingual content-loader checks, actual production static routes, desktop/mobile Korean/English light/dark behavior, keyboard and accessibility checks, input failure and Worker lifecycle checks, discovery/menu checks and preservation of previous units. Introductory topic completion and later research-depth/domain audits remain distinct.

## Sources

Representation conventions and CP/TP criteria were checked against [IBM Quantum Learning, Quantum channels](https://learning.quantum.ibm.com/course/general-formulation-of-quantum-information/quantum-channels) and [Watrous, The Theory of Quantum Information, chapter2](https://jhwatrous.github.io/TQI.pdf). Our Choi factor order is explicitly reference⊗output; translate source conventions before comparing matrices. Worked examples and their exact calculations are authored locally. Later research extensions include process tomography, channel discrimination, dilation freedom, open-system simulation and resource costs; no claim that those extensions are already complete.
