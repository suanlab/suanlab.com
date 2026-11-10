# Finite circuit observation and validation contract

2026-11-09. This is the calculation contract for the unregistered `finite-qubit-circuit-order-interference` lesson. Its scope is1–3 qubits, at most16 chronological gates, and exact Q(√2,i) raw input amplitudes. It complements the [channel contract](quantum-channel-mathematical-contract.md); it is not a user-input decoder, Worker protocol or reader implementation.

## State and ordering conventions

Site0 is the most significant bit. Gate lists follow time: `[A,B]` computes BA. Supported gates are I/X/Y/Z/H/S/S†/T/T†/CX/CZ/SWAP, with explicit distinct control/target or swap sites. The input vector has2ⁿ components and nonzero squared norm N. Raw amplitudes remain raw; every reported density and probability divides by N. No normalized amplitude outside the exact field is replaced by a rounded field coefficient.

Prefix0 is the input; prefix k is the state after k gates. Each prefix reports raw amplitudes, squared norm and its residual from the input norm, probabilities |z_b|²/N, density zz†/N and global purity. Exact rational coefficient pairs encode a+b√2 separately for real and imaginary parts. Reports contain no BigInt and survive JSON transport without a floating-point conversion.

## Local and joint observations

Every site has a2×2 reduced density matrix, purity and Bloch vector ordered X,Y,Z. Two-site arrays report all nine Pauli expectations and their connected values C_ab−r_i,a r_j,b, with increasing site order. A local mixed state can coexist with a globally pure state; the report must not replace an entangled state by a product of local marginals.

Raw correlations need not be connected correlations. Connected correlations need not certify entanglement for a mixed global state. Even all pairwise data need not reconstruct a three-qubit state: GHZ and the classical mixture of |000〉 and |111〉 have identical two-site reductions but different XXX expectations. The complete density matrix remains available.

The final comparison to the input includes an exact normalized overlap and **squared** pure-state fidelity F₂=|z†w|²/[(z†z)(w†w)]. Fields explicitly declare `fidelityConvention: squared-pure-state-overlap`. Amplitude equality and physical pure-state equality are distinct flags. A global phase may change the first while preserving the second and F₂=1. [Watrous/IBM's fidelity lesson](https://quantum.cloud.ibm.com/learning/en/courses/general-formulation-of-quantum-information/purifications-and-fidelity/fidelity) uses the absolute overlap convention, so its reported fidelity must be squared before comparing to F₂.

## Operator and inverse checks

The product executes the circuit on every basis vector to expose the full unitary and U†U−I residual. It also runs the inverse gate list on the final raw vector, reporting restored amplitudes, residual, squared fidelity and exact recovery. Norm preservation alone is insufficient: the wrong unitary can preserve all norms.

The independent validator builds dense embedded matrices and evolves states and operators by a separate path. It obtains reductions by inserting environment coordinates, Bloch values from matrix entries, and pair expectations through sparse Pauli word actions. It restores the state with U† rather than the product inverse-gate function. The Python reference uses dense SymPy gates, tensor contractions, matrix traces and exact simplification. These paths share mathematical definitions; the two TypeScript paths share exact scalar arithmetic and response comparison, with SymPy checking those outputs separately.

## Dependency boundary

`learning-qubit-scalar.ts` holds the previously implemented scalar operations. The circuit kernel preserves its public exports through re-exports, and channel matrix arithmetic imports the common scalars directly. This permits both circuit and channel validator bundles to exclude the product circuit runner, circuit report and channel model. Calculation modules remain outside registered source consumers. The graph check records source hashes, not only entry names.

## Remaining publication work

Next implement bounded input/response protocols, calculation and validation Workers, cancellation and retry. Review manuscript examples, guided experiments and questions against these exact observations. Then build accessible2D/Three.js teaching surfaces, persistence, bilingual responsive readers and actual production-asset verification. Numerical agreement does not establish any of these UI or transport gates, nor complete the original topic or the full curriculum.
