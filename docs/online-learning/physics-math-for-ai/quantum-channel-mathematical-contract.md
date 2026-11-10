# Finite channel, environment and record calculation contract

2026-11-09. This contract covers unregistered numerical preparations for three new channel-related units. It does not certify their reader, Workers or the full quantum curriculum.

## Exact scalars and representations

Complex entries belong to Q(√2,i). A real wire value `[a,b]` denotes a+b√2; a and b are canonical rational strings. Every response preserves those exact coefficients. JSON transport introduces no binary floating-point conversion. The input is a valid2×2 density matrix; zero vectors and nonpositive/non-unit-trace matrices are rejected. Normalizing a vector computes its exact quadratic-field squared norm and quotient.

Maps accept at most eight weighted2×2 operators, a transpose example, or the four images of matrix units. A weighted term means **w AρA†**, not AρA† with an ignored weight. For nonnegative w its formal Kraus operator is √w A; √w need not belong to Q(√2). Exact density-level weighting avoids approximating that root. Signed operator sums are also representable as linear maps; they receive physicality labels only after the complete Choi test. Negative representation weights alone do not prove non-CP:2ρ−ρ is the identity map.

## Map and reference observations

Use J=Σ|i〉〈j|_R⊗Φ(|i〉〈j|)_B on reference⊗output, with no normalization factor. CP means J≥0; TP means Tr_B J=I. HP means J is Hermitian. Unitality tests Φ(I)=I separately. The Hilbert–Schmidt adjoint uses conjugated superoperator coefficients, including for deliberately non-HP examples; its observable trace-pairing interpretation is used for physical maps. Trace nonincrease tests I−Φ*(I)≥0 and does not imply CP.

The product certifies PSD by Hermiticity and **all** principal minors, up to dimension4. Leading minors alone do not suffice for semidefiniteness. It does not claim arbitrary-dimensional PSD support or manufacture an approximate spectrum. The independent validator uses exact pivoted Schur complements and rejects zero diagonal pivots with nonzero rows. The Python reference uses rational characteristic-polynomial negative-root counts, or exact symbolic eigenvalue signs when coefficients are irrational. None of these methods uses a tolerance to turn a negative value into zero.

Reports include input/output matrices, the Choi matrix and15 principal minors, both relevant partial traces, the adjoint identity, unital image, physicality flags, and the normalized Bell output J/2. A nonphysical linear image remains inspectable as a matrix; it is explicitly distinguished from a density matrix. `applyWithReference` acts on the second factor. Composition arguments are chronological: first then second.

## Parameter meanings

- Damping uses a rational stereographic parameter0≤t≤1: c=(1−t²)/(1+t²), s=2t/(1+t²), decay probability γ=s². At t=1/2, c=3/5 and γ=16/25. A future control must label t and γ separately.
- Random phase flip uses Φ_p(ρ)=(1−p)ρ+pZρZ. Its off-diagonal entries acquire1−2p. Full dephasing is p=1/2; p=1 is a deterministic Z unitary. Decoherence is not monotone over this whole parameter interval.
- Depolarization uses Φ_p(ρ)=(1−p)ρ+pI/2 on normalized states. Its weighted Pauli form has identity weight1−3p/4 and X/Y/Z weights p/4. Do not confuse this p with a convention assigning p/3 to each nonidentity Pauli.
- `postselect-zero` is CP and trace nonincreasing, with an unnormalized output. It is not a TP channel. Transpose is the positive/TP but non-CP counterexample; the Bell output exposes the failure.

## Environment and instruments

The environment model is a damping dilation with output order B,E. It reports a4×2 isometry, one4×4 unitary completion, their exact residuals, the joint state and both reductions. It also compares plus/minus environment-basis Kraus operators and a reference-system output. This particular environment qubit does not represent every channel's minimal dilation.

An instrument has two recorded outcomes, each with one or more weighted operators. Each branch must be CP and the sum TP. The report retains probability, effect and unnormalized output; it exposes a conditional state only for strictly positive probability. The classical–quantum state orders record before system. A multi-operator coarse outcome is included in references. Optional outcome1 correction applies X. It is a specified feedback operation, not a claim of universal error correction.

The deferred-measurement example orders system before record. Two controlled permutations transfer the input into a coherent record while resetting the system. The model reports that coherent state, the final record-dephased state and its reordered classical–quantum form. Agreement after final dephasing does not imply equality before dephasing.

## Independence and publication gates

Product matrices use matrix multiplication, Laplace determinants and all principal minors. Validators use direct indexed contractions, permutation determinants and Schur PSD checks. They share exact field/scalar arithmetic and wire conversion, and are independently compared with SymPy. The validator has a type-only reference to product interfaces; a browser bundle graph verifies that the product channel module is absent. Strict response comparisons reject extra/missing fields, changed flags and mathematically unrelated values even when their scalar format is valid.

This is a typed calculation boundary, **not** a user JSON decoder or Worker lifecycle protocol. Before registration, implement input budgets, the circuit observation/validation layer, calculation/validation Workers, cancellation/retry, teaching surfaces, bilingual explanations of the parameter conventions, accessibility and actual production reader checks. Keep heavy arithmetic outside the main-thread reader.

Source check: [Watrous, characterizations of channels](https://cs.uwaterloo.ca/~watrous/TQI-notes/TQI-notes.05.pdf) supports the finite CP/TP criteria. Our factor order is explicitly reference⊗output and must be translated when a source uses output⊗reference. The exact finite examples and cross-check implementations are local calculations.
