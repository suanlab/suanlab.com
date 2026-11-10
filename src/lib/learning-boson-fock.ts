/** Finite projected boson algebra; no claim of an exact finite-dimensional CCR. */
import { f, type F } from './learning-quantum-hilbert-core';
import { bReal as r, bZero as zero, bOne as one, bAdd as add, bSub as sub, bMul as mul, bDiv as divide, bSign as sign, bEqual as equal, bObserve as q, bComplex as c, bCAdd as ca, bCSub as cs, bCMul as cm, bConjugate as conjugate, bScale as scale, bAbs2 as abs2, bSum as sum, bNorm2 as norm2, bDot as dot, bCZero as isZero, bCObserve as cq, bNormalized as normalized, bIdentity as identity, bAdjoint as adjoint, bMSub as mSub, bProduct as product, bApply as apply, bMatrixObserve as mq, type BosonReal as R, type BosonComplex as C, type BosonMatrix as M } from './learning-boson-algebraic';
import { snapshotBosonFockParameters, type BosonFockParameters } from './learning-boson-fock-input';
export { createBosonFockParameters, snapshotBosonFockParameters, isBosonFockParameters, bosonFockParameterKeys, type BosonFockParameters } from './learning-boson-fock-input';
const int = (n: number) => r(f(BigInt(n)));
const real = (v: C): R => { if (sign(v.im) !== 0) throw new RangeError('Real observable expected'); return v.re; };
const frobenius = (a: M) => norm2(a.flat());
const matrixScale = (a: M, x: R) => a.map(row => row.map(v => scale(v, x)));
const trace = (a: M) => a.reduce((s, row, i) => ca(s, row[i]), c(zero));
function evaluateBosonFockState(value: BosonFockParameters, supplied?: readonly C[]) {
  const parameters = snapshotBosonFockParameters(value), p = parameters, modes = p.modes;
  const cutoffs = [p.cutoff0, ...(modes === 2 ? [p.cutoff1] : [])], localDimensions = cutoffs.map(x => x + 1), dimension = localDimensions.reduce((a, b) => a * b, 1);
  const basisOccupations = Array.from({ length: dimension }, (_, index) => modes === 1 ? [index] : [Math.floor(index / localDimensions[1]), index % localDimensions[1]]);
  if (supplied && supplied.length !== dimension) throw new RangeError('Prepared state dimension mismatch');
  const state = supplied ? supplied.map(v => c([...v.re] as unknown as R, [...v.im] as unknown as R)) : basisOccupations.map((_, i) => c(int(p[`stateReal${i}`]), int(p[`stateImag${i}`]))), N = norm2(state);
  if (N.slice(1).some(x => x.n !== BigInt(0)) || sign(N) <= 0) throw new RangeError('Positive rational preparation norm required');
  const norm: F = N[0], D = int(p.hDenominator), divideN = (v: C) => scale(v, divide(one, N));
  const empty = (): M => Array.from({ length: dimension }, () => Array.from({ length: dimension }, () => c(zero)));
  const I = identity(dimension), roots = [zero, one, r(f(BigInt(0)), f(BigInt(1))), r(f(BigInt(0)), f(BigInt(0)), f(BigInt(1)))];
  const annihilation = cutoffs.map((_, mode) => {
    const a = empty(), step = mode === 0 && modes === 2 ? localDimensions[1] : 1;
    basisOccupations.forEach((occupation, column) => { if (occupation[mode] > 0) a[column - step][column] = c(roots[occupation[mode]]); });
    return a;
  }), creation = annihilation.map(adjoint), number = annihilation.map((a, i) => product(creation[i], a));
  const expectation = (a: M) => divideN(dot(state, apply(a, state)));
  const coefficient = (key: string) => divide(int(p[key]), D);
  type Term = { type: string; modes: number[]; coefficient: C; operator: M; paired: boolean };
  const terms: Term[] = [];
  const put = (type: string, indices: number[], coefficient: C, operator: M, paired = false) => { if (!isZero(coefficient)) terms.push({ type, modes: indices, coefficient, operator, paired }); };
  put('offset', [], c(coefficient('offset')), I);
  for (let i = 0; i < modes; i++) {
    put('onsite', [i], c(coefficient(`onsite${i}`)), number[i]);
    put('selfDensity', [i], c(coefficient(`selfDensity${i}`)), matrixScale(product(number[i], mSub(number[i], I)), divide(one, int(2))));
    put('drive', [i], c(coefficient(`driveReal${i}`), coefficient(`driveImag${i}`)), creation[i], true);
  }
  if (modes === 2) {
    put('hopping', [0, 1], c(coefficient('hoppingReal'), coefficient('hoppingImag')), product(creation[0], annihilation[1]), true);
    put('crossDensity', [0, 1], c(coefficient('crossDensity')), product(number[0], number[1]));
  }
  const H = empty();
  for (const term of terms) for (let i = 0; i < dimension; i++) for (let j = 0; j < dimension; j++) {
    H[i][j] = ca(H[i][j], cm(term.coefficient, term.operator[i][j]));
    if (term.paired) H[i][j] = ca(H[i][j], cm(conjugate(term.coefficient), conjugate(term.operator[j][i])));
  }
  const action = apply(H, state), energy = real(divideN(dot(state, action))), secondMoment = divide(norm2(action), N);
  const residual = action.map((v, i) => cs(v, scale(state[i], energy))), variance = divide(norm2(residual), N);
  if (sign(variance) < 0 || !equal(variance, sub(secondMoment, mul(energy, energy)))) throw new RangeError('Variance account failed');
  const density = state.map(a => state.map(b => divideN(cm(a, conjugate(b)))));
  const termValues = terms.map(t => {
    const mean = expectation(t.operator), weighted = cm(t.coefficient, mean);
    return { type: t.type, modes: t.modes, coefficient: cq(t.coefficient), expectation: cq(mean), includesAdjoint: t.paired, energyContribution: q(mul(int(t.paired ? 2 : 1), real(t.paired ? c(weighted.re) : weighted))) };
  });
  const commutator = (a: M, b: M) => mSub(product(a, b), product(b, a));
  const totalOccupations = basisOccupations.map(ns => ns.reduce((a, b) => a + b, 0));
  const totalNumberOperator = I.map((row, i) => row.map((_, j) => c(i === j ? int(totalOccupations[i]) : zero)));
  const totalMean = real(expectation(totalNumberOperator)), totalVariance = sub(real(expectation(product(totalNumberOperator, totalNumberOperator))), mul(totalMean, totalMean));
  const totalComm = frobenius(commutator(H, totalNumberOperator));
  const commutation = [];
  for (let i = 0; i < modes; i++) for (let j = 0; j < modes; j++) {
    const mixed = commutator(annihilation[i], creation[j]);
    const projectedExpected = I.map((row, a) => row.map((_, b) => c(i === j && a === b ? int(1 - (cutoffs[i] + 1) * Number(basisOccupations[a][i] === cutoffs[i])) : zero)));
    commutation.push({ leftMode: i, rightMode: j, mixed: mq(mixed), projectedIdentityResidualFrobeniusSquared: q(frobenius(mSub(mixed, projectedExpected))), infiniteCcrResidualFrobeniusSquared: q(frobenius(mSub(mixed, matrixScale(I, int(Number(i === j)))))), annihilationCommutatorFrobeniusSquared: q(frobenius(commutator(annihilation[i], annihilation[j]))), creationCommutatorFrobeniusSquared: q(frobenius(commutator(creation[i], creation[j]))), trace: cq(trace(mixed)) });
  }
  const invRoot2 = r(f(BigInt(0)), f(BigInt(1), BigInt(2)));
  const localStates = cutoffs.map((cutoff, mode) => {
    const top = I.map((row, i) => row.map((_, j) => c(i === j && basisOccupations[i][mode] === cutoff ? one : zero)));
    const topProbability = real(expectation(top)), mixed = commutator(annihilation[mode], creation[mode]);
    const X = annihilation[mode].map((row, i) => row.map((v, j) => scale(ca(v, creation[mode][i][j]), invRoot2)));
    const P = annihilation[mode].map((row, i) => row.map((v, j) => cm(c(zero, mul(int(-1), invRoot2)), cs(v, creation[mode][i][j]))));
    const xm = real(expectation(X)), pm = real(expectation(P)), xv = sub(real(expectation(product(X, X))), mul(xm, xm)), pv = sub(real(expectation(product(P, P))), mul(pm, pm));
    const covariance = sub(expectation(product(X, P)).re, mul(xm, pm));
    // Re <XP> equals the symmetrized product for Hermitian X,P.
    const commMean = real(expectation(mixed)), bound = divide(mul(commMean, commMean), int(4));
    const determinant = sub(mul(xv, pv), mul(covariance, covariance));
    const reduced = Array.from({ length: cutoff + 1 }, (_, i) => Array.from({ length: cutoff + 1 }, (_, j) => {
      let value = c(zero);
      for (let a = 0; a < dimension; a++) if (basisOccupations[a][mode] === i) for (let b = 0; b < dimension; b++) if (basisOccupations[b][mode] === j && basisOccupations[a].every((n, k) => k === mode || n === basisOccupations[b][k])) value = ca(value, density[a][b]);
      return value;
    }));
    return { mode, cutoff, density: mq(reduced), purity: q(frobenius(reduced)), occupation: q(real(expectation(number[mode]))), annihilationMean: cq(expectation(annihilation[mode])), topProbability: q(topProbability), finiteCommutatorExpectation: q(commMean), boundaryCorrection: q(mul(int(cutoff + 1), topProbability)), quadratures: { xMean: q(xm), pMean: q(pm), xVariance: q(xv), pVariance: q(pv), covariance: q(covariance), robertsonBound: q(bound), covarianceDeterminant: q(determinant), uncertaintySlack: q(sub(determinant, bound)) } };
  });
  const L = p.leftCreation ? creation[p.leftMode] : annihilation[p.leftMode], R = p.rightCreation ? creation[p.rightMode] : annihilation[p.rightMode];
  const forward = apply(product(L, R), state), reverse = apply(product(R, L), state);
  const probeComm = commutator(L, R);
  const isDiagonal = H.every((row, i) => row.every((v, j) => i === j || isZero(v)));
  let diagonalSpectrum = null;
  if (isDiagonal) {
    const groups: { energy: R; indices: number[]; weight: R }[] = [];
    H.forEach((row, i) => {
      const e = real(row[i]); let group = groups.find(g => equal(g.energy, e));
      if (!group) { group = { energy: e, indices: [], weight: zero }; groups.push(group); }
      group.indices.push(i); group.weight = add(group.weight, abs2(state[i]));
    });
    groups.sort((a, b) => sign(sub(a.energy, b.energy)));
    const ground = groups[0], gap = groups.length > 1 ? sub(groups[1].energy, ground.energy) : null;
    const weight = divide(ground.weight, N), ratio = gap === null ? null : divide(sub(energy, ground.energy), gap);
    diagonalSpectrum = { scope: 'finite-projected-Fock-space-only', levels: groups.map(g => ({ energy: q(g.energy), basisIndices: g.indices, multiplicity: g.indices.length, probability: q(divide(g.weight, N)) })), groundEnergy: q(ground.energy), groundMultiplicity: ground.indices.length, groundProbability: q(weight), leakage: q(sub(one, weight)), excitationGap: gap === null ? null : q(gap), excessEnergyToGapRatio: ratio === null ? null : q(ratio), certifiedLeakageUpperBound: ratio === null ? null : q(sign(sub(ratio, one)) > 0 ? one : ratio), selectedStateGround: equal(weight, one) };
  }
  const gamma = annihilation.map((_, i) => annihilation.map(a => cq(expectation(product(creation[i], a)))));
  const boundaryNormResiduals = localStates.map(s => sub(real(expectation(commutator(annihilation[s.mode], creation[s.mode]))), sub(one, mul(int(s.cutoff + 1), real(expectation(I.map((row, i) => row.map((_, j) => c(i === j && basisOccupations[i][s.mode] === s.cutoff ? one : zero)))))))));
  return {
    parameters, modes, cutoffs, localDimensions, dimension, basisOccupations, arithmetic: 'exact-complex-Q-sqrt2-sqrt3', normalizationPrecisionBits: 512,
    state: state.map(cq), normalizedState: state.map(v => normalized(v, norm)), rawNormSquared: q(N), hamiltonian: mq(H), hamiltonianAction: action.map(cq), energy: q(energy), secondMoment: q(secondMoment), variance: q(variance), residual: residual.map(cq), normalizedResidual: residual.map(v => normalized(v, norm)), terms: termValues,
    density: mq(density), localStates, gamma,
    numberDistribution: Array.from({ length: cutoffs.reduce((a, b) => a + b, 0) + 1 }, (_, n) => q(divide(sum(state.map((v, i) => totalOccupations[i] === n ? abs2(v) : zero)), N))),
    totalNumber: { expectation: q(totalMean), variance: q(totalVariance), selectedStateSharp: sign(totalVariance) === 0, conserved: sign(totalComm) === 0, commutatorFrobeniusSquared: q(totalComm) },
    operators: annihilation.map((a, mode) => ({ mode, annihilation: mq(a) })), commutation,
    probe: { forward: forward.map(cq), reverse: reverse.map(cq), commutatorAction: forward.map((v, i) => cq(cs(v, reverse[i]))), forwardOnNormalizedInput: forward.map(v => normalized(v, norm)), reverseOnNormalizedInput: reverse.map(v => normalized(v, norm)), forwardSquaredNormOnNormalizedInput: q(divide(norm2(forward), N)), reverseSquaredNormOnNormalizedInput: q(divide(norm2(reverse), N)), commutator: mq(probeComm) },
    spectralMode: isDiagonal ? 'finite-occupation-basis-enumeration' : 'not-computed-for-nondiagonal-hamiltonian', diagonalSpectrum,
    classification: { exactEigenstate: sign(variance) === 0, selectedStateFiniteCutoffGround: diagonalSpectrum?.selectedStateGround ?? null },
    exactResiduals: { hermiticityFrobeniusSquared: q(frobenius(mSub(H, adjoint(H)))), varianceAccount: q(sub(variance, sub(secondMoment, mul(energy, energy)))), residualOrthogonality: cq(dot(state, residual)), energyAccount: q(sub(energy, sum(terms.map(t => mul(int(t.paired ? 2 : 1), real(t.paired ? c(cm(t.coefficient, expectation(t.operator)).re) : cm(t.coefficient, expectation(t.operator)))))))), boundaryAccount: boundaryNormResiduals.map(q) },
    approximationContract: '1..2 ordered boson modes,each maximum occupancy0..3;exact projected operators in Q(sqrt2,sqrt3),Gaussian-integer nonzero preparation,52 owned bounded fields. No exact finite-dimensional infinite CCR;[a_i,a_i†]=I-(cutoff_i+1)P_top. Quadratures and uncertainty bounds use this finite commutator. Energies and spectra refer only to the finite projected Hamiltonian. No infinite-space ground certificate,time evolution,coherent-state tail certificate or convergence claim is supplied by this raw-state model.',
  };
}

export function bosonFockModel(value: BosonFockParameters) { return evaluateBosonFockState(value); }

/** Internal bridge for exact coherent preparation. Finite input is still validated;
 * callers receive no fabricated Gaussian-integer preparation parameters. */
export function evaluateBosonFockPreparedState(value: BosonFockParameters, state: readonly C[]) {
  const cloneReal = (v: R): R => {
    if (!Array.isArray(v) || v.length !== 4) throw new RangeError('Four exact field coordinates required');
    return v.map(x => {
      if (!x || typeof x.n !== 'bigint' || typeof x.d !== 'bigint' || x.d <= BigInt(0) || x.n.toString(2).length > 4096 || x.d.toString(2).length > 4096) throw new RangeError('Bounded exact rational coordinate required');
      return f(x.n, x.d);
    }) as unknown as R;
  };
  if (!Array.isArray(state) || state.length > 16) throw new RangeError('At most16 prepared amplitudes required');
  const cloned = state.map(v => c(cloneReal(v.re), cloneReal(v.im)));
  const result = evaluateBosonFockState(value, cloned);
  // The raw-state control object supplies the Hamiltonian and dimensions only.
  const { parameters: rawControls, ...observations } = result;
  void rawControls;
  return { ...observations, approximationContract: 'Exact field-valued preparation in a finite projected boson space;all observables and ground statements concern this cutoff Hamiltonian. Input preparation and infinite-space comparison are supplied by the coherent wrapper;no Gaussian-integer preparation is implied.' };
}
