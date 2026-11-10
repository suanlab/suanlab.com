/** Raw boson response attribution by local ladders and Kronecker products.
 * No production Fock/biquadratic observation or interval engine is imported.
 */
import { readManyBodyInput, manyBodyInputKey, type ManyBodyInput } from './learning-many-body-input';
import type { ManyBodySuccess } from './learning-many-body-protocol';
import { snapshotManyBodyWire } from './learning-many-body-owned-wire';
import { bosonFockResultContract } from './learning-many-body-contracts';
import { matchManyBodyExpected, type ManyBodyExpected as Expected } from './learning-many-body-wire-match';
import { int, one, div } from './learning-tensor-wire-math';
import { field, fieldZero as Z, fieldOne as O, fieldInt as integer, fieldAdd as add, fieldSub as sub, fieldMul as mul, fieldScale, fieldDivide, fieldSign as sign, fieldEqual as equal, fieldRational, fieldSum as sum, expectedField as q, type WireField as R } from './learning-many-body-field-wire';
import { fc, fcZero, fcAdd, fcSub, fcMultiply, fcScale, fcAbsSquared, fieldNormSquared, fieldDot, fmEmpty, fmIdentity, fmAdjoint, fmAdd, fmSub, fmScale, fmProduct, fmTensor, fmApply, fmNormSquared, fmTrace, expectedFieldComplex as cq, expectedFieldMatrix as mq, expectedFieldNormalizedComplex as normalized, fieldHalf, type FieldComplex as C, type FieldMatrix as M } from './learning-many-body-field-matrix';

export function bosonFockExpected(p: Readonly<Record<string, number>>, supplied?: readonly C[]): Record<string, Expected> {
  const modes = p.modes, cutoffs = [p.cutoff0, ...(modes === 2 ? [p.cutoff1] : [])], dimensions = cutoffs.map(c => c + 1);
  let basis: number[][] = [[]];
  for (const cutoff of cutoffs) basis = basis.flatMap(prefix => Array.from({ length: cutoff + 1 }, (_, n) => [...prefix, n]));
  const dimension = basis.length, state = supplied ? [...supplied] : basis.map((_, i) => fc(integer(p[`stateReal${i}`]), integer(p[`stateImag${i}`])));
  if (state.length !== dimension) throw new RangeError('Prepared coefficient dimension mismatch');
  const norm = fieldRational(fieldNormSquared(state)), N = field(norm), inverseN = field(div(one, norm)), I = fmIdentity(dimension);
  const roots = [Z, O, field(int(0), int(1)), field(int(0), int(0), int(1))];
  const annihilation = cutoffs.map((cutoff, mode) => {
    const local = fmEmpty(cutoff + 1); for (let n = 1; n <= cutoff; n++) local[n - 1][n] = fc(roots[n]);
    return dimensions.map((d, i) => i === mode ? local : fmIdentity(d)).reduce(fmTensor, [[fc(O)]]);
  });
  const creation = annihilation.map(fmAdjoint), number = annihilation.map((a, i) => fmProduct(creation[i], a));
  const expectation = (op: M) => fcScale(fieldDot(state, fmApply(op, state)), inverseN);
  const real = (z: C): R => { if (sign(z.im)) throw new RangeError('Expected real field account'); return z.re; };
  const coefficient = (name: string) => field(div(int(p[name]), int(p.hDenominator)));
  type Term = { type: string; modes: number[]; coefficient: C; operator: M; paired: boolean };
  const terms: Term[] = [];
  const put = (type: string, indices: number[], coefficient: C, operator: M, paired = false) => { if (!fcZero(coefficient)) terms.push({ type, modes: indices, coefficient, operator, paired }); };
  put('offset', [], fc(coefficient('offset')), I);
  for (let i = 0; i < modes; i++) {
    put('onsite', [i], fc(coefficient(`onsite${i}`)), number[i]);
    put('selfDensity', [i], fc(coefficient(`selfDensity${i}`)), fmScale(fmProduct(number[i], fmSub(number[i], I)), fieldHalf));
    put('drive', [i], fc(coefficient(`driveReal${i}`), coefficient(`driveImag${i}`)), creation[i], true);
  }
  if (modes === 2) {
    put('hopping', [0, 1], fc(coefficient('hoppingReal'), coefficient('hoppingImag')), fmProduct(creation[0], annihilation[1]), true);
    put('crossDensity', [0, 1], fc(coefficient('crossDensity')), fmProduct(number[0], number[1]));
  }
  const H = fmEmpty(dimension);
  for (const t of terms) for (let i = 0; i < dimension; i++) for (let j = 0; j < dimension; j++) if (!fcZero(t.operator[i][j])) {
    const value = fcMultiply(t.coefficient, t.operator[i][j]); H[i][j] = fcAdd(H[i][j], value);
    if (t.paired) H[j][i] = fcAdd(H[j][i], fc(value.re, fieldScale(value.im, int(-1))));
  }
  const action = fmApply(H, state), energy = real(fcScale(fieldDot(state, action), inverseN));
  const second = mul(fieldNormSquared(action), inverseN), residual = action.map((v, i) => fcSub(v, fcScale(state[i], energy))), variance = sub(second, mul(energy, energy));
  const density = state.map(a => state.map(b => fcScale(fcMultiply(a, fc(b.re, fieldScale(b.im, int(-1)))), inverseN)));
  const commutator = (a: M, b: M) => fmSub(fmProduct(a, b), fmProduct(b, a));
  const commutation: Expected[] = [];
  for (let i = 0; i < modes; i++) for (let j = 0; j < modes; j++) {
    const mixed = commutator(annihilation[i], creation[j]);
    const expected = I.map((row, a) => row.map((_, b) => fc(a === b && i === j ? integer(1 - (cutoffs[i] + 1) * Number(basis[a][i] === cutoffs[i])) : Z)));
    commutation.push({ leftMode: i, rightMode: j, mixed: mq(mixed), projectedIdentityResidualFrobeniusSquared: q(fmNormSquared(fmSub(mixed, expected))), infiniteCcrResidualFrobeniusSquared: q(fmNormSquared(fmSub(mixed, fmScale(I, integer(Number(i === j)))))), annihilationCommutatorFrobeniusSquared: q(fmNormSquared(commutator(annihilation[i], annihilation[j]))), creationCommutatorFrobeniusSquared: q(fmNormSquared(commutator(creation[i], creation[j]))), trace: cq(fmTrace(mixed)) });
  }
  const inverseRootTwo = field(int(0), div(one, int(2)));
  const localStates = cutoffs.map((cutoff, mode) => {
    const top = I.map((row, i) => row.map((_, j) => fc(i === j && basis[i][mode] === cutoff ? O : Z)));
    const topProbability = real(expectation(top)), commMean = real(expectation(commutator(annihilation[mode], creation[mode])));
    const X = fmScale(fmAdd(annihilation[mode], creation[mode]), inverseRootTwo);
    const P = fmSub(annihilation[mode], creation[mode]).map(row => row.map(z => fcScale(fc(z.im, fieldScale(z.re, int(-1))), inverseRootTwo)));
    const xm = real(expectation(X)), pm = real(expectation(P));
    const xv = sub(real(expectation(fmProduct(X, X))), mul(xm, xm)), pv = sub(real(expectation(fmProduct(P, P))), mul(pm, pm));
    const covariance = sub(real(expectation(fmScale(fmAdd(fmProduct(X, P), fmProduct(P, X)), fieldHalf))), mul(xm, pm));
    const bound = fieldScale(mul(commMean, commMean), div(one, int(4))), determinant = sub(mul(xv, pv), mul(covariance, covariance));
    const reduced = fmEmpty(cutoff + 1);
    for (let a = 0; a < dimension; a++) for (let b = 0; b < dimension; b++) if (basis[a].every((value, i) => i === mode || value === basis[b][i])) {
      const i = basis[a][mode], j = basis[b][mode]; reduced[i][j] = fcAdd(reduced[i][j], density[a][b]);
    }
    return { mode, cutoff, density: mq(reduced), purity: q(fmNormSquared(reduced)), occupation: q(real(expectation(number[mode]))), annihilationMean: cq(expectation(annihilation[mode])), topProbability: q(topProbability), finiteCommutatorExpectation: q(commMean), boundaryCorrection: q(mul(integer(cutoff + 1), topProbability)), quadratures: { xMean: q(xm), pMean: q(pm), xVariance: q(xv), pVariance: q(pv), covariance: q(covariance), robertsonBound: q(bound), covarianceDeterminant: q(determinant), uncertaintySlack: q(sub(determinant, bound)) } };
  });
  const totals = basis.map(ns => ns.reduce((a, b) => a + b, 0)), probabilities = state.map(v => mul(fcAbsSquared(v), inverseN));
  const mean = sum(probabilities.map((v, i) => mul(v, integer(totals[i])))), totalVariance = sub(sum(probabilities.map((v, i) => mul(v, integer(totals[i] ** 2)))), mul(mean, mean));
  const totalComm = H.map((row, i) => row.map((v, j) => fcScale(v, integer(totals[j] - totals[i])))), commSquared = fmNormSquared(totalComm);
  const L = p.leftCreation ? creation[p.leftMode] : annihilation[p.leftMode], R = p.rightCreation ? creation[p.rightMode] : annihilation[p.rightMode];
  const forward = fmApply(fmProduct(L, R), state), reverse = fmApply(fmProduct(R, L), state);
  const diagonal = H.every((row, i) => row.every((v, j) => i === j || fcZero(v)));
  let diagonalSpectrum: Expected = null, selectedGround: boolean | null = null;
  if (diagonal) {
    const groups: { energy: R; indices: number[]; weight: R }[] = [];
    for (let i = 0; i < dimension; i++) {
      const energy = real(H[i][i]); let group = groups.find(g => equal(g.energy, energy));
      if (!group) { group = { energy, indices: [], weight: Z }; groups.push(group); }
      group.indices.push(i); group.weight = add(group.weight, fcAbsSquared(state[i]));
    }
    groups.sort((a, b) => sign(sub(a.energy, b.energy)));
    const ground = groups[0], weight = mul(ground.weight, inverseN), gap = groups.length > 1 ? sub(groups[1].energy, ground.energy) : null;
    const ratio = gap === null ? null : fieldDivide(sub(energy, ground.energy), gap); selectedGround = equal(weight, O);
    diagonalSpectrum = { scope: 'finite-projected-Fock-space-only', levels: groups.map(g => ({ energy: q(g.energy), basisIndices: g.indices, multiplicity: g.indices.length, probability: q(mul(g.weight, inverseN)) })), groundEnergy: q(ground.energy), groundMultiplicity: ground.indices.length, groundProbability: q(weight), leakage: q(sub(O, weight)), excitationGap: gap === null ? null : q(gap), excessEnergyToGapRatio: ratio === null ? null : q(ratio), certifiedLeakageUpperBound: ratio === null ? null : q(sign(sub(ratio, O)) > 0 ? O : ratio), selectedStateGround: selectedGround };
  }
  return {
    parameters: { ...p }, modes, cutoffs, localDimensions: dimensions, dimension, basisOccupations: basis, arithmetic: 'exact-complex-Q-sqrt2-sqrt3', normalizationPrecisionBits: 512,
    state: state.map(cq), normalizedState: state.map(v => normalized(v, norm)), rawNormSquared: q(N), hamiltonian: mq(H), hamiltonianAction: action.map(cq), energy: q(energy), secondMoment: q(second), variance: q(variance), residual: residual.map(cq), normalizedResidual: residual.map(v => normalized(v, norm)),
    terms: terms.map(t => { const mean = expectation(t.operator), weighted = fcMultiply(t.coefficient, mean); return { type: t.type, modes: t.modes, coefficient: cq(t.coefficient), expectation: cq(mean), includesAdjoint: t.paired, energyContribution: q(mul(integer(t.paired ? 2 : 1), t.paired ? weighted.re : real(weighted))) }; }),
    density: mq(density), localStates, gamma: annihilation.map((_, i) => annihilation.map(a => cq(expectation(fmProduct(creation[i], a))))),
    numberDistribution: Array.from({ length: cutoffs.reduce((a, b) => a + b, 0) + 1 }, (_, n) => q(sum(probabilities.filter((_, i) => totals[i] === n)))),
    totalNumber: { expectation: q(mean), variance: q(totalVariance), selectedStateSharp: sign(totalVariance) === 0, conserved: sign(commSquared) === 0, commutatorFrobeniusSquared: q(commSquared) },
    operators: annihilation.map((a, mode) => ({ mode, annihilation: mq(a) })), commutation,
    probe: { forward: forward.map(cq), reverse: reverse.map(cq), commutatorAction: forward.map((v, i) => cq(fcSub(v, reverse[i]))), forwardOnNormalizedInput: forward.map(v => normalized(v, norm)), reverseOnNormalizedInput: reverse.map(v => normalized(v, norm)), forwardSquaredNormOnNormalizedInput: q(mul(fieldNormSquared(forward), inverseN)), reverseSquaredNormOnNormalizedInput: q(mul(fieldNormSquared(reverse), inverseN)), commutator: mq(commutator(L, R)) },
    spectralMode: diagonal ? 'finite-occupation-basis-enumeration' : 'not-computed-for-nondiagonal-hamiltonian', diagonalSpectrum,
    classification: { exactEigenstate: sign(variance) === 0, selectedStateFiniteCutoffGround: selectedGround },
    exactResiduals: { hermiticityFrobeniusSquared: q(Z), varianceAccount: q(Z), residualOrthogonality: cq(fc(Z)), energyAccount: q(Z), boundaryAccount: cutoffs.map(() => q(Z)) },
    approximationContract: bosonFockResultContract,
  };
}
export type ManyBodyBosonSuccess = Extract<ManyBodySuccess, { kind: 'boson-fock' }>;
export function readBosonFockSuccessForInput(value: unknown, original: ManyBodyInput, requestId: number): ManyBodyBosonSuccess | null {
  try {
    if (!Number.isSafeInteger(requestId) || requestId <= 0) return null;
    const input = readManyBodyInput(original); if (!input || input.kind !== 'boson-fock') return null;
    const packet = snapshotManyBodyWire(value);
    const expected: Expected = { version: 1, requestId, ok: true, kind: 'boson-fock', input: { kind: input.kind, parameters: { ...input.parameters } }, inputKey: manyBodyInputKey(input), result: bosonFockExpected(input.parameters) };
    return matchManyBodyExpected(packet, expected) ? packet as ManyBodyBosonSuccess : null;
  } catch { return null; }
}
