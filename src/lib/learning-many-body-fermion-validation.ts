/** Fermion response attribution with JW Kronecker matrices and dense traces.
 * The production occupation-bit engine and supplied zero residuals are not an
 * oracle for the request's signs, adjoints, parity or sector interpretation.
 */
import { readManyBodyInput, manyBodyInputKey, type ManyBodyInput } from './learning-many-body-input';
import type { ManyBodySuccess } from './learning-many-body-protocol';
import { snapshotManyBodyWire } from './learning-many-body-owned-wire';
import { fermionFockResultContract } from './learning-many-body-contracts';
import { expectedRational as q, expectedComplex as cq, expectedNormalized as normalized, matchManyBodyExpected, type ManyBodyExpected as Expected } from './learning-many-body-wire-match';
import { int, zero, one, two, sub, mul, div, sum, eq, le, c, ca, cc, cm, cs, scale, abs2, norm2, identity, adjoint, tensor, matrixAdd, matrixSub, matrixScale, matrixNorm2, type C, type F, type Matrix } from './learning-tensor-wire-math';

const nonzero = (z: C) => z.re.n !== BigInt(0) || z.im.n !== BigInt(0);
const empty = (n: number): Matrix => Array.from({ length: n }, () => Array.from({ length: n }, () => c(zero)));
const observedMatrix = (a: Matrix): Expected => a.map(row => row.map(cq));
function multiply(a: Matrix, b: Matrix): Matrix {
  const result = empty(a.length);
  for (let i = 0; i < a.length; i++) for (let k = 0; k < a.length; k++) if (nonzero(a[i][k])) {
    for (let j = 0; j < a.length; j++) if (nonzero(b[k][j])) result[i][j] = ca(result[i][j], cm(a[i][k], b[k][j]));
  }
  return result;
}
const apply = (a: Matrix, z: C[]) => a.map(row => row.reduce((s, v, j) => nonzero(v) ? ca(s, cm(v, z[j])) : s, c(zero)));
const dot = (a: C[], b: C[]) => a.reduce((s, v, i) => ca(s, cm(cc(v), b[i])), c(zero));
const traceProduct = (a: Matrix, b: Matrix) => a.reduce((s, row, i) => row.reduce((s, v, j) => nonzero(v) && nonzero(b[j][i]) ? ca(s, cm(v, b[j][i])) : s, s), c(zero));

function fermionExpected(p: Readonly<Record<string, number>>): Expected {
  const modes = p.modes, dimension = 2 ** modes, denominator = int(p.hDenominator), I = identity(dimension);
  const localI = identity(2), Z: Matrix = [[c(one), c(zero)], [c(zero), c(int(-1))]], a: Matrix = [[c(zero), c(one)], [c(zero), c(zero)]];
  const ladder = (mode: number, signed: boolean) => Array.from({ length: modes }, (_, i) => i === mode ? a : signed && i < mode ? Z : localI)
    .reduce((m, factor) => tensor(m, factor), [[c(one)]]);
  const annihilators = Array.from({ length: modes }, (_, i) => ladder(i, true)), creators = annihilators.map(adjoint);
  const naive = Array.from({ length: modes }, (_, i) => ladder(i, false)), naiveCreators = naive.map(adjoint);
  const numbers = annihilators.map((a, i) => multiply(creators[i], a));
  const state = Array.from({ length: dimension }, (_, i) => c(int(p[`stateReal${i}`]), int(p[`stateImag${i}`]))), N = norm2(state);
  const expectation = (op: Matrix) => scale(dot(state, apply(op, state)), div(one, N));
  type Term = { type: string; modes: number[]; coefficient: C; operator: Matrix; naive: Matrix; paired: boolean };
  const terms: Term[] = [], put = (term: Term) => { if (nonzero(term.coefficient)) terms.push(term); };
  const coefficient = (prefix: string, i: number, j: number) => c(div(int(p[`${prefix}Real${i}${j}`]), denominator), div(int(p[`${prefix}Imag${i}${j}`]), denominator));
  put({ type: 'offset', modes: [], coefficient: c(div(int(p.offset), denominator)), operator: I, naive: I, paired: false });
  for (let i = 0; i < modes; i++) put({ type: 'onsite', modes: [i], coefficient: c(div(int(p[`onsite${i}`]), denominator)), operator: numbers[i], naive: numbers[i], paired: false });
  for (let i = 0; i < modes; i++) for (let j = i + 1; j < modes; j++) {
    put({ type: 'hopping', modes: [i, j], coefficient: coefficient('hopping', i, j), operator: multiply(creators[i], annihilators[j]), naive: multiply(naiveCreators[i], naive[j]), paired: true });
    put({ type: 'pairing', modes: [i, j], coefficient: coefficient('pairing', i, j), operator: multiply(creators[i], creators[j]), naive: multiply(naiveCreators[i], naiveCreators[j]), paired: true });
    const density = multiply(numbers[i], numbers[j]);
    put({ type: 'density', modes: [i, j], coefficient: c(div(int(p[`density${i}${j}`]), denominator)), operator: density, naive: density, paired: false });
  }
  function hamiltonian(wrongSigns: boolean): Matrix {
    const result = empty(dimension);
    for (const t of terms) {
      const operator = wrongSigns ? t.naive : t.operator;
      for (let i = 0; i < dimension; i++) for (let j = 0; j < dimension; j++) if (nonzero(operator[i][j])) {
        const value = cm(t.coefficient, operator[i][j]); result[i][j] = ca(result[i][j], value);
        if (t.paired) result[j][i] = ca(result[j][i], cc(value));
      }
    }
    return result;
  }
  const H = hamiltonian(false), wrongH = hamiltonian(true);
  const moments = (H: Matrix) => {
    const action = apply(H, state), energy = div(dot(state, action).re, N), second = div(norm2(action), N);
    const residual = action.map((v, i) => cs(v, scale(state[i], energy)));
    return { action, energy, second, variance: sub(second, mul(energy, energy)), residual };
  };
  const m = moments(H), wrong = moments(wrongH);
  const labels = Array.from({ length: dimension }, (_, i) => i.toString(2).padStart(modes, '0'));
  const occupations = labels.map(label => [...label].filter(bit => bit === '1').length), parity = occupations.map(n => n % 2 ? -1 : 1);
  const density = state.map(a => state.map(b => scale(cm(a, cc(b)), div(one, N))));
  const dephased = density.map((row, i) => row.map((v, j) => parity[i] === parity[j] ? v : c(zero)));
  const gamma = creators.map(left => annihilators.map(right => expectation(multiply(left, right))));
  const anomalous = annihilators.map(left => annihilators.map(right => expectation(multiply(left, right))));
  const modeOccupations = numbers.map(op => expectation(op).re), even = div(sum(state.map((v, i) => parity[i] === 1 ? abs2(v) : zero)), N);
  const numberCorrelations: Expected[] = [];
  for (let i = 0; i < modes; i++) for (let j = i + 1; j < modes; j++) {
    const joint = expectation(multiply(numbers[i], numbers[j])).re;
    numberCorrelations.push({ modes: [i, j], joint: q(joint), connected: q(sub(joint, mul(modeOccupations[i], modeOccupations[j]))) });
  }
  const symmetry = (values: number[]) => {
    const commutator = H.map((row, i) => row.map((v, j) => scale(v, int(values[j] - values[i])))), squared = matrixNorm2(commutator);
    const mean = div(sum(state.map((v, i) => mul(abs2(v), int(values[i])))), N);
    const variance = sub(div(sum(state.map((v, i) => mul(abs2(v), int(values[i] ** 2)))), N), mul(mean, mean));
    return { conserved: squared.n === BigInt(0), commutatorFrobeniusSquared: q(squared), expectation: q(mean), variance: q(variance), selectedStateSharp: variance.n === BigInt(0) };
  };
  const l = p.leftCreation ? creators[p.leftMode] : annihilators[p.leftMode], r = p.rightCreation ? creators[p.rightMode] : annihilators[p.rightMode];
  const nl = p.leftCreation ? naiveCreators[p.leftMode] : naive[p.leftMode], nr = p.rightCreation ? naiveCreators[p.rightMode] : naive[p.rightMode];
  const forward = apply(multiply(l, r), state), reverse = apply(multiply(r, l), state), scalar = Number(p.leftMode === p.rightMode && p.leftCreation !== p.rightCreation);
  const car: Expected[] = [];
  for (let i = 0; i < modes; i++) for (let j = 0; j < modes; j++) car.push({
    leftMode: i, rightMode: j,
    mixedResidualFrobeniusSquared: q(matrixNorm2(matrixSub(matrixAdd(multiply(annihilators[i], creators[j]), multiply(creators[j], annihilators[i])), matrixScale(I, int(Number(i === j)))))),
    annihilationResidualFrobeniusSquared: q(matrixNorm2(matrixAdd(multiply(annihilators[i], annihilators[j]), multiply(annihilators[j], annihilators[i])))),
    creationResidualFrobeniusSquared: q(matrixNorm2(matrixAdd(multiply(creators[i], creators[j]), multiply(creators[j], creators[i])))),
  });
  const diagonal = H.every((row, i) => row.every((v, j) => i === j || !nonzero(v)));
  let diagonalSpectrum: Expected = null, selectedGround: boolean | null = null, selectedParityGround: boolean | null = null;
  if (diagonal) {
    const groups: { energy: F; indices: number[] }[] = [];
    for (let i = 0; i < dimension; i++) {
      const energy = H[i][i].re; let group = groups.find(g => eq(g.energy, energy));
      if (!group) { group = { energy, indices: [] }; groups.push(group); } group.indices.push(i);
    }
    groups.sort((a, b) => eq(a.energy, b.energy) ? 0 : le(a.energy, b.energy) ? -1 : 1);
    const probabilityOf = (indices: number[]) => div(sum(indices.map(i => abs2(state[i]))), N);
    const minimum = (filter: (index: number) => boolean) => {
      const levels = groups.map(g => ({ energy: g.energy, indices: g.indices.filter(filter) })).filter(g => g.indices.length);
      const first = levels[0], probability = probabilityOf(first.indices);
      return { energy: q(first.energy), multiplicity: first.indices.length, basisIndices: first.indices, probability: q(probability), excitationGap: levels.length > 1 ? q(sub(levels[1].energy, first.energy)) : null, selectedStateGround: eq(probability, one) };
    };
    const parityMinima = [0, 1].map(sector => ({ parity: sector === 0 ? 'even' : 'odd', ...minimum(i => occupations[i] % 2 === sector) }));
    const numberMinima = Array.from({ length: modes + 1 }, (_, number) => ({ number, ...minimum(i => occupations[i] === number) }));
    const ground = groups[0], probability = probabilityOf(ground.indices), gap = groups.length > 1 ? sub(groups[1].energy, ground.energy) : null;
    const ratio = gap === null ? null : div(sub(m.energy, ground.energy), gap); selectedGround = eq(probability, one);
    selectedParityGround = p.sector === 0 ? null : parityMinima[p.sector - 1].selectedStateGround;
    diagonalSpectrum = { basisEnergies: H.map((row, i) => q(row[i].re)),
      levels: groups.map(g => ({ energy: q(g.energy), basisIndices: g.indices, multiplicity: g.indices.length, probability: q(probabilityOf(g.indices)) })),
      groundEnergy: q(ground.energy), groundMultiplicity: ground.indices.length, groundProbability: q(probability), leakage: q(sub(one, probability)),
      excitationGap: gap === null ? null : q(gap), excessEnergyToGapRatio: ratio === null ? null : q(ratio), certifiedLeakageUpperBound: ratio === null ? null : q(le(ratio, one) ? ratio : one),
      selectedStateGround: selectedGround, scope: 'full-Fock-space-not-a-fixed-number-or-parity-sector', parityMinima, numberMinima, selectedParityGround };
  }
  const coherence = matrixNorm2(matrixSub(density, dephased));
  return {
    parameters: { ...p }, modes, dimension, basisConvention: 'n0-is-most-significant;ascending-creation-order;rightmost-operator-acts-first', basisLabels: labels, occupationCounts: occupations,
    arithmetic: 'exact-complex-rational', normalizationPrecisionBits: 512,
    hamiltonian: observedMatrix(H), state: state.map(cq), normalizedState: state.map(v => normalized(v, N)), rawNormSquared: q(N),
    hamiltonianAction: m.action.map(cq), energy: q(m.energy), secondMoment: q(m.second), variance: q(m.variance), residual: m.residual.map(cq), normalizedResidual: m.residual.map(v => normalized(v, N)),
    terms: terms.map(t => { const mean = expectation(t.operator), contribution = cm(t.coefficient, mean).re; return { type: t.type, modes: t.modes, coefficient: cq(t.coefficient), expectation: cq(mean), includesAdjoint: t.paired, energyContribution: q(t.paired ? mul(two, contribution) : contribution) }; }),
    density: observedMatrix(density), gamma: observedMatrix(gamma), anomalous: observedMatrix(anomalous), oddMeans: annihilators.map(op => cq(expectation(op))),
    modeOccupations: modeOccupations.map(q), numberDistribution: Array.from({ length: modes + 1 }, (_, n) => q(div(sum(state.map((v, i) => occupations[i] === n ? abs2(v) : zero)), N))),
    numberCorrelations, totalNumber: symmetry(occupations), fermionParity: symmetry(parity),
    parityAssumption: { declaredInput: p.sector === 0 ? 'formal-either-parity-allowed' : p.sector === 1 ? 'even-pure-sector' : 'odd-pure-sector', evenProbability: q(even), oddProbability: q(sub(one, even)), coherenceFrobeniusSquared: q(coherence), pureStateCompatibleWithParitySuperselection: coherence.n === BigInt(0) },
    parityDephased: { density: observedMatrix(dephased), purity: q(matrixNorm2(dephased)), energy: q(traceProduct(dephased, H).re), oddMeans: annihilators.map(op => cq(traceProduct(dephased, op))) },
    operators: annihilators.map((a, mode) => ({ mode, annihilationColumns: Array.from({ length: dimension }, (_, column) => { const row = a.findIndex(r => nonzero(r[column])); return { row: row < 0 ? null : row, sign: row < 0 ? 0 : a[row][column].re.n < BigInt(0) ? -1 : 1 }; }) })),
    car,
    probe: { expectedAnticommutatorScalar: scalar, forward: forward.map(cq), reverse: reverse.map(cq), anticommutatorAction: forward.map((v, i) => cq(ca(v, reverse[i]))),
      forwardOnNormalizedInput: forward.map(v => normalized(v, N)), reverseOnNormalizedInput: reverse.map(v => normalized(v, N)), forwardSquaredNormOnNormalizedInput: q(div(norm2(forward), N)), reverseSquaredNormOnNormalizedInput: q(div(norm2(reverse), N)),
      carResidualFrobeniusSquared: q(matrixNorm2(matrixSub(matrixAdd(multiply(l, r), multiply(r, l)), matrixScale(I, int(scalar))))),
      naiveCarResidualFrobeniusSquared: q(matrixNorm2(matrixSub(matrixAdd(multiply(nl, nr), multiply(nr, nl)), matrixScale(I, int(scalar))))),
    },
    naiveComparison: { energy: q(wrong.energy), variance: q(wrong.variance), hamiltonianDifferenceFrobeniusSquared: q(matrixNorm2(matrixSub(H, wrongH))) },
    spectralMode: diagonal ? 'occupation-basis-enumeration' : 'not-computed-for-nondiagonal-hamiltonian', diagonalSpectrum,
    classification: { exactEigenstate: m.variance.n === BigInt(0), selectedStateGround: selectedGround, selectedStateParitySectorGround: selectedParityGround },
    exactResiduals: { hermiticityFrobeniusSquared: q(zero), varianceAccount: q(zero), residualOrthogonality: cq(c(zero)), energyAccount: q(zero), gammaHermiticity: q(zero), anomalousAntisymmetry: q(zero), gammaTraceNumber: q(zero), parityTwirlEnergyAccount: q(zero) },
    approximationContract: fermionFockResultContract,
  };
}

export type ManyBodyFermionSuccess = Extract<ManyBodySuccess, { kind: 'fermion-fock' }>;
/** Validate all data and return the frozen snapshot, never the caller packet. */
export function readFermionFockSuccessForInput(value: unknown, original: ManyBodyInput, requestId: number): ManyBodyFermionSuccess | null {
  try {
    if (!Number.isSafeInteger(requestId) || requestId <= 0) return null;
    const input = readManyBodyInput(original);
    if (!input || input.kind !== 'fermion-fock') return null;
    const packet = snapshotManyBodyWire(value);
    const expected: Expected = { version: 1, requestId, ok: true, kind: 'fermion-fock', input: { kind: input.kind, parameters: { ...input.parameters } }, inputKey: manyBodyInputKey(input), result: fermionExpected(input.parameters) };
    return matchManyBodyExpected(packet, expected) ? packet as ManyBodyFermionSuccess : null;
  } catch { return null; }
}
