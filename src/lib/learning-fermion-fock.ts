/** Exact occupation-space CAR and an even finite fermion Hamiltonian.
 * Formal parity-coherent inputs are labelled; they are not advertised as
 * physical pure states under the declared parity superselection assumption. */
import { c, ca, cc, cm, cs, scaleC, norm2, dot, f, q, cq, normalized, zero, one, two, add, sub, mul, div, sum, eq, type C, type F } from './learning-quantum-hilbert-core';
import { snapshotFermionFockParameters, fermionOccupationCount, type FermionFockParameters } from './learning-fermion-fock-input';
export { isFermionFockParameters, snapshotFermionFockParameters, createFermionFockParameters, fermionFockParameterKeys, type FermionFockParameters } from './learning-fermion-fock-input';

type Op = number[][];
type Matrix = C[][];
const int = (n: number) => f(BigInt(n));
const zeros = (d: number): Op => Array.from({ length: d }, () => Array<number>(d).fill(0));
const transpose = (a: Op): Op => a[0].map((_, j) => a.map(row => row[j]));
function product(a: Op, b: Op): Op {
  const d = a.length, result = zeros(d);
  for (let i = 0; i < d; i++) for (let k = 0; k < d; k++) if (a[i][k] !== 0) {
    for (let j = 0; j < d; j++) if (b[k][j] !== 0) result[i][j] += a[i][k] * b[k][j];
  }
  return result;
}
const plus = (a: Op, b: Op): Op => a.map((row, i) => row.map((v, j) => v + b[i][j]));
const apply = (a: Op, z: C[]) => a.map(row => row.reduce((s, v, i) => v === 0 ? s : ca(s, scaleC(z[i], int(v))), c(zero)));
const applyComplex = (a: Matrix, z: C[]) => a.map(row => row.reduce((s, v, i) => ca(s, cm(v, z[i])), c(zero)));
const matrixObservation = (a: Matrix) => a.map(row => row.map(cq));
const zeroMatrix = (d: number): Matrix => Array.from({ length: d }, () => Array.from({ length: d }, () => c(zero)));
const frobenius = (a: Matrix) => norm2(a.flat());
const opSquared = (a: Op) => int(a.flat().reduce((s, v) => s + v * v, 0));
function ladder(modes: number, mode: number, creation: boolean, signed = true): Op {
  const d = 2 ** modes, result = zeros(d), mask = 2 ** (modes - mode - 1);
  for (let column = 0; column < d; column++) {
    if (((column & mask) !== 0) === creation) continue;
    const preceding = fermionOccupationCount(Math.floor(column / (2 * mask)));
    result[column ^ mask][column] = signed && preceding % 2 ? -1 : 1;
  }
  return result;
}

export function fermionFockModel(value: FermionFockParameters) {
  const parameters = snapshotFermionFockParameters(value), p = parameters, modes = p.modes, dimension = 2 ** modes, denominator = int(p.hDenominator);
  const state = Array.from({ length: dimension }, (_, i) => c(int(p[`stateReal${i}`]), int(p[`stateImag${i}`]))), N = norm2(state);
  const operators = Array.from({ length: modes }, (_, i) => ladder(modes, i, false)), creators = operators.map(transpose);
  const naiveOperators = Array.from({ length: modes }, (_, i) => ladder(modes, i, false, false)), naiveCreators = naiveOperators.map(transpose);
  const identity = zeros(dimension).map((row, i) => row.map((_, j) => Number(i === j)));
  const numberOperators = operators.map((a, i) => product(creators[i], a));
  const expectation = (a: Op): C => scaleC(dot(state, apply(a, state)), div(one, N));
  const real = (z: C): F => { if (z.im.n !== BigInt(0)) throw new RangeError('Real observable expected'); return z.re; };
  const car = [];
  for (let i = 0; i < modes; i++) for (let j = 0; j < modes; j++) {
    const mixed = plus(product(operators[i], creators[j]), product(creators[j], operators[i]));
    if (i === j) for (let k = 0; k < dimension; k++) mixed[k][k] -= 1;
    car.push({ leftMode: i, rightMode: j,
      mixedResidualFrobeniusSquared: q(opSquared(mixed)),
      annihilationResidualFrobeniusSquared: q(opSquared(plus(product(operators[i], operators[j]), product(operators[j], operators[i])))),
      creationResidualFrobeniusSquared: q(opSquared(plus(product(creators[i], creators[j]), product(creators[j], creators[i])))),
    });
  }
  type Term = { type: 'offset' | 'onsite' | 'hopping' | 'pairing' | 'density'; modes: number[]; coefficient: C; operator: Op; naive: Op; paired: boolean };
  const terms: Term[] = [];
  const coefficient = (prefix: string, i: number, j: number) => c(div(int(p[`${prefix}Real${i}${j}`]), denominator), div(int(p[`${prefix}Imag${i}${j}`]), denominator));
  const put = (t: Term) => { if (t.coefficient.re.n !== BigInt(0) || t.coefficient.im.n !== BigInt(0)) terms.push(t); };
  put({ type: 'offset', modes: [], coefficient: c(div(int(p.offset), denominator)), operator: identity, naive: identity, paired: false });
  for (let i = 0; i < modes; i++) put({ type: 'onsite', modes: [i], coefficient: c(div(int(p[`onsite${i}`]), denominator)), operator: numberOperators[i], naive: numberOperators[i], paired: false });
  for (let i = 0; i < modes; i++) for (let j = i + 1; j < modes; j++) {
    put({ type: 'hopping', modes: [i, j], coefficient: coefficient('hopping', i, j), operator: product(creators[i], operators[j]), naive: product(naiveCreators[i], naiveOperators[j]), paired: true });
    put({ type: 'pairing', modes: [i, j], coefficient: coefficient('pairing', i, j), operator: product(creators[i], creators[j]), naive: product(naiveCreators[i], naiveCreators[j]), paired: true });
    const density = product(numberOperators[i], numberOperators[j]);
    put({ type: 'density', modes: [i, j], coefficient: c(div(int(p[`density${i}${j}`]), denominator)), operator: density, naive: density, paired: false });
  }
  function hamiltonian(naive: boolean): Matrix {
    const matrix = zeroMatrix(dimension);
    for (const t of terms) {
      const a = naive ? t.naive : t.operator;
      for (let i = 0; i < dimension; i++) for (let j = 0; j < dimension; j++) {
        matrix[i][j] = ca(matrix[i][j], scaleC(t.coefficient, int(a[i][j])));
        if (t.paired) matrix[i][j] = ca(matrix[i][j], scaleC(cc(t.coefficient), int(a[j][i])));
      }
    }
    return matrix;
  }
  const H = hamiltonian(false), naiveH = hamiltonian(true);
  function moments(h: Matrix) {
    const action = applyComplex(h, state), energy = real(scaleC(dot(state, action), div(one, N)));
    const second = div(norm2(action), N), residual = action.map((v, i) => cs(v, scaleC(state[i], energy)));
    const variance = div(norm2(residual), N);
    if (variance.n < BigInt(0) || !eq(variance, sub(second, mul(energy, energy)))) throw new RangeError('Hermitian variance certificates disagree');
    return { action, energy, second, residual, variance };
  }
  const m = moments(H), naiveMoments = moments(naiveH);
  const density = state.map(a => state.map(b => scaleC(cm(a, cc(b)), div(one, N))));
  const occupations = Array.from({ length: dimension }, (_, i) => fermionOccupationCount(i));
  const parity = occupations.map(n => n % 2 ? -1 : 1);
  const dephased = density.map((row, i) => row.map((v, j) => parity[i] === parity[j] ? v : c(zero)));
  const traceProduct = (a: Matrix, b: Matrix) => a.reduce((s, row, i) => row.reduce((t, v, j) => ca(t, cm(v, b[j][i])), s), c(zero));
  const opAsComplex = (a: Op) => a.map(row => row.map(v => c(int(v))));
  const gamma = operators.map((_, i) => operators.map(a => expectation(product(creators[i], a))));
  const anomalous = operators.map(a => operators.map(b => expectation(product(a, b))));
  const oddMeans = operators.map(expectation);
  const modeOccupations = numberOperators.map(a => real(expectation(a)));
  const numberDistribution = Array.from({ length: modes + 1 }, (_, n) => q(sum(state.map((v, i) => occupations[i] === n ? div(norm2([v]), N) : zero))));
  const evenProbability = sum(state.map((v, i) => parity[i] === 1 ? div(norm2([v]), N) : zero)), oddProbability = sub(one, evenProbability);
  const numberCorrelations = [];
  for (let i = 0; i < modes; i++) for (let j = i + 1; j < modes; j++) {
    const joint = real(expectation(product(numberOperators[i], numberOperators[j])));
    numberCorrelations.push({ modes: [i, j], joint: q(joint), connected: q(sub(joint, mul(modeOccupations[i], modeOccupations[j]))) });
  }
  function symmetry(values: number[]) {
    const commutator = H.map((row, i) => row.map((v, j) => scaleC(v, int(values[j] - values[i]))));
    const squared = frobenius(commutator), mean = div(sum(state.map((v, i) => mul(norm2([v]), int(values[i])))), N);
    const spread = sub(div(sum(state.map((v, i) => mul(norm2([v]), int(values[i] ** 2)))), N), mul(mean, mean));
    return { conserved: squared.n === BigInt(0), commutatorFrobeniusSquared: q(squared), expectation: q(mean), variance: q(spread), selectedStateSharp: spread.n === BigInt(0) };
  }
  const l = p.leftCreation ? creators[p.leftMode] : operators[p.leftMode], r = p.rightCreation ? creators[p.rightMode] : operators[p.rightMode];
  const nl = p.leftCreation ? naiveCreators[p.leftMode] : naiveOperators[p.leftMode], nr = p.rightCreation ? naiveCreators[p.rightMode] : naiveOperators[p.rightMode];
  const forward = apply(product(l, r), state), reverse = apply(product(r, l), state);
  const scalar = Number(p.leftMode === p.rightMode && p.leftCreation !== p.rightCreation);
  const anti = plus(product(l, r), product(r, l)), naiveAnti = plus(product(nl, nr), product(nr, nl));
  const carError = anti.map((row, i) => row.map((v, j) => v - scalar * Number(i === j)));
  const naiveError = naiveAnti.map((row, i) => row.map((v, j) => v - scalar * Number(i === j)));
  const termValues = terms.map(t => {
    const mean = expectation(t.operator), weighted = cm(t.coefficient, mean);
    if (!t.paired && weighted.im.n !== BigInt(0)) throw new RangeError('Real unpaired energy expected');
    return { type: t.type, modes: t.modes, coefficient: cq(t.coefficient), expectation: cq(mean), includesAdjoint: t.paired, energyContribution: q(t.paired ? mul(two, weighted.re) : weighted.re) };
  });
  const isDiagonal = H.every((row, i) => row.every((v, j) => i === j || v.re.n === BigInt(0) && v.im.n === BigInt(0)));
  let diagonalSpectrum = null;
  if (isDiagonal) {
    const groups = new Map<string, { energy: F; indices: number[]; weight: F }>();
    H.forEach((row, i) => {
      const energy = row[i].re, key = `${energy.n}/${energy.d}`, g = groups.get(key) ?? { energy, indices: [], weight: zero };
      g.indices.push(i); g.weight = add(g.weight, norm2([state[i]])); groups.set(key, g);
    });
    const ordered = [...groups.values()].sort((a, b) => a.energy.n * b.energy.d < b.energy.n * a.energy.d ? -1 : a.energy.n * b.energy.d > b.energy.n * a.energy.d ? 1 : 0);
    const ground = ordered[0], gap = ordered.length > 1 ? sub(ordered[1].energy, ground.energy) : null;
    const groundProbability = div(ground.weight, N), ratio = gap === null ? null : div(sub(m.energy, ground.energy), gap);
    const minimumWithin = (indices: number[]) => {
      const allowed = new Set(indices), levels = ordered.map(g => ({ ...g, indices: g.indices.filter(i => allowed.has(i)) })).filter(g => g.indices.length > 0);
      const first = levels[0], probability = div(sum(first.indices.map(i => norm2([state[i]]))), N);
      return { energy: q(first.energy), multiplicity: first.indices.length, basisIndices: first.indices, probability: q(probability), excitationGap: levels.length > 1 ? q(sub(levels[1].energy, first.energy)) : null, selectedStateGround: eq(probability, one) };
    };
    const parityMinima = [0, 1].map(sector => ({ parity: sector === 0 ? 'even' : 'odd', ...minimumWithin(occupations.flatMap((n, i) => n % 2 === sector ? [i] : [])) }));
    const numberMinima = Array.from({ length: modes + 1 }, (_, number) => ({ number, ...minimumWithin(occupations.flatMap((n, i) => n === number ? [i] : [])) }));
    diagonalSpectrum = {
      basisEnergies: H.map((row, i) => q(row[i].re)),
      levels: ordered.map(g => ({ energy: q(g.energy), basisIndices: g.indices, multiplicity: g.indices.length, probability: q(div(g.weight, N)) })),
      groundEnergy: q(ground.energy), groundMultiplicity: ground.indices.length, groundProbability: q(groundProbability), leakage: q(sub(one, groundProbability)),
      excitationGap: gap === null ? null : q(gap), excessEnergyToGapRatio: ratio === null ? null : q(ratio), certifiedLeakageUpperBound: ratio === null ? null : q(ratio.n > ratio.d ? one : ratio),
      selectedStateGround: eq(groundProbability, one), scope: 'full-Fock-space-not-a-fixed-number-or-parity-sector',
      parityMinima, numberMinima, selectedParityGround: p.sector === 0 ? null : parityMinima[p.sector - 1].selectedStateGround,
    };
  }
  const coherence = frobenius(density.map((row, i) => row.map((v, j) => cs(v, dephased[i][j]))));
  const dephasedEnergy = real(traceProduct(dephased, H));
  return {
    parameters, modes, dimension, basisConvention: 'n0-is-most-significant;ascending-creation-order;rightmost-operator-acts-first',
    basisLabels: Array.from({ length: dimension }, (_, i) => i.toString(2).padStart(modes, '0')), occupationCounts: occupations,
    arithmetic: 'exact-complex-rational', normalizationPrecisionBits: 512,
    hamiltonian: matrixObservation(H), state: state.map(cq), normalizedState: state.map(v => normalized(v, N)), rawNormSquared: q(N),
    hamiltonianAction: m.action.map(cq), energy: q(m.energy), secondMoment: q(m.second), variance: q(m.variance), residual: m.residual.map(cq), normalizedResidual: m.residual.map(v => normalized(v, N)),
    terms: termValues, density: matrixObservation(density), gamma: matrixObservation(gamma), anomalous: matrixObservation(anomalous), oddMeans: oddMeans.map(cq),
    modeOccupations: modeOccupations.map(q), numberDistribution, numberCorrelations, totalNumber: symmetry(occupations), fermionParity: symmetry(parity),
    parityAssumption: {
      declaredInput: p.sector === 0 ? 'formal-either-parity-allowed' : p.sector === 1 ? 'even-pure-sector' : 'odd-pure-sector',
      evenProbability: q(evenProbability), oddProbability: q(oddProbability), coherenceFrobeniusSquared: q(coherence),
      pureStateCompatibleWithParitySuperselection: coherence.n === BigInt(0),
    },
    parityDephased: { density: matrixObservation(dephased), purity: q(frobenius(dephased)), energy: q(dephasedEnergy), oddMeans: operators.map(a => cq(traceProduct(dephased, opAsComplex(a)))) },
    operators: operators.map((a, mode) => ({ mode, annihilationColumns: a[0].map((_, column) => {
      const row = a.findIndex(row => row[column] !== 0); return { row: row < 0 ? null : row, sign: row < 0 ? 0 : a[row][column] };
    }) })), car,
    probe: {
      expectedAnticommutatorScalar: scalar, forward: forward.map(cq), reverse: reverse.map(cq), anticommutatorAction: forward.map((v, i) => cq(ca(v, reverse[i]))),
      forwardOnNormalizedInput: forward.map(v => normalized(v, N)), reverseOnNormalizedInput: reverse.map(v => normalized(v, N)),
      forwardSquaredNormOnNormalizedInput: q(div(norm2(forward), N)), reverseSquaredNormOnNormalizedInput: q(div(norm2(reverse), N)),
      carResidualFrobeniusSquared: q(opSquared(carError)), naiveCarResidualFrobeniusSquared: q(opSquared(naiveError)),
    },
    naiveComparison: { energy: q(naiveMoments.energy), variance: q(naiveMoments.variance), hamiltonianDifferenceFrobeniusSquared: q(frobenius(H.map((row, i) => row.map((v, j) => cs(v, naiveH[i][j]))))) },
    spectralMode: isDiagonal ? 'occupation-basis-enumeration' : 'not-computed-for-nondiagonal-hamiltonian', diagonalSpectrum,
    classification: { exactEigenstate: m.variance.n === BigInt(0), selectedStateGround: diagonalSpectrum?.selectedStateGround ?? null, selectedStateParitySectorGround: diagonalSpectrum?.selectedParityGround ?? null },
    exactResiduals: {
      hermiticityFrobeniusSquared: q(frobenius(H.map((row, i) => row.map((v, j) => cs(v, cc(H[j][i])))))),
      varianceAccount: q(sub(m.variance, sub(m.second, mul(m.energy, m.energy)))),
      residualOrthogonality: cq(dot(state, m.residual)),
      energyAccount: q(sub(m.energy, sum(terms.map(t => { const e = cm(t.coefficient, expectation(t.operator)).re; return t.paired ? mul(two, e) : e; })))),
      gammaHermiticity: q(frobenius(gamma.map((row, i) => row.map((v, j) => cs(v, cc(gamma[j][i])))))),
      anomalousAntisymmetry: q(frobenius(anomalous.map((row, i) => row.map((v, j) => ca(v, anomalous[j][i]))))),
      gammaTraceNumber: q(sub(sum(gamma.map((row, i) => row[i].re)), real(expectation(numberOperators.reduce(plus))))),
      parityTwirlEnergyAccount: q(sub(dephasedEnergy, m.energy)),
    },
    approximationContract: '1..4 ordered fermionic modes; exact occupation-sign creation/annihilation and all CAR; real onsite/density and general complex hopping/pairing with correctly reversed adjoints.74 owned bounded integers,nonzero raw complex state,optional even/odd pure-sector restriction. Formal parity-coherent inputs are identified under the declared parity superselection assumption; parity twirl is a comparison,not a silent correction. Exact moments,one-body/anomalous/number correlations and symmetry commutators; outward512-bit normalized coordinates. Diagonal spectra cover full Fock space,not a selected particle or parity sector. No nondiagonal ground certificate,mode entanglement from qubit partial traces,arbitrary quartic scattering,time evolution or optimization is supplied.',
  };
}
