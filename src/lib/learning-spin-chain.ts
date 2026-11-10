/** General graph-local two-body Pauli Hamiltonians; exact moments, not a
 * noncommuting many-body ground-state solver or a time-evolution method. */
import {
  c, ca, cc, cm, cs, scaleC, abs2, norm2, dot, f, q, cq, normalized,
  zero, one, two, add, sub, mul, div, sum, eq, type C, type F,
} from './learning-quantum-hilbert-core';
import { snapshotSpinChainParameters, spinChainEdges, spinChainAxes, type SpinChainParameters } from './learning-spin-chain-input';
export { isSpinChainParameters, snapshotSpinChainParameters, createSpinChainParameters, spinChainParameterKeys, spinChainEdges, spinChainAxes, type SpinChainParameters } from './learning-spin-chain-input';

type Word = string;
type Matrix = C[][];
const int = (n: number) => f(BigInt(n));
const isZero = (z: C) => z.re.n === BigInt(0) && z.im.n === BigInt(0);
const vectorAdd = (a: C[], b: C[]) => a.map((v, i) => ca(v, b[i]));
const zeroVector = (n: number) => Array.from({ length: n }, () => c(zero));
const matrixObservation = (m: Matrix) => m.map(row => row.map(cq));
const applyMatrix = (m: Matrix, z: C[]) => m.map(row => row.reduce((value, entry, j) => ca(value, cm(entry, z[j])), c(zero)));

/** P|column> has exactly one nonzero row and a phase in {1,i,-1,-i}. */
function wordEntry(word: Word, column: number): { row: number; phase: C } {
  let row = column, exponent = 0;
  for (let site = 0; site < word.length; site++) {
    const mask = 2 ** (word.length - site - 1), bit = (column & mask) !== 0;
    if (word[site] === 'X' || word[site] === 'Y') row ^= mask;
    if (word[site] === 'Y') exponent += bit ? 3 : 1;
    if (word[site] === 'Z' && bit) exponent += 2;
  }
  return { row, phase: [c(one), c(zero, one), c(int(-1)), c(zero, int(-1))][exponent % 4] };
}

function applyWord(word: Word, z: C[]): C[] {
  const result = zeroVector(z.length);
  z.forEach((v, column) => { const { row, phase } = wordEntry(word, column); result[row] = cm(phase, v); });
  return result;
}

function realExpectation(z: C[], action: C[], norm: F): F {
  const value = dot(z, action);
  if (value.im.n !== BigInt(0)) throw new RangeError('A Hermitian expectation must be real');
  return div(value.re, norm);
}

export function spinChainModel(value: SpinChainParameters) {
  const parameters = snapshotSpinChainParameters(value), p = parameters, sites = p.sites, dimension = 2 ** sites;
  const graphEdges = spinChainEdges(sites, p.graph), denominator = int(p.hDenominator);
  const state = Array.from({ length: dimension }, (_, i) => c(int(p[`stateReal${i}`]), int(p[`stateImag${i}`])));
  const N = norm2(state), word = (items: [number, string][]): Word => {
    const letters = Array<string>(sites).fill('I'); items.forEach(([site, axis]) => { letters[site] = axis; }); return letters.join('');
  };
  const expectation = (letters: Word) => realExpectation(state, applyWord(letters, state), N);
  const terms: { sourceKey: string; word: Word; support: number[]; coefficient: F }[] = [];
  const term = (sourceKey: string, items: [number, string][]) => {
    if (p[sourceKey] !== 0) terms.push({ sourceKey, word: word(items), support: items.map(([i]) => i), coefficient: div(int(p[sourceKey]), denominator) });
  };
  term('offset', []);
  for (let i = 0; i < sites; i++) for (const a of spinChainAxes) term(`field${a}${i}`, [[i, a]]);
  for (const [i, j] of graphEdges) for (const a of spinChainAxes) for (const b of spinChainAxes) term(`coupling${a}${b}${i}${j}`, [[i, a], [j, b]]);
  const H: Matrix = Array.from({ length: dimension }, () => zeroVector(dimension));
  let termAction = zeroVector(dimension);
  for (const t of terms) {
    for (let column = 0; column < dimension; column++) {
      const { row, phase } = wordEntry(t.word, column);
      H[row][column] = ca(H[row][column], scaleC(phase, t.coefficient));
    }
    termAction = vectorAdd(termAction, applyWord(t.word, state).map(v => scaleC(v, t.coefficient)));
  }
  const action = applyMatrix(H, state), E = realExpectation(state, action, N), second = div(norm2(action), N);
  const residual = action.map((v, i) => cs(v, scaleC(state[i], E))), variance = div(norm2(residual), N);
  if (variance.n < BigInt(0) || !eq(variance, sub(second, mul(E, E)))) throw new RangeError('Energy moment and residual certificates disagree');
  const termValues = terms.map(t => ({ ...t, expectation: expectation(t.word) }));
  const local = Array.from({ length: sites }, (_, site) => {
    const values = spinChainAxes.map(a => expectation(word([[site, a]])));
    const [x, y, z] = values, p0 = div(add(one, z), two), p1 = div(sub(one, z), two);
    const purity = div(add(one, sum(values.map(v => mul(v, v)))), two);
    const density = [[c(p0), c(div(x, two), div(sub(zero, y), two))], [c(div(x, two), div(y, two)), c(p1)]];
    return { site, values, purity, density };
  });
  const correlations = [];
  for (let i = 0; i < sites; i++) for (let j = i + 1; j < sites; j++) {
    const values = spinChainAxes.map((a, ai) => spinChainAxes.map((b, bi) => {
      const value = expectation(word([[i, a], [j, b]]));
      return { value: q(value), connected: q(sub(value, mul(local[i].values[ai], local[j].values[bi]))) };
    }));
    correlations.push({ sites: [i, j], values });
  }
  const basisLabels = Array.from({ length: dimension }, (_, i) => i.toString(2).padStart(sites, '0'));
  const zValues = basisLabels.map(label => int([...label].reduce((n, bit) => n + (bit === '0' ? 1 : -1), 0)));
  const parityValues = basisLabels.map(label => int([...label].filter(bit => bit === '1').length % 2 ? -1 : 1));
  const symmetry = (values: F[]) => {
    const commutator = H.map((row, i) => row.map((v, j) => scaleC(v, sub(values[j], values[i]))));
    const squaredNorm = norm2(commutator.flat()), mean = sum(state.map((v, i) => mul(abs2(v), values[i])));
    const secondMoment = div(sum(state.map((v, i) => mul(abs2(v), mul(values[i], values[i])))), N);
    const average = div(mean, N), spread = sub(secondMoment, mul(average, average));
    return { conserved: squaredNorm.n === BigInt(0), commutatorFrobeniusSquared: q(squaredNorm), expectation: q(average), variance: q(spread), selectedStateSharp: spread.n === BigInt(0) };
  };
  const isDiagonal = H.every((row, i) => row.every((v, j) => i === j || isZero(v)));
  let diagonalSpectrum = null;
  if (isDiagonal) {
    const groups = new Map<string, { energy: F; indices: number[]; weight: F }>();
    H.forEach((row, i) => {
      const energy = row[i].re, key = `${energy.n}/${energy.d}`, item = groups.get(key) ?? { energy, indices: [], weight: zero };
      item.indices.push(i); item.weight = add(item.weight, abs2(state[i])); groups.set(key, item);
    });
    const ordered = [...groups.values()].sort((a, b) => a.energy.n * b.energy.d < b.energy.n * a.energy.d ? -1 : a.energy.n * b.energy.d > b.energy.n * a.energy.d ? 1 : 0);
    const ground = ordered[0], gap = ordered.length > 1 ? sub(ordered[1].energy, ground.energy) : null;
    const groundProbability = div(ground.weight, N), leakage = sub(one, groundProbability);
    const ratio = gap === null ? null : div(sub(E, ground.energy), gap);
    diagonalSpectrum = {
      basisEnergies: H.map((row, i) => q(row[i].re)),
      levels: ordered.map(g => ({ energy: q(g.energy), basisIndices: g.indices, multiplicity: g.indices.length, probability: q(div(g.weight, N)) })),
      groundEnergy: q(ground.energy), groundMultiplicity: ground.indices.length, groundProbability: q(groundProbability), leakage: q(leakage),
      excitationGap: gap === null ? null : q(gap), excessEnergy: q(sub(E, ground.energy)),
      excessEnergyToGapRatio: ratio === null ? null : q(ratio),
      certifiedLeakageUpperBound: ratio === null ? null : q(ratio.n > ratio.d ? one : ratio),
      selectedStateGround: eq(groundProbability, one),
    };
  }
  const frobeniusSquared = norm2(H.flat());
  return {
    parameters, sites, dimension, graphEdges, activeEdges: graphEdges.filter(([i, j]) => spinChainAxes.some(a => spinChainAxes.some(b => p[`coupling${a}${b}${i}${j}`] !== 0))),
    basisConvention: 'n0-is-most-significant;rightmost-operator-acts-first;Pauli-not-hbar-spin', basisLabels,
    arithmetic: 'exact-complex-rational', normalizationPrecisionBits: 512,
    hamiltonian: matrixObservation(H), state: state.map(cq), normalizedState: state.map(v => normalized(v, N)), hamiltonianAction: action.map(cq),
    rawNormSquared: q(N), energy: q(E), secondMoment: q(second), variance: q(variance), residual: residual.map(cq), normalizedResidual: residual.map(v => normalized(v, N)), hamiltonianFrobeniusSquared: q(frobeniusSquared),
    terms: termValues.map(t => ({ sourceKey: t.sourceKey, word: t.word, support: t.support, coefficient: q(t.coefficient), expectation: q(t.expectation), energyContribution: q(mul(t.coefficient, t.expectation)) })),
    localStates: local.map(v => ({ site: v.site, bloch: v.values.map(q), purity: q(v.purity), pure: eq(v.purity, one), density: matrixObservation(v.density) })),
    correlations, totalZ: symmetry(zValues), globalZParity: symmetry(parityValues),
    spectralMode: isDiagonal ? 'basis-enumeration' : 'not-computed-for-nondiagonal-hamiltonian', diagonalSpectrum,
    classification: { exactEigenstate: variance.n === BigInt(0), selectedStateGround: diagonalSpectrum?.selectedStateGround ?? null },
    cost: { complexStateAmplitudes: dimension, denseComplexMatrixEntries: dimension * dimension, nonzeroPauliTerms: terms.length, nonzeroMatrixEntries: H.flat().filter(v => !isZero(v)).length },
    exactResiduals: {
      hermiticityFrobeniusSquared: q(norm2(H.flatMap((row, i) => row.map((v, j) => cs(v, cc(H[j][i])))))),
      pauliFrobeniusAccount: q(sub(frobeniusSquared, mul(int(dimension), sum(terms.map(t => mul(t.coefficient, t.coefficient)))))),
      termActionAccount: q(norm2(action.map((v, i) => cs(v, termAction[i])))),
      energyAccount: q(sub(E, sum(termValues.map(t => mul(t.coefficient, t.expectation))))),
      varianceAccount: q(sub(variance, sub(second, mul(E, E)))),
      residualOrthogonality: cq(dot(state, residual)),
    },
    approximationContract: '1..4 spin-1/2 sites; all real local XYZ and nine ordered-axis two-site Pauli coefficients on unique open,ring or custom pairs; integer raw complex amplitudes and a positive common energy denominator. Exact rational Hamiltonian,moments,residuals,local density matrices,all-pair connected correlations and symmetry commutators; outward512-bit normalized coordinates. Diagonal Hamiltonians enumerate all product-basis energies and degeneracies. No spectrum,ground/gap certificate or eigenvector is supplied for a nondiagonal Hamiltonian; zero variance alone is not ground-state evidence. No optimization,time evolution,thermodynamic limit or fermionic statistics is inferred.',
  };
}
