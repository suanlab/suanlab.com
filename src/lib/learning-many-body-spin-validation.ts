/** Spin response attribution by Kronecker matrices and direct partial traces.
 * No calculator, spectral engine or supplied residual is trusted as an oracle.
 */
import { readManyBodyInput, manyBodyInputKey, type ManyBodyInput } from './learning-many-body-input';
import type { ManyBodySuccess } from './learning-many-body-protocol';
import { snapshotManyBodyWire } from './learning-many-body-owned-wire';
import { spinChainResultContract } from './learning-many-body-contracts';
import { expectedRational as q, expectedComplex as cq, expectedNormalized as normalized, matchManyBodyExpected, type ManyBodyExpected as Expected } from './learning-many-body-wire-match';
import { int, zero, one, add, sub, mul, div, sum, eq, le, c, ca, cc, cm, cs, scale, abs2, norm2, matrixScale, matrixAdd, matrixNorm2, tensor, type C, type F, type Matrix } from './learning-tensor-wire-math';

const axes = ['X', 'Y', 'Z'] as const;
const pauli: Record<string, Matrix> = {
  I: [[c(one), c(zero)], [c(zero), c(one)]],
  X: [[c(zero), c(one)], [c(one), c(zero)]],
  Y: [[c(zero), c(zero, int(-1))], [c(zero, one), c(zero)]],
  Z: [[c(one), c(zero)], [c(zero), c(int(-1))]],
};
const observedMatrix = (a: Matrix): Expected => a.map(row => row.map(cq));
const apply = (a: Matrix, state: C[]) => a.map(row => row.reduce((s, v, j) => ca(s, cm(v, state[j])), c(zero)));
const dot = (a: C[], b: C[]) => a.reduce((s, v, i) => ca(s, cm(cc(v), b[i])), c(zero));

function spinExpected(p: Readonly<Record<string, number>>): Expected {
  const sites = p.sites, dimension = 2 ** sites, denominator = int(p.hDenominator);
  const state = Array.from({ length: dimension }, (_, i) => c(int(p[`stateReal${i}`]), int(p[`stateImag${i}`])));
  const N = norm2(state), matrices = new Map<string, Matrix>();
  const word = (items: [number, string][]) => {
    const letters = Array<string>(sites).fill('I'); for (const [site, axis] of items) letters[site] = axis; return letters.join('');
  };
  function matrix(word: string): Matrix {
    if (!matrices.has(word)) matrices.set(word, [...word].reduce((m, a) => tensor(m, pauli[a]), [[c(one)]]));
    return matrices.get(word)!;
  }
  const expectation = (word: string) => div(dot(state, apply(matrix(word), state)).re, N);
  const edges: [number, number][] = [];
  for (let i = 0; i < sites; i++) for (let j = i + 1; j < sites; j++) {
    if (p.graph === 2 || j === i + 1 || p.graph === 1 && sites >= 3 && i === 0 && j === sites - 1) edges.push([i, j]);
  }
  const terms: { sourceKey: string; word: string; support: number[]; coefficient: F }[] = [];
  const term = (sourceKey: string, items: [number, string][]) => {
    if (p[sourceKey]) terms.push({ sourceKey, word: word(items), support: items.map(([i]) => i), coefficient: div(int(p[sourceKey]), denominator) });
  };
  term('offset', []);
  for (let i = 0; i < sites; i++) for (const a of axes) term(`field${a}${i}`, [[i, a]]);
  for (const [i, j] of edges) for (const a of axes) for (const b of axes) term(`coupling${a}${b}${i}${j}`, [[i, a], [j, b]]);
  const empty = Array.from({ length: dimension }, () => Array.from({ length: dimension }, () => c(zero)));
  const H = terms.reduce((a, t) => matrixAdd(a, matrixScale(matrix(t.word), t.coefficient)), empty);
  const action = apply(H, state), E = div(dot(state, action).re, N), second = div(norm2(action), N);
  const residual = action.map((v, i) => cs(v, scale(state[i], E))), variance = sub(second, mul(E, E));
  const bloch = Array.from({ length: sites }, (_, i) => axes.map(a => expectation(word([[i, a]]))));
  const local = Array.from({ length: sites }, (_, site) => {
    const rho: Matrix = [[c(zero), c(zero)], [c(zero), c(zero)]], mask = 2 ** (sites - site - 1);
    for (let a = 0; a < dimension; a++) for (let b = 0; b < dimension; b++) if ((a & ~mask) === (b & ~mask)) {
      const i = Number((a & mask) !== 0), j = Number((b & mask) !== 0);
      rho[i][j] = ca(rho[i][j], scale(cm(state[a], cc(state[b])), div(one, N)));
    }
    const purity = matrixNorm2(rho);
    return { site, bloch: bloch[site].map(q), purity: q(purity), pure: eq(purity, one), density: observedMatrix(rho) };
  });
  const correlations: Expected[] = [];
  for (let i = 0; i < sites; i++) for (let j = i + 1; j < sites; j++) correlations.push({ sites: [i, j], values: axes.map((a, ai) => axes.map((b, bi) => {
    const value = expectation(word([[i, a], [j, b]]));
    return { value: q(value), connected: q(sub(value, mul(bloch[i][ai], bloch[j][bi]))) };
  })) });
  const labels = Array.from({ length: dimension }, (_, i) => i.toString(2).padStart(sites, '0'));
  const symmetry = (values: F[]) => {
    const spread = sum(H.flatMap((row, i) => row.map((h, j) => mul(abs2(h), mul(sub(values[j], values[i]), sub(values[j], values[i]))))));
    const mean = div(sum(state.map((z, i) => mul(abs2(z), values[i]))), N);
    const second = div(sum(state.map((z, i) => mul(abs2(z), mul(values[i], values[i])))), N), variance = sub(second, mul(mean, mean));
    return { conserved: spread.n === BigInt(0), commutatorFrobeniusSquared: q(spread), expectation: q(mean), variance: q(variance), selectedStateSharp: variance.n === BigInt(0) };
  };
  const diagonal = H.every((row, i) => row.every((v, j) => i === j || abs2(v).n === BigInt(0)));
  let spectrum: Expected = null, selectedGround: boolean | null = null;
  if (diagonal) {
    const groups: { energy: F; indices: number[]; weight: F }[] = [];
    for (let i = 0; i < dimension; i++) {
      const energy = H[i][i].re; let group = groups.find(g => eq(g.energy, energy));
      if (!group) { group = { energy, indices: [], weight: zero }; groups.push(group); }
      group.indices.push(i); group.weight = add(group.weight, abs2(state[i]));
    }
    groups.sort((a, b) => eq(a.energy, b.energy) ? 0 : le(a.energy, b.energy) ? -1 : 1);
    const ground = groups[0], probability = div(ground.weight, N), excess = sub(E, ground.energy);
    const gap = groups.length > 1 ? sub(groups[1].energy, ground.energy) : null;
    const ratio = gap === null ? null : div(excess, gap); selectedGround = eq(probability, one);
    spectrum = {
      basisEnergies: H.map((row, i) => q(row[i].re)),
      levels: groups.map(g => ({ energy: q(g.energy), basisIndices: g.indices, multiplicity: g.indices.length, probability: q(div(g.weight, N)) })),
      groundEnergy: q(ground.energy), groundMultiplicity: ground.indices.length, groundProbability: q(probability), leakage: q(sub(one, probability)),
      excitationGap: gap === null ? null : q(gap), excessEnergy: q(excess), excessEnergyToGapRatio: ratio === null ? null : q(ratio),
      certifiedLeakageUpperBound: ratio === null ? null : q(le(ratio, one) ? ratio : one), selectedStateGround: selectedGround,
    };
  }
  return {
    parameters: { ...p }, sites, dimension, graphEdges: edges,
    activeEdges: edges.filter(([i, j]) => axes.some(a => axes.some(b => p[`coupling${a}${b}${i}${j}`] !== 0))),
    basisConvention: 'n0-is-most-significant;rightmost-operator-acts-first;Pauli-not-hbar-spin', basisLabels: labels,
    arithmetic: 'exact-complex-rational', normalizationPrecisionBits: 512,
    hamiltonian: observedMatrix(H), state: state.map(cq), normalizedState: state.map(z => normalized(z, N)), hamiltonianAction: action.map(cq),
    rawNormSquared: q(N), energy: q(E), secondMoment: q(second), variance: q(variance), residual: residual.map(cq), normalizedResidual: residual.map(z => normalized(z, N)), hamiltonianFrobeniusSquared: q(matrixNorm2(H)),
    terms: terms.map(t => { const mean = expectation(t.word); return { sourceKey: t.sourceKey, word: t.word, support: t.support, coefficient: q(t.coefficient), expectation: q(mean), energyContribution: q(mul(t.coefficient, mean)) }; }),
    localStates: local, correlations,
    totalZ: symmetry(labels.map(label => int([...label].reduce((n, bit) => n + (bit === '0' ? 1 : -1), 0)))),
    globalZParity: symmetry(labels.map(label => int([...label].filter(bit => bit === '1').length % 2 ? -1 : 1))),
    spectralMode: diagonal ? 'basis-enumeration' : 'not-computed-for-nondiagonal-hamiltonian', diagonalSpectrum: spectrum,
    classification: { exactEigenstate: variance.n === BigInt(0), selectedStateGround: selectedGround },
    cost: { complexStateAmplitudes: dimension, denseComplexMatrixEntries: dimension * dimension, nonzeroPauliTerms: terms.length, nonzeroMatrixEntries: H.flat().filter(z => abs2(z).n !== BigInt(0)).length },
    exactResiduals: { hermiticityFrobeniusSquared: q(zero), pauliFrobeniusAccount: q(zero), termActionAccount: q(zero), energyAccount: q(zero), varianceAccount: q(zero), residualOrthogonality: cq(c(zero)) },
    approximationContract: spinChainResultContract,
  };
}

export type ManyBodySpinSuccess = Extract<ManyBodySuccess, { kind: 'spin-chain' }>;
/** Returns an owned frozen copy, not the untrusted packet that was inspected. */
export function readSpinChainSuccessForInput(value: unknown, original: ManyBodyInput, requestId: number): ManyBodySpinSuccess | null {
  try {
    if (!Number.isSafeInteger(requestId) || requestId <= 0) return null;
    const input = readManyBodyInput(original);
    if (!input || input.kind !== 'spin-chain') return null;
    const packet = snapshotManyBodyWire(value);
    const expected: Expected = { version: 1, requestId, ok: true, kind: 'spin-chain', input: { kind: input.kind, parameters: { ...input.parameters } }, inputKey: manyBodyInputKey(input), result: spinExpected(input.parameters) };
    return matchManyBodyExpected(packet, expected) ? packet as ManyBodySpinSuccess : null;
  } catch { return null; }
}
