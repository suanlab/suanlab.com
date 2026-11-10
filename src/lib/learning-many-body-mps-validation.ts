/** MPS attribution by explicit virtual paths, direct Grams and dense traces.
 * Product transfer recurrences, gauge residuals and supplied ranks are not an
 * oracle; rank comes from exact principal minors of the smaller Gram matrix.
 */
import { readManyBodyInput, manyBodyInputKey, type ManyBodyInput } from './learning-many-body-input';
import type { ManyBodySuccess } from './learning-many-body-protocol';
import { snapshotManyBodyWire } from './learning-many-body-owned-wire';
import { mpsResultContract } from './learning-many-body-contracts';
import { expectedRational as q, expectedComplex as cq, expectedNormalized as normalized, matchManyBodyExpected, type ManyBodyExpected as Expected } from './learning-many-body-wire-match';
import { f, int, zero, one, two, sub, mul, div, sum, c, ca, cc, cm, cs, scale, norm2, identity, transpose, adjoint, product, matrixAdd, matrixSub, matrixNorm2, tensor, invariants, type C, type Matrix } from './learning-tensor-wire-math';
type Tensor = [Matrix, Matrix];
const empty = (n: number, m = n): Matrix => Array.from({ length: n }, () => Array.from({ length: m }, () => c(zero)));
const nonzero = (v: C) => v.re.n !== BigInt(0) || v.im.n !== BigInt(0);
const observedMatrix = (a: Matrix): Expected => a.map(row => row.map(cq));
const conjugate = (a: Matrix) => a.map(row => row.map(cc));
const apply = (a: Matrix, z: C[]) => a.map(row => row.reduce((s, v, j) => nonzero(v) ? ca(s, cm(v, z[j])) : s, c(zero)));
const dot = (a: C[], b: C[]) => a.reduce((s, v, i) => ca(s, cm(cc(v), b[i])), c(zero));
const pauli: Record<string, Matrix> = { I: identity(2), X: [[c(zero), c(one)], [c(one), c(zero)]], Y: [[c(zero), c(zero, int(-1))], [c(zero, one), c(zero)]], Z: [[c(one), c(zero)], [c(zero), c(int(-1))]] };

function evaluate(tensors: Tensor[], dimensions: number[]) {
  const sites = tensors.length, size = 2 ** sites;
  function paths(start: number, end: number, bits: number, left: number, right: number): C {
    let total = c(zero);
    function visit(site: number, virtual: number, value: C) {
      if (site === end) { if (virtual === right) total = ca(total, value); return; }
      const physical = (bits >> (end - site - 1)) & 1;
      for (let next = 0; next < dimensions[site + 1]; next++) {
        const entry = tensors[site][physical][virtual][next];
        if (nonzero(entry)) visit(site + 1, next, cm(value, entry));
      }
    }
    visit(start, left, c(one)); return total;
  }
  const prefixes = Array.from({ length: sites + 1 }, (_, k) => Array.from({ length: 2 ** k }, (_, x) => Array.from({ length: dimensions[k] }, (_, a) => paths(0, k, x, 0, a))));
  const suffixes = Array.from({ length: sites + 1 }, (_, k) => Array.from({ length: dimensions[k] }, (_, a) => Array.from({ length: 2 ** (sites - k) }, (_, y) => paths(k, sites, y, a, 0))));
  const state = prefixes[sites].map(row => row[0]), N = norm2(state), defined = N.n !== BigInt(0);
  const leftEnvironments = prefixes.map(L => product(adjoint(L), L));
  const rightEnvironments = suffixes.map(B => product(conjugate(B), transpose(B)));
  const closures = leftEnvironments.map((E, k) => E.reduce((s, row, a) => row.reduce((s, v, b) => ca(s, cm(v, rightEnvironments[k][a][b])), s), c(zero)).re);
  const cuts = dimensions.slice(1, -1).map((bondDimension, index) => {
    const cut = index + 1, left = 2 ** cut, right = size / left;
    const C = Array.from({ length: left }, (_, i) => state.slice(i * right, (i + 1) * right));
    const gram = left <= right ? product(C, adjoint(C)) : product(adjoint(C), C), rank = invariants(gram).positivePrincipalRank;
    return { cut, bondDimension, leftDimension: left, rightDimension: right, coefficientRank: rank, schmidtRank: defined ? rank : null, coefficientMatrix: observedMatrix(C) };
  });
  const canonicality = tensors.map((t, site) => {
    const left = matrixAdd(product(adjoint(t[0]), t[0]), product(adjoint(t[1]), t[1]));
    const right = matrixAdd(product(t[0], adjoint(t[0])), product(t[1], adjoint(t[1])));
    const l = matrixNorm2(matrixSub(left, identity(left.length))), r = matrixNorm2(matrixSub(right, identity(right.length)));
    return { site, leftGram: observedMatrix(left), rightGram: observedMatrix(right), leftIsometry: l.n === BigInt(0), rightIsometry: r.n === BigInt(0), leftIdentityResidualSquared: q(l), rightIdentityResidualSquared: q(r) };
  });
  const densities = defined ? tensors.map((_, site) => {
    const mask = 2 ** (sites - site - 1), rho = empty(2);
    for (let a = 0; a < size; a++) for (let b = 0; b < size; b++) if ((a & ~mask) === (b & ~mask)) {
      const i = Number((a & mask) !== 0), j = Number((b & mask) !== 0);
      rho[i][j] = ca(rho[i][j], scale(cm(state[a], cc(state[b])), div(one, N)));
    }
    return rho;
  }) : null;
  const bloch = densities?.map(rho => [mul(two, rho[0][1].re), mul(int(-2), rho[0][1].im), sub(rho[0][0].re, rho[1][1].re)]);
  let physical: Expected = null;
  if (defined) {
    const correlations: Expected[] = [];
    for (let i = 0; i < sites; i++) for (let j = i + 1; j < sites; j++) correlations.push({ sites: [i, j], values: ['X', 'Y', 'Z'].map((a, ai) => ['X', 'Y', 'Z'].map((b, bi) => {
      const ops = Array.from({ length: sites }, (_, k) => pauli[k === i ? a : k === j ? b : 'I']);
      const O = ops.reduce((m, a) => tensor(m, a), [[c(one)]]), value = div(dot(state, apply(O, state)).re, N);
      return { value: q(value), connected: q(sub(value, mul(bloch![i][ai], bloch![j][bi]))) };
    })) });
    physical = { normalizedState: state.map(v => normalized(v, N)), localStates: densities!.map((rho, site) => ({ site, density: observedMatrix(rho), bloch: bloch![site].map(q), purity: q(matrixNorm2(rho)) })), correlations };
  }
  const raw: Record<string, Expected> = {
    tensors: tensors.map(t => t.map(observedMatrix)), state: state.map(cq), normSquared: q(N), tensorParameterSquaredNorm: q(sum(tensors.flatMap(t => t.map(matrixNorm2)))),
    prefixes: prefixes.map(observedMatrix), leftEnvironments: leftEnvironments.map(observedMatrix), rightEnvironments: rightEnvironments.map(observedMatrix), cutClosureNorms: closures.map(q), cuts, canonicality,
    physical, stateDefined: defined,
    exactResiduals: { environmentNormAccount: closures.map(v => q(sub(v, N))), localDensityAccount: defined ? tensors.map(() => q(zero)) : null },
  };
  return { raw, state, N, defined, leftEnvironments, rightEnvironments };
}

function mpsExpected(p: Readonly<Record<string, number>>): Expected {
  const sites = p.sites, bonds = Array.from({ length: sites - 1 }, (_, i) => p[`bond${i}`]), dimensions = [1, ...bonds, 1];
  const tensors: Tensor[] = Array.from({ length: sites }, (_, site) => [0, 1].map(s => Array.from({ length: dimensions[site] }, (_, l) => Array.from({ length: dimensions[site + 1] }, (_, r) => c(f(BigInt(p[`t${site}Real${l}${s}${r}`]), BigInt(p.tensorDenominator)), f(BigInt(p[`t${site}Imag${l}${s}${r}`]), BigInt(p.tensorDenominator)))))) as Tensor);
  const original = evaluate(tensors, dimensions), changed = tensors.map(t => t.map(a => a.map(row => [...row]))) as Tensor[];
  let gauge: Expected = null, G: Matrix | null = null, inverse: Matrix | null = null;
  if (p.gaugeBond >= 0) {
    const bond = p.gaugeBond, size = bonds[bond], row = p.gaugeRow, col = p.gaugeColumn;
    const lambda = f(BigInt(p.gaugeScaleNumerator), BigInt(p.gaugeScaleDenominator)), eta = c(f(BigInt(p.gaugeShearReal), BigInt(p.gaugeShearDenominator)), f(BigInt(p.gaugeShearImag), BigInt(p.gaugeShearDenominator)));
    const D = identity(size), Di = identity(size), S = identity(size), Si = identity(size);
    D[row][row] = c(lambda); Di[row][row] = c(div(one, lambda));
    if (row !== col) { S[row][col] = eta; Si[row][col] = scale(eta, int(-1)); }
    G = product(D, S); inverse = product(Si, Di);
    changed[bond] = tensors[bond].map(a => product(a, G!)) as Tensor;
    changed[bond + 1] = tensors[bond + 1].map(a => product(inverse!, a)) as Tensor;
    gauge = { bond, order: 'G=diagonal-scale-times-shear', matrix: observedMatrix(G), inverse: observedMatrix(inverse), inverseResidualSquared: q(matrixNorm2(matrixSub(product(G, inverse), identity(size)))) };
  }
  const gauged = evaluate(changed, dimensions), cut = p.gaugeBond + 1;
  return {
    parameters: { ...p }, sites, physicalDimensionPerSite: 2, dimension: 2 ** sites, bondDimensions: bonds, virtualDimensionsIncludingBoundaries: dimensions,
    arithmetic: 'exact-complex-rational', normalizationPrecisionBits: 512, basisConvention: 'n0-most-significant;A[site][physical][left][right];open-boundaries-one',
    original: original.raw, gauged: gauged.raw, gauge,
    validity: { structurallyValid: true, physicalStateDefined: original.defined, reason: original.defined ? 'nonzero-contracted-vector' : 'zero-contracted-vector-cannot-normalize' },
    cost: { activeComplexTensorEntries: dimensions.slice(0, -1).reduce((s, d, i) => s + 2 * d * dimensions[i + 1], 0), paddedComplexTensorEntries: 128, denseComplexStateAmplitudes: 2 ** sites,
      prefixDenseMultiplyAccumulateSlots: dimensions.slice(0, -1).reduce((s, d, i) => s + 2 ** (i + 1) * d * dimensions[i + 1], 0),
      normTransferDenseSummandSlots: dimensions.slice(0, -1).reduce((s, d, i) => s + 2 * d * d * dimensions[i + 1] ** 2, 0), countsAreAlgorithmicSlotsNotRuntimeOrCompressionGuarantees: true },
    exactResiduals: {
      stateDifferenceSquared: q(norm2(original.state.map((v, i) => cs(v, gauged.state[i])))), normAccount: q(sub(gauged.N, original.N)),
      leftEnvironmentCovariance: G ? q(matrixNorm2(matrixSub(gauged.leftEnvironments[cut], product(product(adjoint(G), original.leftEnvironments[cut]), G)))) : null,
      rightEnvironmentCovariance: inverse ? q(matrixNorm2(matrixSub(gauged.rightEnvironments[cut], product(product(conjugate(inverse), original.rightEnvironments[cut]), transpose(inverse))))) : null,
    },
    approximationContract: mpsResultContract,
  };
}
export type ManyBodyMpsSuccess = Extract<ManyBodySuccess, { kind: 'mps' }>;
export function readMpsSuccessForInput(value: unknown, original: ManyBodyInput, requestId: number): ManyBodyMpsSuccess | null {
  try {
    if (!Number.isSafeInteger(requestId) || requestId <= 0) return null;
    const input = readManyBodyInput(original); if (!input || input.kind !== 'mps') return null;
    const packet = snapshotManyBodyWire(value);
    const expected: Expected = { version: 1, requestId, ok: true, kind: 'mps', input: { kind: input.kind, parameters: { ...input.parameters } }, inputKey: manyBodyInputKey(input), result: mpsExpected(input.parameters) };
    return matchManyBodyExpected(packet, expected) ? packet as ManyBodyMpsSuccess : null;
  } catch { return null; }
}
