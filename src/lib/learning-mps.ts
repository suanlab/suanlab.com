/** Exact open MPS contraction; virtual gauges are representation changes. */
import { c, ca, cs, cc, cm, scaleC, norm2, f, q, cq, normalized, zero, one, two, sub, mul, div, sum, type C, type F } from './learning-quantum-hilbert-core';
import { identity, product, adjoint, matrixSub, matrixSquaredNorm, matrixObservation as mq, type Matrix as M } from './learning-observable-matrix';
import { snapshotMpsParameters, type MpsParameters } from './learning-mps-input';
export { createMpsParameters, snapshotMpsParameters, isMpsParameters, mpsParameterKeys, type MpsParameters } from './learning-mps-input';
type Tensor = [M, M];
const int = (n: number) => f(BigInt(n));
const empty = (n: number, m = n): M => Array.from({ length: n }, () => Array.from({ length: m }, () => c(zero)));
const isZero = (z: C) => z.re.n === BigInt(0) && z.im.n === BigInt(0);
const real = (z: C): F => { if (z.im.n !== BigInt(0)) throw new RangeError('Real MPS observable expected'); return z.re; };
const transpose = (a: M): M => a[0].map((_, j) => a.map(row => row[j]));
const conjugate = (a: M): M => a.map(row => row.map(cc));
const divideC = (a: C, b: C) => scaleC(cm(a, cc(b)), div(one, norm2([b])));
const tensorObservation = (a: Tensor[]) => a.map(site => site.map(mq));
const pauli: Record<string, M> = { I: identity(2), X: [[c(zero), c(one)], [c(one), c(zero)]], Y: [[c(zero), c(zero, sub(zero, one))], [c(zero, one), c(zero)]], Z: [[c(one), c(zero)], [c(zero), c(sub(zero, one))]] };
function rank(value: M): number {
  const a = value.map(row => [...row]); let row = 0;
  for (let col = 0; col < a[0].length && row < a.length; col++) {
    const pivot = a.findIndex((v, i) => i >= row && !isZero(v[col])); if (pivot < 0) continue;
    [a[row], a[pivot]] = [a[pivot], a[row]];
    for (let i = row + 1; i < a.length; i++) if (!isZero(a[i][col])) {
      const ratio = divideC(a[i][col], a[row][col]);
      for (let j = col; j < a[0].length; j++) a[i][j] = cs(a[i][j], cm(ratio, a[row][j]));
    }
    row++;
  }
  return row;
}
function transferLeft(E: M, tensor: Tensor, operator = pauli.I): M {
  const right = tensor[0][0].length, result = empty(right), left = tensor[0].length;
  for (let a = 0; a < left; a++) for (let ap = 0; ap < left; ap++) if (!isZero(E[a][ap])) {
    for (let s = 0; s < 2; s++) for (let t = 0; t < 2; t++) if (!isZero(operator[s][t])) for (let b = 0; b < right; b++) for (let bp = 0; bp < right; bp++) {
      if (!isZero(tensor[s][a][b]) && !isZero(tensor[t][ap][bp])) result[b][bp] = ca(result[b][bp], cm(cm(cm(cc(tensor[s][a][b]), E[a][ap]), operator[s][t]), tensor[t][ap][bp]));
    }
  }
  return result;
}
function transferRight(R: M, tensor: Tensor): M {
  const left = tensor[0].length, right = tensor[0][0].length, result = empty(left);
  for (let a = 0; a < left; a++) for (let ap = 0; ap < left; ap++) for (let s = 0; s < 2; s++) for (let b = 0; b < right; b++) for (let bp = 0; bp < right; bp++) {
    if (!isZero(R[b][bp]) && !isZero(tensor[s][a][b]) && !isZero(tensor[s][ap][bp])) result[a][ap] = ca(result[a][ap], cm(cm(cc(tensor[s][a][b]), tensor[s][ap][bp]), R[b][bp]));
  }
  return result;
}
function evaluate(tensors: Tensor[], bonds: number[]) {
  const sites = tensors.length, dimension = 2 ** sites;
  const prefixes: M[] = [[[c(one)]]];
  tensors.forEach((tensor, site) => { prefixes.push(prefixes[site].flatMap(row => tensor.map(a => product([row], a)[0]))); });
  const state = prefixes[sites].map(row => row[0]), N = norm2(state), stateDefined = N.n !== BigInt(0);
  const leftEnvironments: M[] = [[[c(one)]]], rightEnvironments: M[] = Array.from({ length: sites + 1 }, () => []);
  tensors.forEach((tensor, i) => { leftEnvironments.push(transferLeft(leftEnvironments[i], tensor)); });
  rightEnvironments[sites] = [[c(one)]];
  for (let i = sites - 1; i >= 0; i--) rightEnvironments[i] = transferRight(rightEnvironments[i + 1], tensors[i]);
  const closureNorms = leftEnvironments.map((E, cut) => real(E.reduce((s, row, a) => row.reduce((v, entry, b) => ca(v, cm(entry, rightEnvironments[cut][a][b])), s), c(zero))));
  const wordExpectation = (letters: string) => real(scaleC(tensors.reduce((E, tensor, i) => transferLeft(E, tensor, pauli[letters[i]]), [[c(one)]])[0][0], div(one, N)));
  const localDensities = stateDefined ? tensors.map((tensor, i) => Array.from({ length: 2 }, (_, ket) => Array.from({ length: 2 }, (_, bra) => {
    let value = c(zero); const E = leftEnvironments[i], R = rightEnvironments[i + 1], left = tensor[0].length, right = tensor[0][0].length;
    for (let a = 0; a < left; a++) for (let ap = 0; ap < left; ap++) for (let b = 0; b < right; b++) for (let bp = 0; bp < right; bp++) {
      value = ca(value, cm(cm(cm(cc(tensor[bra][a][b]), E[a][ap]), tensor[ket][ap][bp]), R[b][bp]));
    }
    return scaleC(value, div(one, N));
  }))) : null;
  const denseLocal = stateDefined ? tensors.map((_, site) => {
    const mask = 2 ** (sites - site - 1), result = empty(2);
    for (let a = 0; a < dimension; a++) for (let b = 0; b < dimension; b++) if ((a & ~mask) === (b & ~mask)) {
      const i = Number((a & mask) !== 0), j = Number((b & mask) !== 0); result[i][j] = ca(result[i][j], scaleC(cm(state[a], cc(state[b])), div(one, N)));
    }
    return result;
  }) : null;
  const localStates = localDensities?.map((rho, site) => ({ site, density: mq(rho), bloch: [mul(two, rho[0][1].re), mul(int(-2), rho[0][1].im), sub(rho[0][0].re, rho[1][1].re)].map(q), purity: q(matrixSquaredNorm(rho)) })) ?? null;
  const correlations = stateDefined ? tensors.flatMap((_, i) => tensors.flatMap((__, j) => {
    if (j <= i) return [];
    return [{ sites: [i, j], values: ['X', 'Y', 'Z'].map((a, ai) => ['X', 'Y', 'Z'].map((b, bi) => {
      const letters = Array<string>(sites).fill('I'); letters[i] = a; letters[j] = b;
      const value = wordExpectation(letters.join(''));
      const localI = [mul(two, localDensities![i][0][1].re), mul(int(-2), localDensities![i][0][1].im), sub(localDensities![i][0][0].re, localDensities![i][1][1].re)][ai];
      const localJ = [mul(two, localDensities![j][0][1].re), mul(int(-2), localDensities![j][0][1].im), sub(localDensities![j][0][0].re, localDensities![j][1][1].re)][bi];
      return { value: q(value), connected: q(sub(value, mul(localI, localJ))) };
    })) }];
  })) : null;
  const cuts = bonds.map((bondDimension, cut) => {
    const leftDimension = 2 ** (cut + 1), rightDimension = dimension / leftDimension;
    const coefficientMatrix = Array.from({ length: leftDimension }, (_, i) => state.slice(i * rightDimension, (i + 1) * rightDimension));
    const coefficientRank = rank(coefficientMatrix);
    if (coefficientRank > bondDimension) throw new RangeError('MPS coefficient rank exceeds its virtual bond');
    return { cut: cut + 1, bondDimension, leftDimension, rightDimension, coefficientRank, schmidtRank: stateDefined ? coefficientRank : null, coefficientMatrix: mq(coefficientMatrix) };
  });
  const canonicality = tensors.map((tensor, site) => {
    const leftGram = tensor.map(a => product(adjoint(a), a)).reduce((a, b) => a.map((row, i) => row.map((v, j) => ca(v, b[i][j]))));
    const rightGram = tensor.map(a => product(a, adjoint(a))).reduce((a, b) => a.map((row, i) => row.map((v, j) => ca(v, b[i][j]))));
    const l = matrixSquaredNorm(matrixSub(leftGram, identity(leftGram.length))), r = matrixSquaredNorm(matrixSub(rightGram, identity(rightGram.length)));
    return { site, leftGram: mq(leftGram), rightGram: mq(rightGram), leftIsometry: l.n === BigInt(0), rightIsometry: r.n === BigInt(0), leftIdentityResidualSquared: q(l), rightIdentityResidualSquared: q(r) };
  });
  const parameterNorm = sum(tensors.map(tensor => sum(tensor.map(a => matrixSquaredNorm(a)))));
  return {
    raw: { tensors: tensorObservation(tensors), state: state.map(cq), normSquared: q(N), tensorParameterSquaredNorm: q(parameterNorm), prefixes: prefixes.map(mq), leftEnvironments: leftEnvironments.map(mq), rightEnvironments: rightEnvironments.map(mq), cutClosureNorms: closureNorms.map(q), cuts, canonicality },
    physical: stateDefined ? { normalizedState: state.map(v => normalized(v, N)), localStates: localStates!, correlations: correlations! } : null,
    stateDefined,
    certificates: { environmentNormAccount: closureNorms.map(v => q(sub(v, N))), localDensityAccount: localDensities?.map((rho, i) => q(matrixSquaredNorm(matrixSub(rho, denseLocal![i])))) ?? null },
    internal: { state, N, parameterNorm, leftEnvironments, rightEnvironments },
  };
}
export function mpsModel(value: MpsParameters) {
  const parameters = snapshotMpsParameters(value), p = parameters, sites = p.sites, bonds = Array.from({ length: sites - 1 }, (_, i) => p[`bond${i}`]), dimensions = [1, ...bonds, 1];
  const tensors: Tensor[] = Array.from({ length: sites }, (_, site) => [0, 1].map(s => Array.from({ length: dimensions[site] }, (_, left) => Array.from({ length: dimensions[site + 1] }, (_, right) => c(f(BigInt(p[`t${site}Real${left}${s}${right}`]), BigInt(p.tensorDenominator)), f(BigInt(p[`t${site}Imag${left}${s}${right}`]), BigInt(p.tensorDenominator)))))) as Tensor);
  const original = evaluate(tensors, bonds), changed = tensors.map(tensor => tensor.map(a => a.map(row => [...row]))) as Tensor[];
  let gauge = null, covariance = null;
  if (p.gaugeBond >= 0) {
    const bond = p.gaugeBond, size = bonds[bond], G = identity(size), inverse = identity(size), row = p.gaugeRow, column = p.gaugeColumn;
    const lambda = f(BigInt(p.gaugeScaleNumerator), BigInt(p.gaugeScaleDenominator)), eta = c(f(BigInt(p.gaugeShearReal), BigInt(p.gaugeShearDenominator)), f(BigInt(p.gaugeShearImag), BigInt(p.gaugeShearDenominator)));
    G[row][row] = c(lambda); inverse[row][row] = c(div(one, lambda));
    if (row !== column) { G[row][column] = scaleC(eta, lambda); inverse[row][column] = scaleC(eta, int(-1)); }
    changed[bond] = tensors[bond].map(a => product(a, G)) as Tensor;
    changed[bond + 1] = tensors[bond + 1].map(a => product(inverse, a)) as Tensor;
    gauge = { bond, order: 'G=diagonal-scale-times-shear', matrix: mq(G), inverse: mq(inverse), inverseResidualSquared: q(matrixSquaredNorm(matrixSub(product(G, inverse), identity(size)))) };
    covariance = { G, inverse, cut: bond + 1 };
  }
  const gauged = evaluate(changed, bonds);
  const gaugeResiduals = {
    stateDifferenceSquared: q(norm2(original.internal.state.map((v, i) => cs(v, gauged.internal.state[i])))),
    normAccount: q(sub(gauged.internal.N, original.internal.N)),
    leftEnvironmentCovariance: covariance ? q(matrixSquaredNorm(matrixSub(gauged.internal.leftEnvironments[covariance.cut], product(product(adjoint(covariance.G), original.internal.leftEnvironments[covariance.cut]), covariance.G)))) : null,
    rightEnvironmentCovariance: covariance ? q(matrixSquaredNorm(matrixSub(gauged.internal.rightEnvironments[covariance.cut], product(product(conjugate(covariance.inverse), original.internal.rightEnvironments[covariance.cut]), transpose(covariance.inverse))))) : null,
  };
  const removeInternal = ({ raw, physical, stateDefined, certificates }: ReturnType<typeof evaluate>) => ({ ...raw, physical, stateDefined, exactResiduals: certificates });
  return {
    parameters, sites, physicalDimensionPerSite: 2, dimension: 2 ** sites, bondDimensions: bonds, virtualDimensionsIncludingBoundaries: dimensions,
    arithmetic: 'exact-complex-rational', normalizationPrecisionBits: 512,
    basisConvention: 'n0-most-significant;A[site][physical][left][right];open-boundaries-one',
    original: removeInternal(original), gauged: removeInternal(gauged), gauge,
    validity: { structurallyValid: true, physicalStateDefined: original.stateDefined, reason: original.stateDefined ? 'nonzero-contracted-vector' : 'zero-contracted-vector-cannot-normalize' },
    cost: { activeComplexTensorEntries: tensors.reduce((n, tensor) => n + tensor.reduce((s, a) => s + a.length * a[0].length, 0), 0), paddedComplexTensorEntries: 128, denseComplexStateAmplitudes: 2 ** sites, prefixDenseMultiplyAccumulateSlots: tensors.reduce((n, a, i) => n + 2 ** (i + 1) * a[0].length * a[0][0].length, 0), normTransferDenseSummandSlots: tensors.reduce((n, a) => n + 2 * a[0].length ** 2 * a[0][0].length ** 2, 0), countsAreAlgorithmicSlotsNotRuntimeOrCompressionGuarantees: true },
    exactResiduals: gaugeResiduals,
    approximationContract: '1..4 ordered qubits with open-boundary bonds1..4;269 owned bounded integer fields and a positive common tensor denominator. Exact amplitudes,bra-ket transfer environments,all-cut norm closures,local density and all-pair Pauli correlations,canonicality Gram residuals and exact coefficient ranks. A nonzero tensor at every site does not imply nonzero global contraction;zero contractions retain raw algebra but physical normalized outputs are null. One selected real nonzero scale and complex shear gauge with an exact inverse preserves the full vector;virtual environments and tensor coordinate norms can change. No SVD/automatic canonicalization,Schmidt spectrum or truncation,DMRG/time evolution,ground state,thermodynamic limit or guaranteed compression is supplied.',
  };
}
