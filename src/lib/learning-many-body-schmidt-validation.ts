/** Full Schmidt response ownership and independent spectral/truncation attribution. */
import { readManyBodyInput, manyBodyInputKey, type ManyBodyInput } from './learning-many-body-input';
import type { ManyBodySuccess } from './learning-many-body-protocol';
import { snapshotManyBodyWire } from './learning-many-body-owned-wire';
import { schmidtResultContract } from './learning-many-body-contracts';
import { expectedRational as q, expectedComplex as cq, expectedNormalized, expectedValidated, matchManyBodyExpected, type ManyBodyExpected as Expected } from './learning-many-body-wire-match';
import { f, zero, one, int, sub, mul, div, le, eq, neg, lo, hi, c, norm2, outer, partial, invariants, matrixNorm2, type F, type Matrix, type Range } from './learning-tensor-wire-math';
import { schmidtSpectrumCertificate, schmidtLogCertificate } from './learning-many-body-schmidt-spectral-wire';
import { wireRange as r, wireAdd as add, wireSub as subtract, wireMul as multiply, wireDiv as divide, wireNeg as negative, wireSqrt as sqrt, wireMatrix as matrix, wireZeros as zeros, wireMatrixAdd as plus, wireMatrixScale as scale, wireProduct as product, wireMatrixNormSquared as squaredNorm, wireMatrixSub as minus, wireSpectralProjector, type WireIntervalComplex, type WireIntervalMatrix } from './learning-many-body-schmidt-interval-wire';
import type { PartitionObservation as Q } from './learning-partition-exact-core';
const magnitude = (a: F) => a.n < BigInt(0) ? neg(a) : a;
function directed(target: Range, probability = false): Expected {
  return expectedValidated((value, checks) => {
    if (!checks.q(value)) return false; const v = value as Q, a = lo(v), b = hi(v);
    if (!le(a, target.lower) || !le(target.upper, b) || probability && (!le(zero, a) || !le(b, one))) return false;
    const bound = le(magnitude(target.lower), magnitude(target.upper)) ? magnitude(target.upper) : magnitude(target.lower);
    return le(mul(sub(b, a), f(BigInt(10) ** BigInt(125))), le(bound, one) ? one : bound);
  });
}
const observed = (a: WireIntervalComplex): Expected => ({ real: directed(a.re), imaginary: directed(a.im) });
const observedMatrix = (a: WireIntervalMatrix): Expected => a.map(row => row.map(observed));
const exactMatrix = (a: Matrix): Expected => a.map(row => row.map(cq));
const zeroMatrix = (n: number): Expected => observedMatrix(zeros(n));
const directedZero = () => directed(r(zero));
const directedZeroComplex = (): Expected => ({ real: directedZero(), imaginary: directedZero() });
function probability(target: Range): Expected {
  // Intersect a proved mathematical probability enclosure with [0,1].
  return directed({ lower: le(zero, target.lower) ? target.lower : zero, upper: le(target.upper, one) ? target.upper : one }, true);
}
function schmidtExpected(p: Readonly<Record<string, number>>): Expected {
  const state = Array.from({ length: 2 ** p.sites }, (_, i) => c(int(p[`stateReal${i}`]), int(p[`stateImag${i}`]))), N = norm2(state);
  const rho = outer(state).map(row => row.map(v => ({ re: div(v.re, N), im: div(v.im, N) }))), ln2 = schmidtLogCertificate(int(2));
  const cuts = Array.from({ length: p.sites - 1 }, (_, k) => {
    const cut = k + 1, leftDimension = 2 ** cut, rightDimension = state.length / leftDimension, coefficient = Array.from({ length: leftDimension }, (_, i) => state.slice(i * rightDimension, (i + 1) * rightDimension));
    const rhoLeft = partial(rho, leftDimension, rightDimension, 'a'), rhoRight = partial(rho, leftDimension, rightDimension, 'b');
    const smallerSide = leftDimension <= rightDimension ? 'left' : 'right', density = smallerSide === 'left' ? rhoLeft : rhoRight, size = density.length, inv = invariants(density), rank = inv.positivePrincipalRank;
    const certificates = schmidtSpectrumCertificate(inv.characteristic).roots, roots = certificates.map(v => r(v.lower, v.upper));
    const spectrum = certificates.map((v, i) => ({ multiplicity: v.multiplicity, zero: eq(v.lower, zero) && eq(v.upper, zero), weight: roots[i], projector: wireSpectralProjector(density, roots, i), certificate: v }));
    if (spectrum.filter(v => !v.zero).reduce((n, v) => n + v.multiplicity, 0) !== rank) throw new RangeError('Independent Schmidt rank account failed');
    let remaining = Math.min(p.retainedRank, rank), kept = r(zero), retained = zeros(size), entropy = r(zero), boundaryDegenerate = false;
    const allocation = spectrum.map((v, groupIndex) => {
      const count = v.zero ? 0 : Math.min(remaining, v.multiplicity); remaining -= count;
      kept = add(kept, multiply(r(int(count)), v.weight));
      if (count > 0 && count < v.multiplicity) boundaryDegenerate = true;
      if (count === v.multiplicity && !v.zero) retained = plus(retained, v.projector);
      if (!v.zero) {
        const a = schmidtLogCertificate(v.certificate.lower), b = schmidtLogCertificate(v.certificate.upper);
        entropy = add(entropy, negative(multiply(r(int(v.multiplicity)), multiply(v.weight, r(a.lower, b.upper)))));
      }
      return { groupIndex, keptMultiplicity: count, discardedMultiplicity: v.zero ? 0 : v.multiplicity - count };
    });
    if (p.retainedRank >= rank) kept = r(one);
    const discarded = subtract(r(one), kept), retainedPositiveRank = Math.min(p.retainedRank, rank), normalized = retainedPositiveRank > 0;
    let chosen: Expected = null;
    if (!boundaryDegenerate) {
      const transpose = retained[0].map((_, j) => retained.map(row => row[j]));
      const truncated = smallerSide === 'left' ? product(retained, matrix(coefficient)) : product(matrix(coefficient), transpose);
      const rest = minus(matrix(coefficient), truncated), normInput = sqrt(r(N)), normTruncated = normalized ? sqrt(multiply(r(N), kept)) : null;
      chosen = {
        retainedPositiveSupportProjector: observedMatrix(retained), truncatedRawCoefficientMatrix: observedMatrix(truncated),
        amplitudesOnNormalizedInput: scale(truncated, divide(r(one), normInput)).flat().map(observed),
        normalizedAmplitudes: normTruncated ? scale(truncated, divide(r(one), normTruncated)).flat().map(observed) : null,
        directSquaredError: directed(divide(squaredNorm(rest), r(N))), directRetainedNormSquared: directed(divide(squaredNorm(truncated), r(N))),
        directedResiduals: { errorAccount: directedZero(), retainedNormAccount: directedZero(), orthogonality: directedZeroComplex(), projectorIdempotence: zeroMatrix(size), projectorHermiticity: zeroMatrix(size) },
      };
    }
    const purity = matrixNorm2(density);
    return {
      cut, leftDimension, rightDimension, smallerSide, smallerDimension: size, coefficientMatrix: exactMatrix(coefficient), rhoLeft: exactMatrix(rhoLeft), rhoRight: exactMatrix(rhoRight), schmidtRank: rank,
      leftZeroMultiplicity: leftDimension - rank, rightZeroMultiplicity: rightDimension - rank, characteristicPolynomial: inv.characteristic.map(q), purity: q(purity), linearEntropy: q(sub(one, purity)), entropyNats: directed(entropy), entropyBits: directed(divide(entropy, r(ln2.lower, ln2.upper))),
      spectrum: spectrum.map(v => ({ weight: probability(v.weight), singularValue: directed(sqrt(v.weight)), multiplicity: v.multiplicity, zeroWeight: v.zero, projectorOnSmallerPhysicalSide: observedMatrix(v.projector) })),
      truncation: {
        rankBudget: p.retainedRank, retainedPositiveRank, allocation, boundaryDegenerate, optimalUnnormalizedStateUnique: !boundaryDegenerate, normalizedStateDefinedForEveryOptimum: normalized, chosenStateAvailable: chosen !== null,
        retainedWeight: probability(kept), discardedWeight: probability(discarded), optimalUnnormalizedSquaredError: probability(discarded), optimalNormalizedFidelitySquared: normalized ? probability(kept) : null,
        optimalNormalizedDistanceSquared: normalized ? directed(multiply(r(int(2)), subtract(r(one), sqrt(kept)))) : null,
        chosen, noArbitraryDegenerateEigenvectorsChosen: true,
      },
      exactResiduals: { leftHermiticity: q(zero), rightHermiticity: q(zero), leftTrace: cq(c(zero)), rightTrace: cq(c(zero)), purityAgreement: q(zero) },
      directedResiduals: {
        projectorCompleteness: zeroMatrix(size), densityReconstruction: zeroMatrix(size), probabilityMass: directedZero(), purityAccount: directedZero(),
        spectralGroups: spectrum.map(() => ({ hermiticity: zeroMatrix(size), idempotence: zeroMatrix(size), eigenEquation: zeroMatrix(size), traceMultiplicity: directedZeroComplex() })),
      },
    };
  });
  return { parameters: { ...p }, sites: p.sites, dimension: state.length, state: state.map(cq), normalizedState: state.map(v => expectedNormalized(v, N)), rawNormSquared: q(N), selectedCut: p.selectedCut, selectedCutIndex: p.selectedCut - 1, cuts,
    arithmetic: 'exact-density-algebra-and-certified-spectral-projectors', rootIsolationBits: 4096, intervalWorkBits: 2048, logarithmPrecisionBits: 768, outputPrecisionBits: 512, approximationContract: schmidtResultContract };
}
export type ManyBodySchmidtSuccess = Extract<ManyBodySuccess, { kind: 'schmidt' }>;
export function readSchmidtSuccessForInput(value: unknown, original: ManyBodyInput, requestId: number): ManyBodySchmidtSuccess | null {
  try {
    if (!Number.isSafeInteger(requestId) || requestId <= 0) return null;
    const input = readManyBodyInput(original); if (!input || input.kind !== 'schmidt') return null;
    const packet = snapshotManyBodyWire(value), expected: Expected = { version: 1, requestId, ok: true, kind: 'schmidt', input: { kind: input.kind, parameters: { ...input.parameters } }, inputKey: manyBodyInputKey(input), result: schmidtExpected(input.parameters) };
    return matchManyBodyExpected(packet, expected) ? packet as ManyBodySchmidtSuccess : null;
  } catch { return null; }
}
