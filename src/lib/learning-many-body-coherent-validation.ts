/** Coherent-state attribution with an independent alternating-exp certificate. */
import { readManyBodyInput, manyBodyInputKey, type ManyBodyInput } from './learning-many-body-input';
import type { ManyBodySuccess } from './learning-many-body-protocol';
import { bosonFockParameterKeys } from './learning-boson-fock-input';
import { snapshotManyBodyWire } from './learning-many-body-owned-wire';
import { bosonCoherentResultContract, bosonPreparedResultContract } from './learning-many-body-contracts';
import { bosonFockExpected } from './learning-many-body-boson-validation';
import { expectedRational as q, matchManyBodyExpected, type ManyBodyExpected as Expected } from './learning-many-body-wire-match';
import { f, int, zero, one, two, add, sub, mul, div, sum } from './learning-tensor-wire-math';
import { field, fieldZero as Z, fieldOne as O, fieldInt as integer, fieldScale, fieldSum, expectedField as fq } from './learning-many-body-field-wire';
import { fc, fcMultiply, fcConjugate, fcScale, expectedFieldComplex as cq, expectedFieldNormalizedComplex as normalized, type FieldComplex } from './learning-many-body-field-matrix';
import { coherentExpCertificate, coherentScaleRange, coherentComplementRange, coherentRetainedRange, expectedCoherentRange, expectedCoherentAmplitude, expectedCoherentZeroResidual, expectedCoherentOverlap, expectedCoherentDistance } from './learning-many-body-coherent-wire';

function coherentExpected(p: Readonly<Record<string, number>>): Expected {
  const cutoffs = [p.cutoff0, ...(p.modes === 2 ? [p.cutoff1] : [])];
  const alpha = cutoffs.map((_, i) => fc(field(f(BigInt(p[`alphaReal${i}`]), BigInt(p[`alphaDenominator${i}`]))), field(f(BigInt(p[`alphaImag${i}`]), BigInt(p[`alphaDenominator${i}`])))));
  const mu = alpha.map(a => add(mul(a.re[0], a.re[0]), mul(a.im[0], a.im[0]))), totalMu = sum(mu);
  const inverseRoots = [O, O, field(zero, div(one, two)), field(zero, zero, div(one, int(3)))];
  const localCoefficients = cutoffs.map((cutoff, i) => {
    const values = [fc(O)]; for (let n = 1; n <= cutoff; n++) values.push(fcScale(fcMultiply(values[n - 1], alpha[i]), inverseRoots[n])); return values;
  });
  const weights = cutoffs.map((cutoff, i) => {
    const values = [one]; for (let n = 1; n <= cutoff; n++) values.push(div(mul(values[n - 1], mu[i]), int(n))); return values;
  }), series = weights.map(sum), rawNorm = series.reduce(mul, one);
  const rawCoefficients = p.modes === 1 ? localCoefficients[0] : localCoefficients[0].flatMap(a => localCoefficients[1].map(b => fcMultiply(a, b)));
  const rawControls = Object.fromEntries(bosonFockParameterKeys.map(key => [key, key.startsWith('state') ? Number(key === 'stateReal0') : p[key]]));
  const { parameters, ...finite } = bosonFockExpected(rawControls, rawCoefficients); void parameters;
  const retained = coherentRetainedRange(totalMu, rawNorm), discarded = coherentComplementRange(retained);
  const intervalComplex = (a: FieldComplex, mu: ReturnType<typeof int>): Expected => ({ real: expectedCoherentAmplitude(a.re, mu), imaginary: expectedCoherentAmplitude(a.im, mu) });
  const perMode = cutoffs.map((cutoff, i) => {
    const keep = coherentRetainedRange(mu[i], series[i]), top = div(weights[i][cutoff], series[i]);
    // The exact coherent recurrence cancels at every retained n<C. Only
    // the boundary coefficient survives in (a_C-alpha)|raw>.
    const residual = localCoefficients[i].map((a, n) => n === cutoff ? fcScale(fcMultiply(alpha[i], a), integer(-1)) : fc(Z));
    return {
      mode: i, cutoff, alpha: cq(alpha[i]), mu: q(mu[i]), rawCoefficients: localCoefficients[i].map(cq), rawSeriesNorm: q(series[i]),
      infiniteState: { representation: 'exp(-mu/2)*sum(alpha^n/sqrt(n!),n>=0)', meanNumber: q(mu[i]), numberVariance: q(mu[i]), xMean: fq(fieldScale(field(zero, one), alpha[i].re[0])), pMean: fq(fieldScale(field(zero, one), alpha[i].im[0])), xVariance: q(div(one, two)), pVariance: q(div(one, two)) },
      unnormalizedProjection: { amplitudes: localCoefficients[i].map(a => intervalComplex(a, mu[i])), probabilities: weights[i].map(w => expectedCoherentRange(coherentScaleRange(coherentExpCertificate(mu[i]), w))), retainedProbability: expectedCoherentRange(keep), discardedProbability: expectedCoherentRange(coherentComplementRange(keep)) },
      normalizedProjection: { amplitudes: localCoefficients[i].map(a => normalized(a, series[i])), probabilities: weights[i].map(w => q(div(w, series[i]))), topProbability: q(top), finiteCommutatorExpectation: q(sub(one, mul(int(cutoff + 1), top))), annihilationEigenResidual: residual.map(a => normalized(a, series[i])), annihilationEigenResidualSquared: fq(field(mul(mu[i], top))) },
      exactResiduals: { coefficientSeriesNorm: q(zero), eigenResidualAccount: fq(Z) },
    };
  });
  const coefficient = (name: string) => field(div(int(p[name]), int(p.hDenominator))), energyTerms = [coefficient('offset')];
  for (let i = 0; i < p.modes; i++) {
    energyTerms.push(fieldScale(coefficient(`onsite${i}`), mu[i]), fieldScale(coefficient(`selfDensity${i}`), div(mul(mu[i], mu[i]), two)));
    energyTerms.push(fieldScale(fcMultiply(fc(coefficient(`driveReal${i}`), coefficient(`driveImag${i}`)), fcConjugate(alpha[i])).re, two));
  }
  if (p.modes === 2) {
    energyTerms.push(fieldScale(coefficient('crossDensity'), mul(mu[0], mu[1])));
    energyTerms.push(fieldScale(fcMultiply(fc(coefficient('hoppingReal'), coefficient('hoppingImag')), fcMultiply(fcConjugate(alpha[0]), alpha[1])).re, two));
  }
  return {
    parameters: { ...p }, preparationKind: 'product-coherent-rectangular-cutoff', modes: p.modes, cutoffs, totalMu: q(totalMu), perMode,
    exactRawCoefficients: rawCoefficients.map(cq), rawSeriesNorm: q(rawNorm),
    infiniteState: { representation: 'tensor-product-of-infinite-coherent-series', normalized: true, meanTotalNumber: q(totalMu), varianceTotalNumber: q(totalMu), energyForSameNormallyOrderedHamiltonian: fq(fieldSum(energyTerms)), noGroundCertificate: true },
    unnormalizedProjection: { normalizationApplied: false, amplitudes: rawCoefficients.map(a => intervalComplex(a, totalMu)), normSquared: expectedCoherentRange(retained), distanceSquaredToInfinite: expectedCoherentRange(discarded) },
    normalizedProjection: { ...finite, approximationContract: bosonPreparedResultContract, normalizationApplied: true, fidelitySquaredToInfinite: expectedCoherentRange(retained), overlapWithInfinite: expectedCoherentOverlap(retained), distanceSquaredToInfinite: expectedCoherentDistance(retained) },
    truncation: { retainedProbability: expectedCoherentRange(retained), discardedProbability: expectedCoherentRange(discarded), productOfModeRetained: expectedCoherentRange(retained), jointMinusProductRetained: expectedCoherentZeroResidual() },
    exactResiduals: { coefficientSeriesNorm: q(zero) },
    certificate: { exponentialPrecisionBits: 768, exponentialGuardBits: 32, outputPrecisionBits: 512, exponentialMethod: 'range-reduced-positive-Taylor-bound-inverse-and-squaring', maximumPerModeMu: 16, maximumTotalMu: 32, finiteCoefficients: 'exact-Q-sqrt2-sqrt3', normalization: 'rational-series-norm-before-directed-square-root' },
    approximationContract: bosonCoherentResultContract,
  };
}
export type ManyBodyCoherentSuccess = Extract<ManyBodySuccess, { kind: 'boson-coherent' }>;
export function readBosonCoherentSuccessForInput(value: unknown, original: ManyBodyInput, requestId: number): ManyBodyCoherentSuccess | null {
  try {
    if (!Number.isSafeInteger(requestId) || requestId <= 0) return null;
    const input = readManyBodyInput(original); if (!input || input.kind !== 'boson-coherent') return null;
    const packet = snapshotManyBodyWire(value);
    const expected: Expected = { version: 1, requestId, ok: true, kind: 'boson-coherent', input: { kind: input.kind, parameters: { ...input.parameters } }, inputKey: manyBodyInputKey(input), result: coherentExpected(input.parameters) };
    return matchManyBodyExpected(packet, expected) ? packet as ManyBodyCoherentSuccess : null;
  } catch { return null; }
}
