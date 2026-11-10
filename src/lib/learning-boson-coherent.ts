/** Product coherent states, exact finite coefficients and certified Poisson tails. */
import { f, zero, one, add, sub, mul, div, sum, eq, le, q, sqrtRange, type F } from './learning-quantum-hilbert-core';
import { createTranscendentalContext } from './learning-interval-transcendentals';
import { createQuantumEvolutionContext } from './learning-quantum-evolution-interval';
import { bReal as r, bZero as z, bOne as o, bComplex as c, bCSub as cs, bCMul as cm, bConjugate as conjugate, bScale as scale, bNorm2 as norm2, bDiv as divide, bSub as subtract, bMul as times, bObserve as observeField, bCObserve as observeComplex, bRange as range, bNormalized as normalized, bEqual as equal, type BosonComplex as C, type BosonReal as R } from './learning-boson-algebraic';
import { bosonFockParameterKeys, type BosonFockParameters } from './learning-boson-fock-input';
import { evaluateBosonFockPreparedState } from './learning-boson-fock';
import { snapshotBosonCoherentParameters, type BosonCoherentParameters } from './learning-boson-coherent-input';
export { createBosonCoherentParameters, snapshotBosonCoherentParameters, isBosonCoherentParameters, bosonCoherentParameterKeys, type BosonCoherentParameters } from './learning-boson-coherent-input';
const int = (n: number) => f(BigInt(n));
const rational = (v: R): F => { if (v.slice(1).some(x => x.n !== BigInt(0))) throw new RangeError('Rational account expected'); return v[0]; };
export function bosonCoherentModel(value: BosonCoherentParameters) {
  const parameters = snapshotBosonCoherentParameters(value), p = parameters;
  const ctx = createQuantumEvolutionContext(), exp = createTranscendentalContext(768);
  const exponential = (x: F) => {
    if (!le(zero, x) || !le(x, int(32))) throw new RangeError('Nonnegative exponent<=32 required');
    const v = exp.expNegativeRatio({ coefficient: x.n, exponent: 0 }, { coefficient: x.d, exponent: 0 });
    return ctx.range({ lower: f(v.lower, exp.scale), upper: f(v.upper, exp.scale) });
  };
  const intervalComplex = (v: C) => ({ real: ctx.range(range(v.re)), imaginary: ctx.range(range(v.im)) });
  const alpha = Array.from({ length: p.modes }, (_, i) => c(r(f(BigInt(p[`alphaReal${i}`]), BigInt(p[`alphaDenominator${i}`]))), r(f(BigInt(p[`alphaImag${i}`]), BigInt(p[`alphaDenominator${i}`])))));
  const mu = alpha.map(v => rational(norm2([v]))), cutoffs = [p.cutoff0, ...(p.modes === 2 ? [p.cutoff1] : [])];
  const inverseRoots = [o, o, r(zero, f(BigInt(1), BigInt(2))), r(zero, zero, f(BigInt(1), BigInt(3)))];
  const localCoefficients = cutoffs.map((cutoff, i) => {
    const values = [c(o)]; for (let n = 1; n <= cutoff; n++) values.push(scale(cm(values[n - 1], alpha[i]), inverseRoots[n]));
    return values;
  });
  const weights = cutoffs.map((cutoff, i) => {
    const values = [one]; for (let n = 1; n <= cutoff; n++) values.push(div(mul(values[n - 1], mu[i]), int(n)));
    return values;
  }), series = weights.map(sum);
  const rawCoefficients = p.modes === 1 ? localCoefficients[0] : localCoefficients[0].flatMap(a => localCoefficients[1].map(b => cm(a, b)));
  const rawNorm = rational(norm2(rawCoefficients)), productSeries = series.reduce(mul, one), totalMu = sum(mu);
  if (!eq(rawNorm, productSeries)) throw new RangeError('Coherent coefficient norm and Poisson series disagree');
  const commonFactor = exponential(div(totalMu, int(2))), jointExp = exponential(totalMu);
  const retained = ctx.multiply(jointExp, ctx.from(rawNorm)), discarded = ctx.sub(ctx.one, retained);
  // These intersections use known exact probability ranges, not state corrections.
  const retainedObservation = ctx.probability(retained), discardedObservation = ctx.probability(discarded);
  const retainedRange = ctx.rationalRange(retained);
  const overlap = ctx.range({ lower: sqrtRange(le(zero, retainedRange.lower) ? retainedRange.lower : zero).lower, upper: sqrtRange(retainedRange.upper).upper });
  const normalizedDistance = ctx.multiply(ctx.from(int(2)), ctx.sub(ctx.one, overlap));
  const rawControls = Object.fromEntries(bosonFockParameterKeys.map(key => [key, key.startsWith('state') ? Number(key === 'stateReal0') : p[key]])) as BosonFockParameters;
  const finiteProjection = evaluateBosonFockPreparedState(rawControls, rawCoefficients);
  const perMode = cutoffs.map((cutoff, i) => {
    const decay = exponential(mu[i]), keep = ctx.multiply(decay, ctx.from(series[i]));
    const topProbability = div(weights[i][cutoff], series[i]);
    const residual = localCoefficients[i].map((v, n) => {
      const raisedInput = n < cutoff ? scale(localCoefficients[i][n + 1], divide(o, inverseRoots[n + 1])) : c(z);
      return cs(raisedInput, cm(alpha[i], v));
    });
    const eigenResidualSquared = divide(norm2(residual), r(series[i]));
    const eigenResidualAccount = subtract(eigenResidualSquared, r(mul(mu[i], topProbability)));
    if (!equal(eigenResidualAccount, z)) throw new RangeError('Finite coherent annihilator residual account failed');
    const localFactor = exponential(div(mu[i], int(2)));
    return {
      mode: i, cutoff, alpha: observeComplex(alpha[i]), mu: q(mu[i]), rawCoefficients: localCoefficients[i].map(observeComplex), rawSeriesNorm: q(series[i]),
      infiniteState: { representation: 'exp(-mu/2)*sum(alpha^n/sqrt(n!),n>=0)', meanNumber: q(mu[i]), numberVariance: q(mu[i]), xMean: observeField(times(r(int(2)), times(inverseRoots[2], alpha[i].re))), pMean: observeField(times(r(int(2)), times(inverseRoots[2], alpha[i].im))), xVariance: q(div(one, int(2))), pVariance: q(div(one, int(2))) },
      unnormalizedProjection: { amplitudes: localCoefficients[i].map(v => ctx.observeComplex(ctx.complexScale(intervalComplex(v), localFactor))), probabilities: weights[i].map(w => ctx.probability(ctx.multiply(decay, ctx.from(w)))), retainedProbability: ctx.probability(keep), discardedProbability: ctx.probability(ctx.sub(ctx.one, keep)) },
      normalizedProjection: { amplitudes: localCoefficients[i].map(v => normalized(v, series[i])), probabilities: weights[i].map(w => q(div(w, series[i]))), topProbability: q(topProbability), finiteCommutatorExpectation: q(sub(one, mul(int(cutoff + 1), topProbability))), annihilationEigenResidual: residual.map(v => normalized(v, series[i])), annihilationEigenResidualSquared: observeField(eigenResidualSquared) },
      exactResiduals: { coefficientSeriesNorm: q(sub(rational(norm2(localCoefficients[i])), series[i])), eigenResidualAccount: observeField(eigenResidualAccount) },
    };
  });
  const productRetained = mu.reduce((v, value, i) => ctx.multiply(v, ctx.multiply(exponential(value), ctx.from(series[i]))), ctx.one);
  let infiniteEnergy = r(div(int(p.offset), int(p.hDenominator)));
  const coefficient = (key: string) => r(div(int(p[key]), int(p.hDenominator)));
  const plus = (v: R) => { infiniteEnergy = [add(infiniteEnergy[0], v[0]), add(infiniteEnergy[1], v[1]), add(infiniteEnergy[2], v[2]), add(infiniteEnergy[3], v[3])]; };
  for (let i = 0; i < p.modes; i++) {
    plus(times(coefficient(`onsite${i}`), r(mu[i])));
    plus(times(coefficient(`selfDensity${i}`), r(div(mul(mu[i], mu[i]), int(2)))));
    const drive = c(coefficient(`driveReal${i}`), coefficient(`driveImag${i}`));
    plus(times(r(int(2)), cm(drive, conjugate(alpha[i])).re));
  }
  if (p.modes === 2) {
    plus(times(coefficient('crossDensity'), r(mul(mu[0], mu[1]))));
    const hopping = c(coefficient('hoppingReal'), coefficient('hoppingImag'));
    plus(times(r(int(2)), cm(hopping, cm(conjugate(alpha[0]), alpha[1])).re));
  }
  return {
    parameters, preparationKind: 'product-coherent-rectangular-cutoff', modes: p.modes, cutoffs, totalMu: q(totalMu), perMode,
    exactRawCoefficients: rawCoefficients.map(observeComplex), rawSeriesNorm: q(rawNorm),
    infiniteState: { representation: 'tensor-product-of-infinite-coherent-series', normalized: true, meanTotalNumber: q(totalMu), varianceTotalNumber: q(totalMu), energyForSameNormallyOrderedHamiltonian: observeField(infiniteEnergy), noGroundCertificate: true },
    unnormalizedProjection: { normalizationApplied: false, amplitudes: rawCoefficients.map(v => ctx.observeComplex(ctx.complexScale(intervalComplex(v), commonFactor))), normSquared: retainedObservation, distanceSquaredToInfinite: discardedObservation },
    normalizedProjection: { ...finiteProjection, normalizationApplied: true, fidelitySquaredToInfinite: retainedObservation, overlapWithInfinite: ctx.observation(overlap), distanceSquaredToInfinite: ctx.observation(normalizedDistance) },
    truncation: { retainedProbability: retainedObservation, discardedProbability: discardedObservation, productOfModeRetained: ctx.probability(productRetained), jointMinusProductRetained: ctx.observation(ctx.sub(retained, productRetained)) },
    exactResiduals: { coefficientSeriesNorm: q(sub(rawNorm, productSeries)) },
    certificate: { exponentialPrecisionBits: 768, exponentialGuardBits: 32, outputPrecisionBits: 512, exponentialMethod: 'range-reduced-positive-Taylor-bound-inverse-and-squaring', maximumPerModeMu: 16, maximumTotalMu: 32, finiteCoefficients: 'exact-Q-sqrt2-sqrt3', normalization: 'rational-series-norm-before-directed-square-root' },
    approximationContract: 'Product coherent input with exact rational complex alpha and per-mode0<=mu<=16;1..2 modes,maximum occupancies0..3. Infinite coherent series,unnormalized rectangular projection and normalized projection are distinct. Poisson retained/tail probabilities use directed768-bit exponential enclosures with512-bit output;exact finite coefficients and all cutoff observables are retained. Finite annihilator eigenstate failure and boundary CCR are explicit. No finite displacement-operator exponential is substituted for projected infinite coherence;no time evolution,cutoff convergence or infinite-space ground certificate is supplied.',
  };
}
