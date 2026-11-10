/** Exact finite spatial preparations; no stochastic time or empirical sampling. */
import { markovFraction as fraction, exactMarkovObservation as observe, type MarkovFraction as F } from './learning-finite-markov-exact';
import type { PartitionObservation as Q } from './learning-partition-exact-core';
export interface SpatialSelfAveragingParameters {
 siteCount: number;
 occupiedWeight: number;
 emptyWeight: number;
 commonNumerator: number;
 commonDenominator: number;
 commonScaling: 'fixed' | 'inverse-size' | 'inverse-size-squared';
 epsilonNumerator: number;
 epsilonDenominator: number;
}
export type SpatialLimit = { kind: 'finite'; value: Q } | { kind: 'positive-infinity' | 'not-applicable' };
const Z = BigInt(0), O = BigInt(1);
const abs = (n: bigint) => n < Z ? -n : n;
const output = (q: F) => observe(q.n, q.d);
const plus = (a: F, b: F): F => fraction(a.n * b.d + b.n * a.d, a.d * b.d);
const minus = (a: F, b: F): F => fraction(a.n * b.d - b.n * a.d, a.d * b.d);
const times = (a: F, b: F): F => fraction(a.n * b.n, a.d * b.d);
const over = (a: F, b: F): F => fraction(a.n * b.d, a.d * b.n);
const atMostOne = (q: F): F => q.n >= q.d ? fraction(O) : q;
const integer = (value: unknown, minimum: number, maximum = 1000000): value is number => typeof value === 'number' && Number.isInteger(value) && value >= minimum && value <= maximum;
export function isSpatialSelfAveragingParameters(value: unknown): value is SpatialSelfAveragingParameters {
 if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
 const p = value as Record<string, unknown>, keys = ['siteCount', 'occupiedWeight', 'emptyWeight', 'commonNumerator', 'commonDenominator', 'commonScaling', 'epsilonNumerator', 'epsilonDenominator'];
 return Object.keys(p).length === keys.length && keys.every(k => Object.prototype.hasOwnProperty.call(p, k))
  && integer(p.siteCount, 1, 512) && integer(p.occupiedWeight, 0) && integer(p.emptyWeight, 0) && p.occupiedWeight + p.emptyWeight > 0
  && integer(p.commonNumerator, 0) && integer(p.commonDenominator, 1) && p.commonNumerator <= p.commonDenominator
  && integer(p.epsilonNumerator, 1) && integer(p.epsilonDenominator, 1)
  && ['fixed', 'inverse-size', 'inverse-size-squared'].includes(p.commonScaling as string);
}
export function spatialDensitySelfAveragingModel(input: SpatialSelfAveragingParameters) {
 if (!isSpatialSelfAveragingParameters(input)) throw new RangeError('Expected 1..512 sites, nonnegative integer occupancy weights with positive sum, 0≤common numerator≤positive denominator, a supported common-mode path and positive rational epsilon; integer weights/denominators at most 1000000');
 const parameters = Object.freeze({ siteCount: input.siteCount, occupiedWeight: input.occupiedWeight + 0, emptyWeight: input.emptyWeight + 0,
  commonNumerator: input.commonNumerator + 0, commonDenominator: input.commonDenominator, commonScaling: input.commonScaling,
  epsilonNumerator: input.epsilonNumerator, epsilonDenominator: input.epsilonDenominator });
 const A = BigInt(input.occupiedWeight), B = BigInt(input.emptyWeight), W = A + B, C = BigInt(input.commonNumerator), D = BigInt(input.commonDenominator);
 const E = BigInt(input.epsilonNumerator), V = BigInt(input.epsilonDenominator), p = fraction(A, W), v = fraction(A * B, W * W), epsilonSquared = fraction(E * E, V * V);
 const power = input.commonScaling === 'fixed' ? 0 : input.commonScaling === 'inverse-size' ? 1 : 2;
 const inEvent = (count: bigint, n: bigint) => abs(count * W - n * A) * V >= E * n * W;
 const collectiveEvent = fraction((inEvent(Z, O) ? B : Z) + (inEvent(O, O) ? A : Z), W);
 function calculate(siteCount: number, includeDistribution: boolean) {
  const N = BigInt(siteCount), DN = D * N ** BigInt(power), denominator = W ** N, mixtureDenominator = DN * denominator, sharedBase = denominator / W;
  const c = fraction(C, DN), independentWeight = fraction(DN - C, DN);
  let coefficient = B ** N, sum = Z, first = Z, second = Z, tail = Z, iidTail = Z, positiveMassCounts = 0;
  const eventCounts: number[] = [];
  const distribution: { count: number; density: Q; deviation: Q; independentProbability: Q; collectiveProbability: Q; probability: Q; inDeviationEvent: boolean }[] = [];
  for (let k = 0; k <= siteCount; k++) {
   // With B=0 the law is deterministic; otherwise the integer recurrence
   // computes C(N,k) A^k B^(N-k) without floating-point probabilities.
   const iid = B === Z ? (k === siteCount ? denominator : Z) : coefficient;
   const shared = k === 0 ? B * sharedBase : k === siteCount ? A * sharedBase : Z;
   const mass = (DN - C) * iid + C * shared, K = BigInt(k), event = inEvent(K, N);
   sum += mass; first += K * mass; second += K * K * mass;
   if (mass > Z) positiveMassCounts++;
   if (event) { tail += mass; iidTail += iid; eventCounts.push(k); }
   if (includeDistribution) distribution.push({ count: k, density: observe(K, N), deviation: observe(K * W - N * A, N * W),
    independentProbability: observe(iid, denominator), collectiveProbability: observe(shared, denominator), probability: observe(mass, mixtureDenominator), inDeviationEvent: event });
   if (B > Z && k < siteCount) coefficient = coefficient * (N - K) * A / ((K + O) * B);
  }
  const factor = fraction(DN + (N - O) * C, DN), densityVariance = fraction(A * B * (DN + (N - O) * C), W * W * N * DN);
  const countVariance = times(densityVariance, fraction(N * N)), iidVariance = fraction(A * B, W * W * N);
  const fromDistribution = fraction(second * mixtureDenominator - first * first, mixtureDenominator * mixtureDenominator * N * N);
  const exactEvent = fraction(tail, mixtureDenominator), independentEvent = fraction(iidTail, denominator);
  const independentBound = atMostOne(over(iidVariance, epsilonSquared)), varianceBound = atMostOne(over(densityVariance, epsilonSquared));
  const independentEventContribution = times(independentWeight, independentEvent), commonEventContribution = times(c, collectiveEvent);
  const mixtureBound = plus(times(independentWeight, independentBound), commonEventContribution), diagonalSum = times(v, fraction(N)), crossSum = times(v, fraction(N * (N - O) * C, DN));
  const summary = { siteCount, commonWeight: output(c), meanDensity: output(p), densityVariance: output(densityVariance), scaledDensityVariance: output(times(densityVariance, fraction(N))),
   independentDensityVariance: output(iidVariance), eventProbability: output(exactEvent), independentEventProbability: output(independentEvent), collectiveEventProbability: output(collectiveEvent),
   chebyshevBound: output(varianceBound), mixtureAwareBound: output(mixtureBound), independentReferenceBound: output(independentBound),
   varianceEquivalentCount: v.n === Z ? null : output(over(fraction(N), factor)) };
  return { summary, distribution,
   siteVariance: output(v), distinctPairCovariance: siteCount === 1 ? null : output(times(c, v)),
   distinctPairCorrelation: siteCount === 1 || v.n === Z ? null : output(c),
   meanCount: output(times(p, fraction(N))), countVariance: output(countVariance), densitySecondMoment: output(plus(times(p, p), densityVariance)),
   diagonalCovarianceMultiplicity: siteCount, orderedCrossCovarianceMultiplicity: siteCount * (siteCount - 1),
   diagonalCovarianceSum: output(diagonalSum), orderedCrossCovarianceSum: output(crossSum), covarianceSumDensityVariance: output(over(plus(diagonalSum, crossSum), fraction(N * N))),
   designFactor: output(factor), varianceRatioToIndependent: v.n === Z ? null : output(factor),
   distributionNormalization: observe(sum, mixtureDenominator), densityMeanFromDistribution: observe(first, mixtureDenominator * N), densityVarianceFromDistribution: output(fromDistribution), varianceIdentityResidual: output(minus(fromDistribution, densityVariance)),
   distributionMomentsMatch: sum === mixtureDenominator && first * W === mixtureDenominator * N * A && fromDistribution.n * densityVariance.d === densityVariance.n * fromDistribution.d,
   independentEventContribution: output(independentEventContribution), collectiveEventContribution: output(commonEventContribution), eventMixtureIdentityResidual: output(minus(exactEvent, plus(independentEventContribution, commonEventContribution))),
   chebyshevSlack: output(minus(varianceBound, exactEvent)), mixtureAwareSlack: output(minus(mixtureBound, exactEvent)),
   eventProbabilityMinusIndependent: output(minus(exactEvent, independentEvent)), eventTailRatioToIndependent: independentEvent.n === Z ? null : output(over(exactEvent, independentEvent)),
   independentReferenceBoundFailsForMixture: exactEvent.n * independentBound.d > independentBound.n * exactEvent.d,
   eventCounts, eventContainsPositiveMass: tail > Z, eventIsWholeGrid: eventCounts.length === siteCount + 1, positiveMassCount: positiveMassCounts,
   distributionIsProductForThisSize: siteCount === 1 || C === Z || v.n === Z,
   approximationContract: 'Finite spatial occupancy law: one common-mode choice for the entire system, then either N independent Bernoulli(p) sites or one Bernoulli(p) value copied to every site. A common-mode choice is not made separately for each site. p and the base common weight are fixed along the specified size path; epsilon is fixed and strictly positive. The deviation event includes equality on the exact rational lattice. Covariance sums count ordered distinct pairs. A missing distinct pair or normalized correlation is not zero. The independent-only probability and Chebyshev bound are reference values, not general guarantees for the mixture. Variance-equivalent count is a same-observable variance comparison and is undefined at deterministic occupancy. Counts in the event may have zero preparation mass. Distribution moments are summed separately and checked against the closed formula.' };
 }
 const sizes: number[] = []; for (let n = 1; n <= input.siteCount; n *= 2) sizes.push(n); if (sizes[sizes.length - 1] !== input.siteCount) sizes.push(input.siteCount);
 const current = calculate(input.siteCount, true), history = sizes.map(n => n === input.siteCount ? current.summary : calculate(n, false).summary);
 const cInfinity = input.commonScaling === 'fixed' ? fraction(C, D) : fraction(Z), limitVariance = times(cInfinity, v), finite = (value: F): SpatialLimit => ({ kind: 'finite', value: output(value) });
 const scaledLimit: SpatialLimit = v.n > Z && input.commonScaling === 'fixed' && C > Z ? { kind: 'positive-infinity' }
  : finite(input.commonScaling === 'inverse-size' ? times(v, fraction(D + C, D)) : v);
 const countLimit: SpatialLimit = v.n === Z ? { kind: 'not-applicable' } : input.commonScaling === 'fixed' && C > Z ? finite(fraction(D, C)) : { kind: 'positive-infinity' };
 const fractionLimit: SpatialLimit = v.n === Z ? { kind: 'not-applicable' } : finite(input.commonScaling === 'fixed' && C > Z ? fraction(Z) : input.commonScaling === 'inverse-size' ? fraction(D, D + C) : fraction(O));
 const atoms: { density: F; probability: F }[] = [];
 for (const [density, probability] of [[fraction(Z), times(cInfinity, fraction(B, W))], [p, minus(fraction(O), cInfinity)], [fraction(O), times(cInfinity, p)]]) {
  if (probability.n === Z) continue;
  const existing = atoms.find(atom => atom.density.n * density.d === density.n * atom.density.d);
  if (existing) existing.probability = plus(existing.probability, probability); else atoms.push({ density, probability });
 }
 atoms.sort((a, b) => a.density.n * b.density.d < b.density.n * a.density.d ? -1 : 1);
 return { parameters, arithmetic: 'exact-bigint-rational' as const, fullyResolved: true, history, current,
  limits: { commonWeight: output(cInfinity), densityVariance: output(limitVariance), scaledDensityVariance: scaledLimit,
   deviationEventProbability: output(times(cInfinity, collectiveEvent)), varianceEquivalentCount: countLimit, varianceEquivalentFraction: fractionLimit,
   distinctPairCorrelation: v.n === Z ? null : output(cInfinity), densityLawAtoms: atoms.map(atom => ({ density: output(atom.density), probability: output(atom.probability) })) },
  classification: { deterministicOccupancy: v.n === Z, distinctPairExistsAtCurrentSize: input.siteCount >= 2,
   selfAveragesInProbabilityAlongPath: limitVariance.n === Z, commonModeVanishesAlongPath: cInfinity.n === Z,
   countVarianceGrowth: v.n === Z ? 'zero' as const : input.commonScaling === 'fixed' && C > Z ? 'quadratic-in-site-count' as const : 'linear-in-site-count' as const },
  approximationContract: 'Exact finite spatial mixture for N=1..512 with integer occupancy weights in 0..1000000 and positive sum; common numerator/denominator specify 0≤c≤1 with denominator 1..1000000; c_N is c, c/N or c/N². Positive epsilon numerator/denominator are at most 1000000 and are not clipped to the occupancy range. Finite probabilities, signed changes, moments, covariances, event sums and bounds are exact BigInt rational points. Number and magnitude-log fields are display projections: finite nonzero underflow or outside-range tail ratios are never zero or mathematical infinity. History visits powers of two and the requested endpoint, with the same p,c,path,epsilon; it is not a time sequence or a sampled trajectory. The last summary belongs to the current exact distribution. Analytic limits concern the prescribed fixed-parameter spatial preparation family; no almost-sure coupling across different N is specified. The atomic limiting law is not a finite-N distribution or a fitted equilibrium phase diagram. Independence, Gibbs locality, physical temperature, a fluctuation-response identity, empirical MCMC effective sample size, or general AI sampler guarantees are not inferred. Equal leading variance scaling does not establish equal rare-event tails.' };
}
