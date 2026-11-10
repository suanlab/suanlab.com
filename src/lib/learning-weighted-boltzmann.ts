/** Finite Boltzmann laws relative to explicit nonnegative reference weights.
 * Positive real weights are not automatically physical state multiplicities.
 * No log/exp binary64 approximation is used to certify an enclosure.
 */
import { binaryProduct, binarySum, type BinaryValue } from './learning-binary-products';
import { createTranscendentalContext } from './learning-interval-transcendentals';
import { partitionObservationFromRange, type PartitionObservation } from './learning-partition-exact-core';
import { canonicalBoltzmannConstant as kB } from './learning-canonical-spectrum';
export interface WeightedBoltzmannParameters {
  energyLevels: readonly number[];
  referenceWeights: readonly number[];
  energyOffset: number;
  temperature: number;
}
interface Fraction { n: bigint; d: bigint }
interface Interval { lower: Fraction; upper: Fraction }
const zero = BigInt(0), one = BigInt(1);
const abs = (n: bigint) => n < zero ? -n : n;
function gcd(a: bigint, b: bigint): bigint { a = abs(a); while (b) { const t = a % b; a = b; b = t; } return a; }
function fraction(n: bigint, d = one): Fraction {
  if (d <= zero) throw new RangeError('Expected positive rational denominator');
  const g = gcd(n, d); return { n: n / g, d: d / g };
}
function binary(v: BinaryValue): Fraction {
  return v.exponent >= 0 ? fraction(v.coefficient << BigInt(v.exponent)) : fraction(v.coefficient, one << BigInt(-v.exponent));
}
const scalar = (x: number) => binary(binaryProduct(x));
const add = (a: Fraction, b: Fraction) => fraction(a.n * b.d + b.n * a.d, a.d * b.d);
const neg = (a: Fraction) => ({ n: -a.n, d: a.d });
const sub = (a: Fraction, b: Fraction) => add(a, neg(b));
const mul = (a: Fraction, b: Fraction) => fraction(a.n * b.n, a.d * b.d);
function div(a: Fraction, b: Fraction): Fraction {
  if (!b.n) throw new RangeError('Zero denominator');
  return fraction(a.n * b.d * (b.n < zero ? -one : one), a.d * abs(b.n));
}
const compare = (a: Fraction, b: Fraction) => a.n * b.d - b.n * a.d;
const point = (a: Fraction): Interval => ({ lower: a, upper: a });
const sum = (a: Interval, b: Interval): Interval => ({ lower: add(a.lower, b.lower), upper: add(a.upper, b.upper) });
const negative = (a: Interval): Interval => ({ lower: neg(a.upper), upper: neg(a.lower) });
const difference = (a: Interval, b: Interval) => sum(a, negative(b));
function multiply(a: Interval, b: Interval): Interval {
  const xs = [mul(a.lower, b.lower), mul(a.lower, b.upper), mul(a.upper, b.lower), mul(a.upper, b.upper)];
  return { lower: xs.reduce((a, b) => compare(a, b) < zero ? a : b), upper: xs.reduce((a, b) => compare(a, b) > zero ? a : b) };
}
function dividePositive(a: Interval, b: Interval) {
  if (b.lower.n <= zero) throw new RangeError('Expected positive interval denominator');
  return multiply(a, { lower: div(fraction(one), b.upper), upper: div(fraction(one), b.lower) });
}
const reduce = (xs: Interval[]) => xs.reduce(sum, point(fraction(zero)));
const dyadicInteger = (n: bigint): BinaryValue => ({ coefficient: n, exponent: 0 });
const observe = (a: Interval): PartitionObservation => partitionObservationFromRange({
  lower: { n: dyadicInteger(a.lower.n), d: dyadicInteger(a.lower.d) },
  upper: { n: dyadicInteger(a.upper.n), d: dyadicInteger(a.upper.d) },
});
const infinity = (sign: number): PartitionObservation => ({ value: null, sign, logMagnitude: null, logResolution: 'not-applicable', resolution: 'mathematical-infinity', bounds: null });
const precisions = [128, 256, 512, 1024, 2048, 4096, 8192];
export interface WeightedBoltzmannOptions { minimumPrecision?: number; maximumPrecision?: number }
export function weightedBoltzmannModel(input: WeightedBoltzmannParameters, options: WeightedBoltzmannOptions = {}) {
  if (!input || !Array.isArray(input.energyLevels) || !Array.isArray(input.referenceWeights)
    || input.energyLevels.length < 1 || input.energyLevels.length > 32 || input.energyLevels.length !== input.referenceWeights.length
    || !Array.from(input.energyLevels).every(Number.isFinite)
    || !Array.from(input.referenceWeights).every(v => Number.isFinite(v) && v >= 0)
    || !input.referenceWeights.some(v => v > 0) || !Number.isFinite(input.energyOffset) || !Number.isFinite(input.temperature))
    throw new RangeError('Expected 1..32 finite levels and nonnegative finite reference weights with nonempty support');
  const minimum = options.minimumPrecision ?? 128, maximum = options.maximumPrecision ?? 8192;
  if (!precisions.includes(minimum) || !precisions.includes(maximum) || minimum > maximum) throw new RangeError('Invalid precision range');
  const energies = input.energyLevels.map(e => binary(binarySum([binaryProduct(e), binaryProduct(input.energyOffset)])));
  const weights = input.referenceWeights.map(scalar), allowed = weights.map((m, i) => m.n ? i : -1).filter(i => i >= 0);
  const T = input.temperature, direction = T < 0 ? -1 : 1;
  const anchor = allowed.map(i => energies[i]).reduce((a, b) =>
    (direction > 0 ? compare(a, b) <= zero : compare(a, b) >= zero) ? a : b);
  const costs = energies.map(e => direction > 0 ? sub(e, anchor) : sub(anchor, e));
  const thermal = binary(binaryProduct(kB, Math.abs(T)));
  const total = weights.reduce(add, fraction(zero));
  const ground = allowed.filter(i => costs[i].n === zero), groundTotal = ground.map(i => weights[i]).reduce(add, fraction(zero));
  const equalEnergy = allowed.every(i => costs[i].n === zero);
  const attempts: number[] = [];
  function calculate(P: number) {
    const ctx = createTranscendentalContext(P), scale = one << BigInt(P);
    const fixed = (v: { lower: bigint; upper: bigint }): Interval => ({ lower: fraction(v.lower, scale), upper: fraction(v.upper, scale) });
    const log = (v: Fraction) => fixed(ctx.logRatio(dyadicInteger(v.n), dyadicInteger(v.d)));
    const logInterval = (v: Interval): Interval => ({ lower: log(v.lower).lower, upper: log(v.upper).upper });
    const raws = weights.map((m, i) => !m.n || T === 0 && costs[i].n !== zero ? point(fraction(zero))
      : multiply(point(m), T === 0 ? point(fraction(one)) : fixed(ctx.expNegativeRatio(dyadicInteger(costs[i].n * thermal.d), dyadicInteger(costs[i].d * thermal.n)))));
    const W = reduce(raws), probabilities = raws.map(q => dividePositive(q, W));
    const logW = logInterval(W);
    const logWeights = weights.map(m => m.n ? log(m) : point(fraction(zero)));
    const logProbabilities = raws.map((raw, i): Interval | null => {
      if (!weights[i].n || T === 0 && costs[i].n !== zero) return null;
      if ((T === 0 ? ground.length : allowed.length) === 1) return point(fraction(zero));
      if (raw.lower.n > zero) {
        const quotient = dividePositive(W, raw);
        // W/raw>=1 analytically; clamp only to this proven bound.
        if (compare(quotient.lower, fraction(one)) < zero) quotient.lower = fraction(one);
        return negative(logInterval(quotient));
      }
      return difference(difference(logWeights[i], T === 0 ? point(fraction(zero)) : point(div(costs[i], thermal))), logW);
    });
    const mean = reduce(probabilities.map((p, i) => multiply(p, point(energies[i]))));
    // Pair identity avoids catastrophic E[E²]-E[E]² cancellation and preserves tiny gaps under a common offset.
    const pairTerms: Interval[] = [];
    for (const i of allowed) for (const j of allowed) if (i < j) {
      const gap = sub(energies[i], energies[j]);
      pairTerms.push(multiply(multiply(raws[i], raws[j]), point(mul(gap, gap))));
    }
    const variance = dividePositive(reduce(pairTerms), multiply(W, W));
    const labelEntropy = negative(reduce(probabilities.map((p, i) => logProbabilities[i] ? multiply(p, logProbabilities[i]!) : point(fraction(zero)))));
    const correction = reduce(probabilities.map((p, i) => multiply(p, logWeights[i])));
    const relativeEntropy = T === 0 ? log(groundTotal) : equalEnergy ? log(total) : sum(labelEntropy, correction);
    const heatCapacity = T === 0 ? point(fraction(zero)) : dividePositive(variance, point(mul(thermal, thermal)));
    const signedThermal = T < 0 ? neg(thermal) : thermal;
    const freeEnergy = T === 0 ? point(anchor) : difference(point(anchor), multiply(point(signedThermal), logW));
    const logPartition = T === 0 ? anchor.n === zero ? observe(logW) : infinity(anchor.n < zero ? 1 : -1)
      : observe(difference(logW, point(div(anchor, signedThermal))));
    const states = energies.map((energy, i) => {
      const probability = observe(probabilities[i]);
      if (weights[i].n && (T !== 0 || costs[i].n === zero) && probability.sign === null) probability.sign = 1;
      const logProbability = logProbabilities[i] ? observe(logProbabilities[i]!) : infinity(-1);
      if (logProbabilities[i] && (T === 0 ? ground.length : allowed.length) > 1 && logProbability.sign === null) logProbability.sign = -1;
      // log(p) is computed before the projection p≈1 loses its rare complement.
      if (probability.sign === 1) {
        probability.logMagnitude = logProbability.value;
        probability.logResolution = logProbability.resolution === 'unresolved-at-precision-ceiling' ? 'unresolved'
          : logProbability.value === null ? 'outside-number-range' : 'finite';
      }
      return { index: i, allowed: weights[i].n !== zero, energy: observe(point(energy)), referenceWeight: observe(point(weights[i])),
        probability, logProbability };
    });
    const observations = { referenceMeasureTotal: observe(point(total)), meanEnergy: observe(mean), energyVariance: observe(variance),
      labelEntropyNats: observe(labelEntropy), meanLogReferenceWeight: observe(correction), relativeEntropyInKB: observe(relativeEntropy),
      heatCapacityInKB: observe(heatCapacity), freeEnergy: observe(freeEnergy), logPartition };
    // Strict positivity follows from finite-temperature support, even when
    // an enclosure at the resource ceiling still touches zero.
    if (T !== 0 && !equalEnergy) {
      observations.energyVariance.sign = 1; observations.heatCapacityInKB.sign = 1;
    }
    if ((T === 0 ? ground.length : allowed.length) > 1) observations.labelEntropyNats.sign = 1;
    const all = [...Object.values(observations), ...states.flatMap(s => [s.probability, s.logProbability])];
    return { ...observations, states, precisionBits: P, fullyResolved: all.every(q => q.resolution !== 'unresolved-at-precision-ceiling') };
  }
  let latest: ReturnType<typeof calculate> | undefined;
  for (const P of precisions.filter(p => p >= minimum && p <= maximum)) {
    attempts.push(P); latest = calculate(P); if (latest.fullyResolved) break;
  }
  return { scope: 'finite-reference-measure-boltzmann' as const, boltzmannConstant: kB, ...latest!, attempts,
    mode: T === 0 ? 'positive-side-zero-temperature-limit' : T < 0 ? 'bounded-negative-temperature' : 'positive-temperature',
    allowedStateCount: allowed.length, anchorStateCount: ground.length,
    approximationContract: 'Exact supplied binary64 energies, offset and nonnegative reference weights; stored binary64 kB. Directed rational enclosures resolve only at relative width <=2^-48. Nonzero outside display range is null with sign; unresolved is not zero. Real reference weights are not automatically physical multiplicities. All derivatives hold reference weights and energies fixed.' };
}
