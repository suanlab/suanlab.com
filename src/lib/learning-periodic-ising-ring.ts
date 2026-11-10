/** Zero-field nearest-neighbor Ising ring, N distinct bonds and N>=3.
 * Directed fixed-point geometric sums handle odd antiferromagnetic rings.
 * Correlation powers retain a separate binary exponent: tiny nonzero spatial
 * tails are never truncated to the working fixed-point quantum.
 */
import { binaryProduct, binaryMultiply, binarySum, binaryNegate, type BinaryValue } from './learning-binary-products';
import { createTranscendentalContext, type FixedInterval } from './learning-interval-transcendentals';
import { partitionObservationFromRange, scalePartitionObservation, type PartitionObservation } from './learning-partition-exact-core';

export interface PeriodicIsingRingParameters { spinCount: number; couplingK: number }
export interface PeriodicIsingRingOptions { minimumPrecision?: number; maximumPrecision?: number }
interface Ratio { n: BinaryValue; d: BinaryValue }
interface Range { lower: Ratio; upper: Ratio }
const Z = BigInt(0), O = BigInt(1), TWO = BigInt(2);
const one = binaryProduct(1);
const dyadic = (coefficient: bigint, exponent = 0): BinaryValue => ({ coefficient, exponent });
const zeroObservation = () => partitionObservationFromRange({ lower: { n: binaryProduct(0), d: one }, upper: { n: binaryProduct(0), d: one } });
const unavailable = (sign: number | null): PartitionObservation => ({ value: null, sign, logMagnitude: null, logResolution: 'unresolved', resolution: 'unresolved-at-precision-ceiling', bounds: null });
const negativeInfinity = (): PartitionObservation => ({ value: null, sign: -1, logMagnitude: null, logResolution: 'not-applicable', resolution: 'mathematical-infinity', bounds: null });
const precisions = [128, 256, 512, 1024, 2048, 4096, 8192];

export function periodicIsingRingModel(input: PeriodicIsingRingParameters, options: PeriodicIsingRingOptions = {}) {
  if (!input || !Number.isInteger(input.spinCount) || input.spinCount < 3 || input.spinCount > 128
    || !Number.isFinite(input.couplingK) || Math.abs(input.couplingK) > 8)
    throw new RangeError('Expected a periodic ring with N=3..128 and finite dimensionless K in [-8,8]');
  const minimum = options.minimumPrecision ?? 128, maximum = options.maximumPrecision ?? 8192;
  if (!precisions.includes(minimum) || !precisions.includes(maximum) || minimum > maximum) throw new RangeError('Invalid precision range');
  const parameters = Object.freeze({ spinCount: input.spinCount, couplingK: input.couplingK }), N = input.spinCount, K = input.couplingK;
  const absoluteK = binaryProduct(Math.abs(K)), twiceK = binaryMultiply(absoluteK, binaryProduct(2));
  const oddAntiferromagnetic = K < 0 && N % 2 === 1, attempts: number[] = [];

  function calculate(P: number) {
    const ctx = createTranscendentalContext(P), S = O << BigInt(P), fixedOne = { lower: S, upper: S };
    const add = (a: FixedInterval, b: FixedInterval): FixedInterval => ({ lower: a.lower + b.lower, upper: a.upper + b.upper });
    const subtract = (a: FixedInterval, b: FixedInterval): FixedInterval => ({ lower: a.lower - b.upper, upper: a.upper - b.lower });
    const multiply = (a: FixedInterval, b: FixedInterval): FixedInterval => ({ lower: a.lower * b.lower / S, upper: (a.upper * b.upper + S - O) / S });
    const divide = (a: FixedInterval, b: FixedInterval): FixedInterval => ({ lower: a.lower * S / b.upper, upper: (a.upper * S + b.lower - O) / b.lower });
    const fixedRange = (v: FixedInterval): Range => ({ lower: { n: dyadic(v.lower, -P), d: one }, upper: { n: dyadic(v.upper, -P), d: one } });
    const observe = (v: FixedInterval) => partitionObservationFromRange(fixedRange(v));
    const log = (v: FixedInterval): FixedInterval => ({
      lower: ctx.logRatio(dyadic(v.lower, -P), one).lower,
      upper: ctx.logRatio(dyadic(v.upper, -P), one).upper,
    });
    const shifted = (v: FixedInterval, b: BinaryValue): Range => ({
      lower: { n: binarySum([dyadic(v.lower, -P), b]), d: one }, upper: { n: binarySum([dyadic(v.upper, -P), b]), d: one },
    });
    const u = ctx.expNegativeRatio(twiceK, one); // exp(-2|K|)>0 on this bounded domain
    const v = K === 0 ? { lower: Z, upper: Z } : divide({ lower: S - u.upper, upper: S - u.lower }, add(fixedOne, u));
    // 0<=|tanhK|<1 is known analytically, including a capped tiny K.
    v.lower = v.lower < Z ? Z : v.lower;
    const powers: FixedInterval[] = [fixedOne], geometric: FixedInterval[] = [{ lower: Z, upper: Z }];
    for (let n = 1; n <= N; n++) {
      geometric.push(add(geometric[n - 1], powers[n - 1]));
      powers.push(multiply(powers[n - 1], v));
    }
    const plusLoop = add(fixedOne, powers[N]);
    // These positive forms avoid alternating susceptibility sums and the
    // cancellation 1-v^N on an odd antiferromagnetic ring.
    const oneMinusV = divide({ lower: TWO * u.lower, upper: TWO * u.upper }, add(fixedOne, u));
    const susceptibility = K >= 0
      ? divide(multiply(add(fixedOne, v), geometric[N]), plusLoop)
      : oddAntiferromagnetic
        ? divide(plusLoop, multiply(add(fixedOne, v), geometric[N]))
        : divide(multiply(multiply(oneMinusV, oneMinusV), geometric[N]), multiply(add(fixedOne, v), plusLoop));
    const logV = K !== 0 && v.lower > Z ? log(v) : null;
    const maxDistance = Math.floor(N / 2), computed: { correlation: PartitionObservation; logAbsoluteCorrelation: PartitionObservation }[] = [];
    for (let d = 0; d <= maxDistance; d++) {
      if (d === 0) { computed.push({ correlation: observe(fixedOne), logAbsoluteCorrelation: zeroObservation() }); continue; }
      if (K === 0) { computed.push({ correlation: zeroObservation(), logAbsoluteCorrelation: negativeInfinity() }); continue; }
      const numerator = oddAntiferromagnetic ? geometric[N - 2 * d] : add(fixedOne, powers[N - 2 * d]);
      const denominator = oddAntiferromagnetic ? geometric[N] : plusLoop;
      const range: Range = {
        lower: { n: dyadic(v.lower ** BigInt(d) * numerator.lower, -P * d), d: dyadic(denominator.upper) },
        upper: { n: dyadic(v.upper ** BigInt(d) * numerator.upper, -P * d), d: dyadic(denominator.lower) },
      };
      const sign = K < 0 && d % 2 ? -1 : 1;
      const signedRange: Range = sign > 0 ? range : {
        lower: { n: binaryNegate(range.upper.n), d: range.upper.d }, upper: { n: binaryNegate(range.lower.n), d: range.lower.d },
      };
      const correlation = partitionObservationFromRange(signedRange); correlation.sign = sign;
      const logAbsoluteCorrelation = logV ? observe(subtract(add({ lower: BigInt(d) * logV.lower, upper: BigInt(d) * logV.upper }, log(numerator)), log(denominator))) : unavailable(-1);
      computed.push({ correlation, logAbsoluteCorrelation });
    }
    const logTwoCosh = shifted(log(add(fixedOne, u)), absoluteK);
    const loopLog = oddAntiferromagnetic ? add(log(oneMinusV), log(geometric[N])) : log(plusLoop);
    const n = binaryProduct(N), logPartition = partitionObservationFromRange({
      lower: { n: binarySum([binaryMultiply(logTwoCosh.lower.n, n), dyadic(loopLog.lower, -P)]), d: one },
      upper: { n: binarySum([binaryMultiply(logTwoCosh.upper.n, n), dyadic(loopLog.upper, -P)]), d: one },
    });
    const chi = observe(susceptibility), varianceMean = scalePartitionObservation(chi, one, n), varianceTotal = scalePartitionObservation(chi, n);
    const xi = K === 0 ? zeroObservation() : logV && logV.upper < Z ? partitionObservationFromRange({
      lower: { n: dyadic(S), d: dyadic(-logV.lower) }, upper: { n: dyadic(S), d: dyadic(-logV.upper) },
    }) : unavailable(1);
    const infiniteResponse = observe(K <= 0 ? u : divide(fixedOne, u));
    const oneStep = observe(K < 0 ? { lower: -v.upper, upper: -v.lower } : v);
    if (K !== 0) oneStep.sign = Math.sign(K);
    const logOneStep = K === 0 ? negativeInfinity() : logV ? observe(logV) : unavailable(-1);
    const observations = [chi, varianceMean, varianceTotal, logPartition, xi, infiniteResponse, oneStep, logOneStep, ...computed.flatMap(c => [c.correlation, c.logAbsoluteCorrelation])];
    return { meanSpin: zeroObservation(), correlations: Array.from({ length: N }, (_, offset) => ({ offset, distance: Math.min(offset, N - offset), ...computed[Math.min(offset, N - offset)] })),
      commonFieldResponsePerSpin: chi, magnetizationMeanVariance: varianceMean, totalSpinVariance: varianceTotal,
      logPartition, infiniteChainCorrelationLength: xi, infiniteChainCommonFieldResponse: infiniteResponse,
      infiniteChainOneStepCorrelation: oneStep, logAbsoluteOneStepCorrelation: logOneStep,
      correlationLengthStatus: K === 0 ? 'independent-zero-range-limit' as const : 'finite-positive-infinite-chain-envelope' as const,
      precisionBits: P, fullyResolved: observations.every(v => v.resolution !== 'unresolved-at-precision-ceiling') };
  }
  let latest: ReturnType<typeof calculate> | undefined;
  for (const P of precisions.filter(v => v >= minimum && v <= maximum)) { attempts.push(P); latest = calculate(P); if (latest.fullyResolved) break; }
  return { scope: 'zero-field-periodic-nearest-neighbor-ising-ring' as const, parameters, bondCount: N, microstateCount: (O << BigInt(N)).toString(), oddAntiferromagnetic, ...latest!, attempts,
    approximationContract: 'N=3..128 named classical spins, N distinct periodic nearest-neighbor bonds, zero local fields and exact supplied binary64 K in [-8,8]. Directed exponential/logarithm intervals and exact powers retain connected correlations C(r)=(t^r+t^(N-r))/(1+t^N), t=tanhK. Independent zero means follow spin reversal. Odd antiferromagnetic correlations use geometric sums; susceptibility uses positive expressions. Working precision 128..8192 bits; interval-resolved relative width <=2^-48. A capped value is unavailable, never a structural zero. Signed correlations and directed log absolute correlations are separate observations; magnitude-log projections are approximate. The length -1/log|t| is an infinite-chain envelope in lattice spacings, with its independent zero-range limit explicitly assigned at K=0. It is not a finite-ring distance cutoff, physical length without a lattice spacing, time correlation or finite-temperature phase transition. Physical field response needs a unit conversion.' };
}
