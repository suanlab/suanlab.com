/** Specified equally spaced spectrum, not a derivation of quantization.
 * Exact binary input ratios, directed intervals, and explicit cold asymptotics.
 * The finite preparation remains meaningful when the infinite sum diverges. */
import { binaryProduct as b, binarySum as sum, binaryMultiply as multiply, binaryNegate as neg,
  binaryRatioObservation as ratio, type BinaryValue } from './learning-binary-products';
import { createTranscendentalContext, type FixedInterval } from './learning-interval-transcendentals';
import { partitionObservationFromRange as enclose, scalePartitionObservation as scaleObservation,
  partitionObservationRange as rangeOf, type PartitionObservation } from './learning-partition-exact-core';
import { binomialExponential } from './learning-binomial-logit';
import { canonicalBoltzmannConstant as kB } from './learning-canonical-spectrum';
export interface EnergyLadderParameters { gapEnergy: number; temperature: number; cutoff: number }
type Range = NonNullable<ReturnType<typeof rangeOf>>;
type Rational = Range['lower'];
const Z = BigInt(0), O = BigInt(1), one = b(1), zero = b(0);
const r = (n: BinaryValue, d = one): Rational => ({ n, d });
const dyadic = (coefficient: bigint, exponent: number): BinaryValue => ({ coefficient, exponent });
const add = (a: Rational, c: Rational): Rational => r(sum([multiply(a.n, c.d), multiply(c.n, a.d)]), multiply(a.d, c.d));
const minus = (a: Rational): Rational => r(neg(a.n), a.d);
const times = (a: Rational, c: Rational): Rational => r(multiply(a.n, c.n), multiply(a.d, c.d));
const divide = (a: Rational, c: Rational): Rational => {
  if (c.n.coefficient <= Z) throw new RangeError('Expected positive interval denominator');
  return r(multiply(a.n, c.d), multiply(a.d, c.n));
};
const point = (q: Rational): Range => ({ lower: q, upper: q });
const exact = (v: BinaryValue, d = one) => enclose(point(r(v, d)));
const constant = (v: number) => exact(b(v));
const addRange = (a: Range, c: Range): Range => ({ lower: add(a.lower, c.lower), upper: add(a.upper, c.upper) });
const negativeRange = (a: Range): Range => ({ lower: minus(a.upper), upper: minus(a.lower) });
const subtractRange = (a: Range, c: Range) => addRange(a, negativeRange(c));
const scaleRange = (a: Range, n: BinaryValue, d = one): Range => {
  const f = r(n, d);
  return n.coefficient < Z ? { lower: times(a.upper, f), upper: times(a.lower, f) } : { lower: times(a.lower, f), upper: times(a.upper, f) };
};
const positiveMultiply = (a: Range, c: Range): Range => ({ lower: times(a.lower, c.lower), upper: times(a.upper, c.upper) });
const positiveDivide = (a: Range, c: Range): Range => ({ lower: divide(a.lower, c.upper), upper: divide(a.upper, c.lower) });
const fixedRange = (a: FixedInterval, P: number): Range => ({ lower: r(dyadic(a.lower, -P)), upper: r(dyadic(a.upper, -P)) });
const infinity = (sign: number): PartitionObservation => ({ value: null, sign, logMagnitude: null, logResolution: 'not-applicable', resolution: 'mathematical-infinity', bounds: null });
const unresolved = (bounds: Range | null = null, sign: number | null = null): PartitionObservation => ({ ...enclose(bounds ?? { lower: r(b(-1)), upper: r(one) }), value: null, sign, resolution: 'unresolved-at-precision-ceiling', logMagnitude: null, logResolution: 'unresolved' });
function withSign(q: PartitionObservation, sign: number): PartitionObservation {
  return q.resolution === 'unresolved-at-precision-ceiling' ? { ...q, sign } : q;
}
function exponential(numerator: BinaryValue, denominator: BinaryValue, enclosure?: Range): PartitionObservation {
  const q = binomialExponential(numerator, denominator);
  return { value: q.value, sign: 1, logMagnitude: q.logMagnitude,
    logResolution: q.logMagnitudeStatus === 'finite' ? 'finite' : 'outside-number-range',
    resolution: 'derived-exponential', bounds: enclosure ? enclose(enclosure).bounds : null };
}
function coldMoment(cost: BinaryValue, thermal: BinaryValue, factorLog = 0, sign = 1, enclosure?: Range): PartitionObservation {
  const q = exponential(sum([neg(cost), multiply(thermal, b(factorLog))]), thermal, enclosure);
  return { ...q, value: q.value === null ? null : sign * q.value, sign, resolution: 'log-tail-approximation',
    bounds: sign < 0 && enclosure ? enclose(negativeRange(enclosure)).bounds : q.bounds };
}
/** Positive polynomial jets. Concatenation evaluates geometric sums and the
 * first two raw index moments in O(log cutoff), without an O(cutoff²) pair loop.
 * All fixed-point products round outwards after each operation. */
function geometricJets(q: FixedInterval, count: number, P: number) {
  const S = O << BigInt(P), floor = (n: bigint) => n / S, ceil = (n: bigint) => (n + S - O) / S;
  const pointF = (n: bigint): FixedInterval => ({ lower: n, upper: n });
  const plus = (a: FixedInterval, c: FixedInterval): FixedInterval => ({ lower: a.lower + c.lower, upper: a.upper + c.upper });
  const mul = (a: FixedInterval, c: FixedInterval): FixedInterval => ({ lower: floor(a.lower * c.lower), upper: ceil(a.upper * c.upper) });
  const scale = (a: FixedInterval, n: number): FixedInterval => ({ lower: a.lower * BigInt(n), upper: a.upper * BigInt(n) });
  type Jet = { count: number; power: FixedInterval; w: FixedInterval; first: FixedInterval; second: FixedInterval };
  const empty: Jet = { count: 0, power: pointF(S), w: pointF(Z), first: pointF(Z), second: pointF(Z) };
  let result = empty, block: Jet = { count: 1, power: q, w: pointF(S), first: pointF(Z), second: pointF(Z) };
  const join = (a: Jet, c: Jet): Jet => ({ count: a.count + c.count, power: mul(a.power, c.power),
    w: plus(a.w, mul(a.power, c.w)),
    first: plus(a.first, mul(a.power, plus(c.first, scale(c.w, a.count)))),
    second: plus(a.second, mul(a.power, plus(c.second, plus(scale(c.first, 2 * a.count), scale(c.w, a.count ** 2))))) });
  for (let remaining = count; remaining > 0; remaining = Math.floor(remaining / 2)) {
    if (remaining % 2) result = join(result, block);
    if (remaining > 1) block = join(block, block);
  }
  return result;
}
function sampleIndices(L: number) {
  if (L <= 62) return Array.from({ length: L + 2 }, (_, n) => n);
  return [...new Set([...Array.from({ length: 16 }, (_, n) => n), ...Array.from({ length: 16 }, (_, n) => L - n),
    ...Array.from({ length: 31 }, (_, n) => Math.round(n * L / 30)), L + 1])].sort((a, c) => a - c);
}
export function energyLadderModel(parameters: EnergyLadderParameters) {
  const { gapEnergy: gap, temperature: T, cutoff: L } = parameters;
  if (!Number.isFinite(gap) || gap < 0 || !Number.isFinite(T) || !Number.isInteger(L) || L < 0 || L > 4096) throw new RangeError('Expected a finite nonnegative gap, finite temperature and integer cutoff 0..4096');
  const m = L + 1, epsilon = b(gap), halfGap = r(epsilon, b(2)), indices = sampleIndices(L);
  const normalizable = gap > 0 && T > 0;
  const common = { scope: 'specified-equally-spaced-spectrum-with-finite-conditional-cutoff' as const, parameters: { ...parameters },
    finiteStateCount: m, boltzmannConstant: kB, sampledAllFiniteStates: L <= 62,
    contract: 'E_n=epsilon(n+1/2) is specified, not derived. The finite preparation is normalized on 0..L. Infinite equilibrium requires epsilon>0,T>0; T=0 denotes its positive-side ground limit for epsilon>0. Exact binary input ratios use directed intervals; a finite nonzero result outside binary64 range is not a mathematical infinity. Cold leading moments have separate approximate logarithms and bounded physical omissions. Sample rows are not renormalized.' as const };
  if (T === 0 || gap === 0) {
    const uniform = gap === 0, groundLimit = gap > 0;
    const ctx = createTranscendentalContext(128), entropy = uniform && m > 1 ? enclose(fixedRange(ctx.logRatio(b(m), one), 128)) : constant(0);
    const energy = uniform ? constant(0) : exact(epsilon, b(2));
    const f = T === 0 ? energy : scaleObservation(entropy, b(-kB, T));
    const finite = { meanIndex: uniform ? exact(b(L), b(2)) : constant(0), indexVariance: uniform ? exact(b(L, L + 2), b(12)) : constant(0),
      meanEnergy: energy, energyVariance: constant(0), heatCapacityInKB: constant(0), heatCapacity: constant(0), entropyInKB: entropy,
      entropy: scaleObservation(entropy, b(kB)), freeEnergy: f, logPartition: T === 0 && groundLimit ? infinity(-1) : entropy,
      partitionFunction: T === 0 && groundLimit ? constant(0) : constant(m) };
    const infinite = groundLimit ? { status: 'positive-side-zero-temperature-limit' as const, reason: null,
      quantities: { ...finite, meanIndex: constant(0), indexVariance: constant(0), entropyInKB: constant(0), entropy: constant(0) } }
      : { status: 'not-normalizable' as const, reason: 'infinitely-many-equal-energy-states' as const, quantities: null };
    const comparison = groundLimit ? { missingMass: constant(0), retainedMass: constant(1), logRetainedMass: constant(0),
      totalVariation: constant(0), forwardKL: constant(0), reverseKL: constant(0), meanIndexBias: constant(0), indexVarianceBias: constant(0), meanEnergyBias: constant(0) } : null;
    return { ...common, mode: uniform ? 'finite-degenerate-spectrum' as const : 'positive-side-zero-temperature-limit' as const,
      precisionBits: 128, attempts: [128], fullyResolved: true, coldTailApproximation: false, x: T === 0 ? groundLimit ? infinity(1) : null : constant(0), q: groundLimit ? constant(0) : constant(1), anchoredDecayRatio: groundLimit ? constant(0) : constant(1),
      finite, infinite, comparison, samples: indices.map(n => ({ n, energy: exact(b(gap, 2 * n + 1), b(2)),
        finiteProbability: n > L || groundLimit && n !== 0 ? constant(0) : groundLimit ? constant(1) : exact(one, b(m)),
        infiniteProbability: groundLimit ? constant(n === 0 ? 1 : 0) : null })) };
  }
  const thermal = b(kB, Math.abs(T)), direction = T > 0 ? 1 : -1;
  const x = r(epsilon, thermal), logX = ratio(epsilon, thermal).logMagnitude!;
  const cold = sum([epsilon, neg(multiply(thermal, b(4096)))]).coefficient > Z;
  const cutoffCost = b(gap, m), coldCutoff = sum([cutoffCost, neg(multiply(thermal, b(4096)))]).coefficient > Z;
  const attempts: number[] = [];
  let latest: ReturnType<typeof calculate> | undefined;
  function calculate(P: number) {
    const ctx = createTranscendentalContext(P), qFixed = ctx.expNegativeRatio(epsilon, thermal), qRange = fixedRange(qFixed, P);
    const d = subtractRange(point(r(one)), qRange);
    const jets = geometricJets(qFixed, m, P), W = fixedRange(jets.w, P);
    const logRange = (a: Range): Range => {
      const lo = ctx.logRatio(a.lower.n, a.lower.d), hi = ctx.logRatio(a.upper.n, a.upper.d);
      return { lower: r(dyadic(lo.lower, -P)), upper: r(dyadic(hi.upper, -P)) };
    };
    const logW = m === 1 ? point(r(zero)) : logRange(W);
    const meanK = positiveDivide(fixedRange(jets.first, P), W);
    const denLower = jets.w.lower ** BigInt(2), denUpper = jets.w.upper ** BigInt(2);
    const rawLo = jets.second.lower * jets.w.lower - jets.first.upper ** BigInt(2);
    const rawHi = jets.second.upper * jets.w.upper - jets.first.lower ** BigInt(2);
    const varianceK: Range = { lower: r(dyadic(rawLo < Z ? Z : rawLo, 0), dyadic(denUpper, 0)), upper: r(dyadic(rawHi, 0), dyadic(denLower, 0)) };
    const meanN = direction > 0 ? meanK : subtractRange(point(r(b(L))), meanK);
    const anchor = direction > 0 ? halfGap : r(b(gap, 2 * L + 1), b(2));
    const uFinite = addRange(point(anchor), scaleRange(meanK, b(direction * gap)));
    const logZFinite = subtractRange(logW, point(r(multiply(anchor.n, b(direction)), multiply(anchor.d, thermal))));
    const fFinite = subtractRange(point(anchor), scaleRange(logW, b(direction * kB, Math.abs(T))));
    const sFinite = addRange(logW, positiveMultiply(point(x), meanK));
    const vEnergy = scaleRange(varianceK, b(gap, gap));
    const cKB = scaleRange(varianceK, b(gap, gap), multiply(thermal, thermal));
    const energyLog = Math.log(gap), capacityLog = 2 * logX;
    const projectedX = ratio(epsilon, thermal).value;
    const entropyFactorLog = projectedX === null ? logX : Math.log1p(projectedX);
    const finite = { meanIndex: m === 1 ? constant(0) : cold && direction > 0 ? coldMoment(epsilon, thermal, 0, 1, meanN) : enclose(meanN),
      indexVariance: m === 1 ? constant(0) : cold ? coldMoment(epsilon, thermal, 0, 1, varianceK) : withSign(enclose(varianceK), 1),
      meanEnergy: enclose(uFinite), energyVariance: m === 1 ? constant(0) : cold ? coldMoment(epsilon, thermal, 2 * energyLog, 1, vEnergy) : withSign(enclose(vEnergy), 1),
      heatCapacityInKB: m === 1 ? constant(0) : cold ? coldMoment(epsilon, thermal, capacityLog, 1, cKB) : withSign(enclose(cKB), 1),
      heatCapacity: m === 1 ? constant(0) : cold ? coldMoment(epsilon, thermal, capacityLog + Math.log(kB)) : withSign(enclose(scaleRange(cKB, b(kB))), 1),
      entropyInKB: m === 1 ? constant(0) : cold ? coldMoment(epsilon, thermal, entropyFactorLog, 1, sFinite) : withSign(enclose(sFinite), 1),
      entropy: m === 1 ? constant(0) : cold ? coldMoment(epsilon, thermal, entropyFactorLog + Math.log(kB)) : withSign(enclose(scaleRange(sFinite, b(kB))), 1),
      freeEnergy: enclose(fFinite), logPartition: enclose(logZFinite),
      partitionFunction: exponential(add(logZFinite.lower, logZFinite.upper).n, multiply(add(logZFinite.lower, logZFinite.upper).d, b(2))) };
    // Even when q projects to one, an unresolved 1-q must never imply divergence.
    const haveDenominator = d.lower.n.coefficient > Z;
    let infiniteQuantities: typeof finite | null = null;
    let comparison: { missingMass: PartitionObservation; retainedMass: PartitionObservation; logRetainedMass: PartitionObservation; totalVariation: PartitionObservation; forwardKL: PartitionObservation; reverseKL: PartitionObservation; meanIndexBias: PartitionObservation; indexVarianceBias: PartitionObservation; meanEnergyBias: PartitionObservation } | null = null;
    let infiniteSampleRanges: Range[] | null = null;
    let infiniteLogD: Range | null = null;
    if (normalizable && haveDenominator) {
      const mean = positiveDivide(qRange, d), variance = positiveDivide(qRange, positiveMultiply(d, d));
      const logD = logRange(d); infiniteLogD = logD;
      const lnZ = subtractRange(negativeRange(point(r(epsilon, multiply(thermal, b(2))))), logD);
      const u = scaleRange(addRange(point(r(one, b(2))), mean), epsilon);
      const v = scaleRange(variance, b(gap, gap)), c = scaleRange(variance, b(gap, gap), multiply(thermal, thermal));
      const entropy = subtractRange(positiveMultiply(point(x), mean), logD);
      infiniteQuantities = { meanIndex: cold ? coldMoment(epsilon, thermal, 0, 1, mean) : withSign(enclose(mean), 1),
        indexVariance: cold ? coldMoment(epsilon, thermal, 0, 1, variance) : withSign(enclose(variance), 1),
        meanEnergy: enclose(u), energyVariance: cold ? coldMoment(epsilon, thermal, 2 * energyLog, 1, v) : withSign(enclose(v), 1),
        heatCapacityInKB: cold ? coldMoment(epsilon, thermal, capacityLog, 1, c) : withSign(enclose(c), 1),
        heatCapacity: cold ? coldMoment(epsilon, thermal, capacityLog + Math.log(kB)) : withSign(enclose(scaleRange(c, b(kB))), 1),
        entropyInKB: cold ? coldMoment(epsilon, thermal, entropyFactorLog, 1, entropy) : withSign(enclose(entropy), 1),
        entropy: cold ? coldMoment(epsilon, thermal, entropyFactorLog + Math.log(kB)) : withSign(enclose(scaleRange(entropy, b(kB))), 1),
        freeEnergy: enclose(addRange(point(halfGap), scaleRange(logD, thermal))), logPartition: enclose(lnZ),
        partitionFunction: exponential(add(lnZ.lower, lnZ.upper).n, multiply(add(lnZ.lower, lnZ.upper).d, b(2))) };
      const deltaRange = fixedRange(ctx.expNegativeRatio(cutoffCost, thermal), P), retained = subtractRange(point(r(one)), deltaRange);
      if (retained.lower.n.coefficient > Z) {
        const logRetained = logRange(retained), bias = scaleRange(positiveDivide(deltaRange, retained), b(m));
        const varianceBias = scaleRange(positiveDivide(deltaRange, positiveMultiply(retained, retained)), b(m, m));
        const delta = exponential(neg(cutoffCost), thermal, deltaRange);
        const kl = coldCutoff ? coldMoment(cutoffCost, thermal, 0, 1, negativeRange(logRetained)) : withSign(enclose(negativeRange(logRetained)), 1);
        comparison = { missingMass: delta, retainedMass: enclose(retained), logRetainedMass: scaleObservation(kl, b(-1)), totalVariation: delta,
          forwardKL: kl, reverseKL: infinity(1),
          meanIndexBias: coldCutoff ? coldMoment(cutoffCost, thermal, Math.log(m), 1, bias) : withSign(enclose(bias), 1),
          indexVarianceBias: coldCutoff ? coldMoment(cutoffCost, thermal, 2 * Math.log(m), 1, varianceBias) : withSign(enclose(varianceBias), 1),
          meanEnergyBias: coldCutoff ? coldMoment(cutoffCost, thermal, Math.log(m) + energyLog, 1, scaleRange(bias, epsilon)) : withSign(enclose(scaleRange(bias, epsilon)), 1) };
      }
      infiniteSampleRanges = indices.map(n => positiveMultiply(d, fixedRange(ctx.expNegativeRatio(b(gap, n), thermal), P)));
    }
    const samples = indices.map((n, i) => {
      const finiteAllowed = n <= L, costIndex = direction > 0 ? n : L - n;
      const weight = finiteAllowed ? fixedRange(ctx.expNegativeRatio(b(gap, costIndex), thermal), P) : point(r(zero));
      const finiteRange = positiveDivide(weight, W), infiniteRange = infiniteSampleRanges?.[i];
      const exponent = (log: Range, enclosure: Range) => {
        const twice = add(log.lower, log.upper);
        return exponential(twice.n, multiply(twice.d, b(2)), enclosure);
      };
      const finiteP = !finiteAllowed ? constant(0) : m === 1 ? constant(1)
        : exponent(subtractRange(point(r(b(-gap, costIndex), thermal)), logW), finiteRange);
      const infiniteP = !normalizable ? null : !infiniteRange || !infiniteLogD ? unresolved(null, 1)
        : exponent(addRange(infiniteLogD, point(r(b(-gap, n), thermal))), infiniteRange);
      return { n, energy: exact(b(gap, 2 * n + 1), b(2)), finiteProbability: finiteP, infiniteProbability: infiniteP };
    });
    const values = [...Object.values(finite), ...(infiniteQuantities ? Object.values(infiniteQuantities) : []), ...(comparison ? Object.values(comparison) : []), ...samples.flatMap(s => [s.finiteProbability, ...(s.infiniteProbability ? [s.infiniteProbability] : [])])];
    return { precisionBits: P, finite, infinite: normalizable ? { status: 'normalizable' as const, reason: null, quantities: infiniteQuantities }
      : { status: 'not-normalizable' as const, reason: 'weights-grow-on-an-unbounded-above-ladder' as const, quantities: null }, comparison, samples,
      anchoredDecayRatio: exponential(neg(epsilon), thermal, qRange),
      q: direction > 0 ? exponential(neg(epsilon), thermal, qRange) : exponential(epsilon, thermal), fullyResolved: (!normalizable || infiniteQuantities !== null && comparison !== null) && values.every(q => q.resolution !== 'unresolved-at-precision-ceiling') };
  }
  for (const P of [128, 256, 512, 1024, 2048, 4096, 8192]) { attempts.push(P); latest = calculate(P); if (latest.fullyResolved) break; }
  return { ...common, ...latest!, mode: T > 0 ? 'positive-temperature' as const : 'finite-negative-temperature' as const, attempts,
    x: exact(b(direction * gap), thermal), coldTailApproximation: cold,
    coldOmissionRelativeBound: cold || coldCutoff ? '2^-100 for leading moment and cutoff corrections only; floating logarithm errors are not certified' : null };
}
