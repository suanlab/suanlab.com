/** Fixed finite Hamiltonian, explicit labeled microstates, common SI offset.
 * Directed rational intervals certify reported moment signs and enclosures.
 * An analytic cold-tail branch preserves nonzero logarithmic observations;
 * its omission bound does not certify Math.log/exp rounding errors.
 */
import { binaryProduct as product, binarySum as sum, binaryMultiply as multiply, binaryNegate as negate,
  binaryRatioObservation as ratio, binaryObservation as observe, type BinaryValue, type BinaryObservation } from './learning-binary-products';
import { createTranscendentalContext } from './learning-interval-transcendentals';
import { binomialExponential } from './learning-binomial-logit';
import { canonicalBoltzmannConstant as kB } from './learning-canonical-spectrum';
const Z = BigInt(0), O = BigInt(1), one = product(1);
interface Rational { n: BinaryValue; d: BinaryValue }
interface Range { lower: Rational; upper: Rational }
export interface PartitionRational { numerator: { coefficient: string; exponent: number }; denominator: { coefficient: string; exponent: number } }
export interface PartitionObservation {
  value: number | null;
  sign: number | null;
  logMagnitude: number | null;
  logResolution: 'finite' | 'outside-number-range' | 'not-applicable' | 'unresolved';
  resolution: 'exact-rational' | 'structural-zero' | 'interval-resolved' | 'derived-exponential' | 'log-tail-approximation' | 'unresolved-at-precision-ceiling' | 'mathematical-infinity';
  bounds: { lower: PartitionRational; upper: PartitionRational } | null;
}
const sub = (a: BinaryValue, b: BinaryValue) => sum([a, negate(b)]);
const dyadic = (n: bigint, exponent: number): BinaryValue => ({ coefficient: n, exponent });
const r = (n: BinaryValue, d = one): Rational => ({ n, d });
const add = (a: Rational, b: Rational): Rational => r(sum([multiply(a.n, b.d), multiply(b.n, a.d)]), multiply(a.d, b.d));
const negative = (a: Rational): Rational => r(negate(a.n), a.d);
const scale = (a: Rational, n: BinaryValue, d = one): Rational => r(multiply(a.n, n), multiply(a.d, d));
const exactRange = (a: Rational): Range => ({ lower: a, upper: a });
const addRange = (a: Range, b: Range): Range => ({ lower: add(a.lower, b.lower), upper: add(a.upper, b.upper) });
const negateRange = (a: Range): Range => ({ lower: negative(a.upper), upper: negative(a.lower) });
const scaleRange = (a: Range, n: BinaryValue, d = one): Range => n.coefficient < Z
  ? { lower: scale(a.upper, n, d), upper: scale(a.lower, n, d) } : { lower: scale(a.lower, n, d), upper: scale(a.upper, n, d) };
function ratioRange(a: bigint, b: bigint, lowerDen: bigint, upperDen: bigint, exponent = 0): Range {
  const lower = r(dyadic(a, exponent), dyadic(a < Z ? lowerDen : upperDen, 0));
  const upper = r(dyadic(b, exponent), dyadic(b < Z ? upperDen : lowerDen, 0));
  return { lower, upper };
}
const midpoint = (a: Range): Rational => scale(add(a.lower, a.upper), one, product(2));
const serialize = (v: BinaryValue) => ({ coefficient: v.coefficient.toString(), exponent: v.exponent });
const serializeRational = (v: Rational): PartitionRational => ({ numerator: serialize(v.n), denominator: serialize(v.d) });
const bounds = (v: Range) => ({ lower: serializeRational(v.lower), upper: serializeRational(v.upper) });
function observation(q: BinaryObservation, resolution: PartitionObservation['resolution'], enclosure: Range | null = null): PartitionObservation {
  return { ...q, logResolution: q.sign === 0 ? 'not-applicable' : q.logMagnitude === null ? 'outside-number-range' : 'finite', resolution, bounds: enclosure ? bounds(enclosure) : null };
}
const zero = (): PartitionObservation => observation(observe(product(0)), 'structural-zero', exactRange(r(product(0))));
const exact = (v: Rational): PartitionObservation => v.n.coefficient === Z ? zero() : observation(ratio(v.n, v.d), 'exact-rational', exactRange(v));
const unresolved = (v: Range | null, sign: number | null = null): PartitionObservation => ({ value: null, sign: sign ?? (v && v.lower.n.coefficient > Z ? 1 : v && v.upper.n.coefficient < Z ? -1 : null), logMagnitude: null, logResolution: 'unresolved', resolution: 'unresolved-at-precision-ceiling', bounds: v ? bounds(v) : null });
const infinity = (sign: number): PartitionObservation => ({ value: null, sign, logMagnitude: null, logResolution: 'not-applicable', resolution: 'mathematical-infinity', bounds: null });
function resolved(v: Range): boolean {
  const lo = v.lower.n.coefficient, hi = v.upper.n.coefficient;
  if (lo === Z && hi === Z) return true;
  if (lo <= Z && hi >= Z) return false;
  const difference = sub(multiply(v.upper.n, v.lower.d), multiply(v.lower.n, v.upper.d));
  const near = lo > Z ? multiply(v.lower.n, v.upper.d) : negate(multiply(v.upper.n, v.lower.d));
  return sub(multiply(difference, product(2 ** 48)), near).coefficient <= Z;
}
function enclosed(v: Range, structuralZero = false): PartitionObservation {
  if (structuralZero) return zero();
  if (!resolved(v) || v.lower.n.coefficient === Z && v.upper.n.coefficient === Z) return unresolved(v);
  const m = midpoint(v); return observation(ratio(m.n, m.d), 'interval-resolved', v);
}
function negativeObservation(q: PartitionObservation): PartitionObservation {
  return { ...q, value: q.value === null ? null : q.value === 0 ? 0 : -q.value, sign: q.sign === null || q.sign === 0 ? q.sign : -q.sign,
    bounds: q.bounds ? { lower: { ...q.bounds.upper, numerator: { ...q.bounds.upper.numerator, coefficient: (-BigInt(q.bounds.upper.numerator.coefficient)).toString() } }, upper: { ...q.bounds.lower, numerator: { ...q.bounds.lower.numerator, coefficient: (-BigInt(q.bounds.lower.numerator.coefficient)).toString() } } } : null };
}
/** A logarithm is represented as numerator / thermal without first rounding
 * two enormous exponents and subtracting their floating projections. */
function logSum(terms: BinaryValue[], thermal: BinaryValue): BinaryValue {
  if (!terms.length) throw new RangeError('Expected positive terms');
  let largest = terms[0]; for (const term of terms) if (sub(term, largest).coefficient > Z) largest = term;
  let total = 0;
  for (const term of terms) { const delta = ratio(sub(term, largest), thermal); total += delta.value === null ? delta.logMagnitude! < 0 ? 1 : 0 : Math.exp(delta.value); }
  return sum([largest, multiply(thermal, product(Math.log(total)))]);
}
function tail(numerator: BinaryValue, thermal: BinaryValue, sign = 1, enclosure: Range | null = null): PartitionObservation {
  const q = binomialExponential(numerator, thermal);
  return { ...observation({ value: q.value === null ? null : sign * q.value, sign, logMagnitude: q.logMagnitude }, 'log-tail-approximation', enclosure), logResolution: q.logMagnitudeStatus === 'finite' ? 'finite' : 'outside-number-range' };
}
function exponential(logQ: PartitionObservation, enclosure: Range | null): PartitionObservation {
  if (logQ.resolution === 'structural-zero') return exact(r(one));
  if (logQ.resolution === 'unresolved-at-precision-ceiling') return unresolved(null, 1);
  if (enclosure) {
    const m = midpoint(enclosure), q = binomialExponential(m.n, m.d);
    return { ...observation(q, 'derived-exponential'), logResolution: q.logMagnitudeStatus === 'finite' ? 'finite' : 'outside-number-range' };
  }
  // Here log Q is a nonzero cold-tail correction below the linear range.
  // Q rounds to 1; this does not assert that log Q is exactly zero.
  return { value: 1, sign: 1, logMagnitude: logQ.value, logResolution: logQ.logResolution === 'unresolved' ? 'unresolved' : logQ.value === null ? 'outside-number-range' : 'finite', resolution: 'log-tail-approximation', bounds: null };
}
export interface ExactPartitionOptions {
  features?: readonly (readonly number[])[];
  minimumPrecision?: number;
  maximumPrecision?: number;
}
/** Internal finite-energy core. Inputs are exact binary energies on the
 * binary64 energy grid, with magnitude below2^1027 J. This includes exact
 * linear combinations used by the two-site Hamiltonian without overflow.
 */
export function exactFinitePartition(energies: readonly BinaryValue[], T: number, options: ExactPartitionOptions = {}) {
  if (!Array.isArray(energies) || energies.length < 1 || energies.length > 32 || !Number.isFinite(T)) throw new RangeError('Invalid exact finite spectrum');
  const actual = Array.from(energies, v => {
    if (!v || typeof v.coefficient !== 'bigint' || !Number.isInteger(v.exponent) || Math.abs(v.exponent) > 8192 || v.coefficient.toString(2).length > 8192) throw new RangeError('Invalid exact energy');
    let coefficient = v.coefficient, exponent = v.exponent;
    if (coefficient === Z) return product(0);
    while (coefficient % BigInt(2) === Z) { coefficient /= BigInt(2); exponent++; }
    const magnitude = coefficient < Z ? -coefficient : coefficient;
    if (exponent < -1074 || exponent + magnitude.toString(2).length > 1027) throw new RangeError('Exact energy outside core budget');
    return { coefficient, exponent };
  });
  const count = actual.length, featureValues = options.features;
  const featureCount = featureValues?.[0]?.length ?? 0;
  if (featureValues && (!Array.isArray(featureValues) || featureValues.length !== count || featureCount < 1 || featureCount > 4 || !Array.from(featureValues).every(row => Array.isArray(row) && row.length === featureCount && Array.from(row).every(v => Number.isInteger(v) && Math.abs(v) <= 4)))) throw new RangeError('Expected one to four bounded integer features per state');
  const precisions = [128, 256, 512, 1024, 2048, 4096, 8192];
  const minimumPrecision = options.minimumPrecision ?? 128, maximumPrecision = options.maximumPrecision ?? 8192;
  if (!precisions.includes(minimumPrecision) || !precisions.includes(maximumPrecision) || maximumPrecision < minimumPrecision) throw new RangeError('Invalid precision range');
  const active = actual.filter(v => v.coefficient !== Z), unit = active.length ? Math.min(...active.map(v => v.exponent)) : 0;
  const integers = actual.map(v => v.coefficient << BigInt(v.exponent - unit));
  const direction = T < 0 ? -1 : 1;
  let anchor = integers[0]; for (const v of integers) if (direction > 0 ? v < anchor : v > anchor) anchor = v;
  const costs = integers.map(v => direction > 0 ? v - anchor : anchor - v), gaps = costs.map(v => dyadic(v, unit));
  const anchorEnergy = dyadic(anchor, unit), g0 = costs.filter(v => v === Z).length, degenerate = g0 === count;
  const common = { scope: 'finite-exact-energy-canonical-core' as const,
    boltzmannConstant: kB, stateCount: count, anchorMultiplicity: g0, energyAnchor: exact(r(anchorEnergy)),
    mode: T === 0 ? 'positive-side-zero-temperature-limit' as const : T < 0 ? 'bounded-negative-temperature' as const : 'positive-temperature' as const,
    approximationContract: 'Exact binary input energies and directed exponential/logarithm enclosures. The stored kB literal is a binary64 numerical scale. Interval-resolved quantities have certified signs and relative enclosure width at most 2^-48. Cold leading moments omit relative corrections below 2^-100; floating logarithms in that branch are approximate, not certified. Unresolved is not zero. Probabilities are per labeled microstate, not per energy level.' as const };
  const meanGroups = Array.from({ length: featureCount }, (_, f) => {
    const groups = new Map<string, { cost: bigint; coefficient: bigint }>();
    for (let i = 0; i < count; i++) { const key = costs[i].toString(), g = groups.get(key) ?? { cost: costs[i], coefficient: Z }; g.coefficient += BigInt(featureValues![i][f]); groups.set(key, g); }
    return [...groups.values()].filter(g => g.coefficient !== Z);
  });
  const covarianceGroups = Array.from({ length: featureCount }, (_, f) => Array.from({ length: featureCount }, (_, h) => {
    const groups = new Map<string, { cost: bigint; coefficient: bigint }>();
    for (let i = 0; i < count; i++) for (let j = i + 1; j < count; j++) {
      const cost = costs[i] + costs[j], key = cost.toString(), g = groups.get(key) ?? { cost, coefficient: Z };
      g.coefficient += BigInt((featureValues![i][f] - featureValues![j][f]) * (featureValues![i][h] - featureValues![j][h])); groups.set(key, g);
    }
    return [...groups.values()].filter(g => g.coefficient !== Z);
  }));
  const ctx0 = createTranscendentalContext(128), groundLog = ctx0.logRatio(product(g0), one);
  const groundLogRange: Range = { lower: r(dyadic(groundLog.lower, -128)), upper: r(dyadic(groundLog.upper, -128)) };
  if (T === 0) {
    const entropyKB = g0 === 1 ? zero() : enclosed(groundLogRange), u = exact(r(anchorEnergy));
    const logZ = anchor === Z ? entropyKB : infinity(anchor < Z ? 1 : -1);
    const partition = anchor === Z ? exact(r(product(g0))) : anchor < Z ? infinity(1) : zero();
    return { ...common, precisionBits: 128, attempts: [128], fullyResolved: true, thermalEnergy: zero(), inverseTemperature: infinity(1),
      states: actual.map((energy, i) => ({ index: i, energy: exact(r(energy)),
        probability: costs[i] === Z ? exact(r(one, product(g0))) : zero(),
        logProbability: costs[i] === Z ? negativeObservation(entropyKB) : infinity(-1) })),
      meanEnergy: u, energyVariance: zero(), thirdEnergyCumulant: zero(), heatCapacity: zero(), heatCapacityInKB: zero(),
      entropyInKB: entropyKB, entropy: g0 === 1 ? zero() : enclosed(scaleRange(groundLogRange, product(kB))),
      centeredLogPartition: entropyKB, logPartition: logZ, partitionFunction: partition, freeEnergy: u,
      logZBetaDerivatives: { first: negativeObservation(u), second: zero(), third: zero() }, features: featureValues ? { means: meanGroups.map(groups => exact(r(dyadic(groups.filter(g => g.cost === Z).reduce((a, g) => a + g.coefficient, Z), 0), product(g0)))), covariance: covarianceGroups.map(row => row.map(groups => exact(r(dyadic(groups.filter(g => g.cost === Z).reduce((a, g) => a + g.coefficient, Z), 0), product(g0 * g0))))) } : undefined, coldTailApproximation: false };
  }
  const thermal = product(kB, Math.abs(T));
  const cold = !degenerate && gaps.every(g => g.coefficient === Z || sub(g, multiply(thermal, product(8192))).coefficient > Z);
  const excited = gaps.filter(g => g.coefficient !== Z), logThermal = observe(thermal).logMagnitude!;
  const coldLog = (power: number, extra = 0) => logSum(excited.map(g => sum([negate(g), multiply(thermal, product(power * observe(g).logMagnitude! - Math.log(g0) + extra))])), thermal);
  const coldR = cold ? logSum(excited.map(g => negate(g)), thermal) : null;
  const meanTail = cold ? tail(coldLog(1), thermal) : null;
  const varianceTail = cold ? tail(coldLog(2), thermal, 1, { lower: r(product(0)), upper: r(dyadic(O, -5000)) }) : null;
  const thirdTail = cold ? tail(coldLog(3), thermal, direction, direction > 0 ? { lower: r(product(0)), upper: r(dyadic(O, -4000)) } : { lower: r(dyadic(-O, -4000)), upper: r(product(0)) }) : null;
  const coldCapacity = cold ? tail(coldLog(2, -2 * logThermal), thermal) : null;
  const coldEntropyLog = cold ? logSum(excited.map(g => {
    const u = ratio(g, thermal), logFactor = u.value === null ? u.logMagnitude! : Math.log1p(u.value);
    return sum([negate(g), multiply(thermal, product(logFactor))]);
  }), thermal) : null;
  const attempts: number[] = [];
  let latest: ReturnType<typeof calculate> | undefined;
  function calculate(P: number) {
    const c = createTranscendentalContext(P), weights = gaps.map(g => c.expNegativeRatio(g, thermal));
    const Wl = weights.reduce((a, v) => a + v.lower, Z), Wu = weights.reduce((a, v) => a + v.upper, Z);
    let m1l = Z, m1u = Z, m2l = Z, m2u = Z, m3l = Z, m3u = Z;
    for (let i = 0; i < costs.length; i++) { const a = costs[i], w = weights[i]; m1l += w.lower * a; m1u += w.upper * a; m2l += w.lower * a * a; m2u += w.upper * a * a; m3l += w.lower * a * a * a; m3u += w.upper * a * a * a; }
    const lowLog = c.logRatio(dyadic(Wl, -P), one), highLog = c.logRatio(dyadic(Wu, -P), one);
    let logWRange: Range = { lower: r(dyadic(lowLog.lower, -P)), upper: r(dyadic(highLog.upper, -P)) };
    if (cold) {
      const base = c.logRatio(product(g0), one);
      logWRange = { lower: r(dyadic(base.lower, -P)), upper: add(r(dyadic(base.upper, -P)), r(dyadic(O, -4096))) };
    }
    const logW = cold && g0 === 1 ? tail(coldR!, thermal) : enclosed(logWRange, count === 1);
    const meanCostRange = ratioRange(m1l, m1u, Wl, Wu, unit);
    let uRange = addRange(exactRange(r(anchorEnergy)), scaleRange(meanCostRange, product(direction)));
    if (cold && anchor !== Z) uRange = direction > 0 ? { lower: r(anchorEnergy), upper: add(r(anchorEnergy), r(dyadic(O, -6000))) } : { lower: add(r(anchorEnergy), r(dyadic(-O, -6000))), upper: r(anchorEnergy) };
    const u = cold && anchor === Z ? (direction > 0 ? meanTail! : negativeObservation(meanTail!)) : degenerate ? exact(r(anchorEnergy)) : enclosed(uRange);
    let vl = Z, vu = Z;
    for (let i = 0; i < costs.length; i++) for (let j = i + 1; j < costs.length; j++) {
      const delta = costs[i] - costs[j], square = delta * delta;
      vl += weights[i].lower * weights[j].lower * square; vu += weights[i].upper * weights[j].upper * square;
    }
    const varianceRange = ratioRange(vl, vu, Wl * Wl, Wu * Wu, 2 * unit);
    const variance = degenerate ? zero() : cold ? varianceTail! : enclosed(varianceRange);
    const thirdLower = m3l * Wl * Wl - BigInt(3) * m2u * m1u * Wu + BigInt(2) * m1l ** BigInt(3);
    const thirdUpper = m3u * Wu * Wu - BigInt(3) * m2l * m1l * Wl + BigInt(2) * m1u ** BigInt(3);
    let thirdRange = ratioRange(thirdLower, thirdUpper, Wl ** BigInt(3), Wu ** BigInt(3), 3 * unit);
    if (direction < 0) thirdRange = negateRange(thirdRange);
    const third = degenerate ? zero() : cold ? thirdTail! : enclosed(thirdRange);
    const capacityRange = scaleRange(varianceRange, one, multiply(thermal, thermal));
    const capacityKB = degenerate ? zero() : cold ? coldCapacity! : enclosed(capacityRange);
    const capacity = degenerate ? zero() : cold ? tail(coldLog(2, Math.log(kB) - 2 * logThermal), thermal) : enclosed(scaleRange(capacityRange, product(kB)));
    let sRange = addRange(logWRange, scaleRange(meanCostRange, one, thermal));
    if (cold && g0 > 1) sRange = { lower: logWRange.lower, upper: add(logWRange.upper, r(dyadic(O, -100))) };
    const sKB = count === 1 ? zero() : cold && g0 === 1 ? tail(coldEntropyLog!, thermal) : enclosed(sRange);
    const entropy = count === 1 ? zero() : cold && g0 === 1 ? tail(sum([coldEntropyLog!, multiply(thermal, product(Math.log(kB)))]), thermal) : enclosed(scaleRange(sRange, product(kB)));
    const fRange = addRange(exactRange(r(anchorEnergy)), scaleRange(logWRange, multiply(thermal, product(-direction))));
    const f = cold && anchor === Z && g0 === 1 ? tail(sum([coldR!, multiply(thermal, product(logThermal))]), thermal, -direction) : enclosed(fRange, count === 1 && anchor === Z);
    const logZRange = addRange(logWRange, exactRange(r(multiply(anchorEnergy, product(-direction)), thermal)));
    const logZ = cold && anchor === Z && g0 === 1 ? logW : enclosed(logZRange, count === 1 && anchor === Z);
    const states = actual.map((energy, i) => {
      const lpRange = addRange(negateRange(logWRange), exactRange(r(negate(gaps[i]), thermal)));
      const lp = cold && costs[i] === Z && g0 === 1 ? negativeObservation(logW) : enclosed(lpRange, count === 1);
      const p = degenerate ? { ...exact(r(one, product(count))), logMagnitude: lp.value, logResolution: lp.value === null ? 'outside-number-range' as const : 'finite' as const } : exponential(lp, cold && costs[i] === Z && g0 === 1 ? null : lpRange);
      return { index: i, energy: exact(r(energy)), probability: { ...p, bounds: p.bounds ?? bounds(ratioRange(weights[i].lower, weights[i].upper, Wl, Wu)) }, logProbability: lp };
    });
    const featureCache = new Map<string, { lower: bigint; upper: bigint }>();
    const groupedNumerator = (groups: { cost: bigint; coefficient: bigint }[]) => {
      let lower = Z, upper = Z;
      for (const group of groups) {
        const key = group.cost.toString(); let weight = featureCache.get(key);
        if (!weight) { weight = c.expNegativeRatio(dyadic(group.cost, unit), thermal); featureCache.set(key, weight); }
        lower += group.coefficient * (group.coefficient > Z ? weight.lower : weight.upper);
        upper += group.coefficient * (group.coefficient > Z ? weight.upper : weight.lower);
      }
      return { lower, upper };
    };
    const features = featureValues ? {
      means: meanGroups.map(groups => { const n = groupedNumerator(groups); return groups.length ? enclosed(ratioRange(n.lower, n.upper, Wl, Wu)) : zero(); }),
      covariance: covarianceGroups.map((row, i) => row.map((groups, j) => {
        if (!groups.length) return zero();
        const n = groupedNumerator(groups), range = ratioRange(n.lower, n.upper, Wl * Wl, Wu * Wu, P);
        if (cold && i === j && !groups.some(g => g.cost === Z)) {
          const terms = groups.map(g => sum([negate(dyadic(g.cost, unit)), multiply(thermal, product(observe(dyadic(g.coefficient, 0)).logMagnitude! - 2 * Math.log(g0)))]));
          return tail(logSum(terms, thermal), thermal, 1, range);
        }
        const q = enclosed(range); return i === j && q.sign === null ? { ...q, sign: 1 } : q;
      }))
    } : undefined;
    const quantities = [u, variance, third, capacityKB, capacity, sKB, entropy, logW, f, logZ, ...states.flatMap(v => [v.probability, v.logProbability]), ...(features ? [...features.means, ...features.covariance.flat()] : [])];
    return { precisionBits: P, features, fullyResolved: quantities.every(q => q.resolution !== 'unresolved-at-precision-ceiling'), states,
      meanEnergy: u, energyVariance: variance, thirdEnergyCumulant: third, heatCapacity: capacity, heatCapacityInKB: capacityKB,
      entropyInKB: sKB, entropy, centeredLogPartition: logW, logPartition: logZ,
      partitionFunction: exponential(logZ, cold && anchor === Z && g0 === 1 ? null : logZRange), freeEnergy: f,
      logZBetaDerivatives: { first: negativeObservation(u), second: { ...variance }, third: negativeObservation(third) } };
  }
  for (const P of precisions.filter(v => v >= minimumPrecision && v <= maximumPrecision)) { attempts.push(P); latest = calculate(P); if (latest.fullyResolved) break; }
  return { ...common, ...latest!, attempts, thermalEnergy: exact(r(product(kB, T))), inverseTemperature: exact(r(product(direction), thermal)),
    coldTailApproximation: cold, coldTailOmissionRelativeBound: cold ? '2^-100 (omitted physical corrections only; not floating transcendental error)' : null };
}

export function partitionObservationRange(q: PartitionObservation): Range | null {
  if (!q.bounds) return null;
  const decode = (v: PartitionRational): Rational => r({ coefficient: BigInt(v.numerator.coefficient), exponent: v.numerator.exponent }, { coefficient: BigInt(v.denominator.coefficient), exponent: v.denominator.exponent });
  return { lower: decode(q.bounds.lower), upper: decode(q.bounds.upper) };
}
/** Turn a directed rational enclosure into a public observation. An exact
 * zero interval is a proof of zero; intervals merely containing zero are not. */
export function partitionObservationFromRange(range: Range): PartitionObservation {
  return enclosed(range, range.lower.n.coefficient === Z && range.upper.n.coefficient === Z);
}
export function scalePartitionObservation(q: PartitionObservation, numerator: BinaryValue, denominator = one): PartitionObservation {
  if (denominator.coefficient <= Z) throw new RangeError('Expected positive denominator');
  if (numerator.coefficient === Z || q.resolution === 'structural-zero') return zero();
  const range = partitionObservationRange(q);
  if (q.resolution !== 'log-tail-approximation' && range) return q.resolution === 'exact-rational' ? exact(scale(range.lower, numerator, denominator)) : enclosed(scaleRange(range, numerator, denominator));
  if (q.resolution === 'log-tail-approximation') {
    const factor = ratio(numerator, denominator), sign = (q.sign ?? 1) * factor.sign;
    const log = q.logMagnitude === null ? null : q.logMagnitude + factor.logMagnitude!;
    const value = log === null ? null : Math.exp(log);
    return { value: value === null || value === 0 || !Number.isFinite(value) ? null : sign * value, sign, logMagnitude: log, logResolution: log === null ? 'outside-number-range' : 'finite', resolution: 'log-tail-approximation', bounds: null };
  }
  return unresolved(null, q.sign === null ? null : q.sign * (numerator.coefficient > Z ? 1 : -1));
}
export function differencePartitionObservations(a: PartitionObservation, b: PartitionObservation, multiplier = 1): PartitionObservation {
  const x = partitionObservationRange(a), y = partitionObservationRange(b);
  return x && y ? partitionObservationFromRange(addRange(x, scaleRange(y, product(-multiplier)))) : unresolved(null);
}
