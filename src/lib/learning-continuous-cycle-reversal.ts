/** Positive three-state CTMC. Exact rational rates and path ratios;
 * directed logarithms of dimensionless ratios. No simulated trajectories. */
import { markovFraction as fraction, exactMarkovObservation as exact, type MarkovFraction as F } from '@/lib/learning-finite-markov-exact';
import { createTranscendentalContext } from '@/lib/learning-interval-transcendentals';
import { partitionObservationFromRange } from '@/lib/learning-partition-exact-core';

export interface ContinuousCycleReversalParameters {
  clockwise01: number;
  clockwise12: number;
  clockwise20: number;
  counter10: number;
  counter21: number;
  counter02: number;
  rateDenominator: number;
  states: readonly number[];
  dwellNumerators: readonly number[];
  timeDenominator: number;
}

const ZERO = BigInt(0), ONE = BigInt(1);
const zero = fraction(ZERO), one = fraction(ONE);
const add = (a: F, b: F) => fraction(a.n * b.d + b.n * a.d, a.d * b.d);
const neg = (a: F) => fraction(-a.n, a.d);
const sub = (a: F, b: F) => add(a, neg(b));
const mul = (a: F, b: F) => fraction(a.n * b.n, a.d * b.d);
const div = (a: F, b: F) => fraction(a.n * b.d, a.d * b.n);
const equal = (a: F, b: F) => a.n * b.d === b.n * a.d;
const sum = (a: readonly F[]) => a.reduce(add, zero);
const output = (a: F) => exact(a.n, a.d);
interface Interval { lower: F; upper: F }
const point = (a: F): Interval => ({ lower: a, upper: a });
const intervalAdd = (a: Interval, b: Interval): Interval => ({ lower: add(a.lower, b.lower), upper: add(a.upper, b.upper) });
const scale = (a: Interval, b: F): Interval => b.n < ZERO
  ? { lower: mul(a.upper, b), upper: mul(a.lower, b) }
  : { lower: mul(a.lower, b), upper: mul(a.upper, b) };
function observe(a: Interval) {
  if (equal(a.lower, a.upper)) return output(a.lower);
  const binary = (coefficient: bigint) => ({ coefficient, exponent: 0 });
  return partitionObservationFromRange({
    lower: { n: binary(a.lower.n), d: binary(a.lower.d) },
    upper: { n: binary(a.upper.n), d: binary(a.upper.d) },
  });
}

const rateFields = ['clockwise01', 'clockwise12', 'clockwise20', 'counter10', 'counter21', 'counter02'] as const;
const fields = [...rateFields, 'rateDenominator', 'states', 'dwellNumerators', 'timeDenominator'];
const integer = (n: unknown, minimum = 1, maximum = 1000000): n is number =>
  typeof n === 'number' && Number.isInteger(n) && n >= minimum && n <= maximum;
function denseArray(value: unknown): value is number[] {
  return Array.isArray(value) && Object.getPrototypeOf(value) === Array.prototype
    && Reflect.ownKeys(value).length === value.length + 1
    && Array.from({ length: value.length }, (_, i) => Object.prototype.hasOwnProperty.call(value, i)).every(Boolean);
}
export function isContinuousCycleReversalParameters(value: unknown): value is ContinuousCycleReversalParameters {
  if (!value || typeof value !== 'object' || Object.getPrototypeOf(value) !== Object.prototype) return false;
  const p = value as Record<string, unknown>;
  if (Reflect.ownKeys(p).length !== fields.length || !fields.every(k => Object.prototype.hasOwnProperty.call(p, k))) return false;
  if (!rateFields.every(k => integer(p[k])) || !integer(p.rateDenominator) || !integer(p.timeDenominator)) return false;
  if (!denseArray(p.states) || !denseArray(p.dwellNumerators) || p.states.length < 1 || p.states.length > 65
    || p.states.length !== p.dwellNumerators.length) return false;
  return p.states.every((state, i, states) => integer(state, 0, 2) && (i === 0 || state !== states[i - 1]))
    && p.dwellNumerators.every((n, i, values) => integer(n, i === values.length - 1 ? 0 : 1));
}

export function continuousCycleReversalModel(input: ContinuousCycleReversalParameters) {
  if (!isContinuousCycleReversalParameters(input)) throw new RangeError(
    'Six positive integer rate numerators and two denominators in1..1000000;1..65 states in0..2; adjacent states differ; matching integer dwell numerators in1..1000000, with final dwell allowed0.');
  const p = Object.freeze({ ...input, states: Object.freeze(input.states.map(n => n + 0)), dwellNumerators: Object.freeze(input.dwellNumerators.map(n => n + 0)) });
  const rational = (n: number, d: number) => fraction(BigInt(n), BigInt(d));
  const [a, b, c, d, e, f] = rateFields.map(k => rational(p[k], p.rateDenominator));
  const Q = [[neg(add(a, f)), a, f], [d, neg(add(d, b)), b], [c, e, neg(add(c, e))]];
  const treeWeights = [sum([mul(c, d), mul(d, e), mul(b, c)]), sum([mul(a, e), mul(a, c), mul(e, f)]), sum([mul(b, f), mul(a, b), mul(d, f)])];
  const normalization = sum(treeWeights), pi = treeWeights.map(t => div(t, normalization));
  const escape = Q.map((row, i) => neg(row[i]));
  const reversed = Q.map((row, i) => row.map((q, j) => i === j ? q : div(mul(pi[j], Q[j][i]), pi[i])));
  const forwardProduct = mul(mul(a, b), c), backwardProduct = mul(mul(d, e), f);
  const cycleRatio = div(forwardProduct, backwardProduct), current = div(sub(forwardProduct, backwardProduct), normalization);
  const ctx = createTranscendentalContext(256);
  const log = (q: F): Interval => {
    if (q.n <= ZERO) throw new RangeError('Logarithm needs a positive dimensionless ratio');
    if (q.n === q.d) return point(zero);
    const interval = ctx.logRatio({ coefficient: q.n, exponent: 0 }, { coefficient: q.d, exponent: 0 });
    return { lower: fraction(interval.lower, ctx.scale), upper: fraction(interval.upper, ctx.scale) };
  };
  const affinity = log(cycleRatio), relativeEntropyRate = scale(affinity, current);
  const orientedEdges = [[0, 1], [1, 2], [2, 0]].map(([i, j]) => {
    const forward = mul(pi[i], Q[i][j]), backward = mul(pi[j], Q[j][i]);
    const net = sub(forward, backward), ratio = div(forward, backward), logarithm = log(ratio);
    return { from: i, to: j, forwardFlow: output(forward), backwardFlow: output(backward), netCurrent: output(net),
      stationaryFlowRatio: output(ratio), logStationaryFlowRatio: observe(logarithm), relativeEntropyContribution: observe(scale(logarithm, net)) };
  });
  const activity = sum(pi.map((v, i) => mul(v, escape[i])));
  const jumpStationary = pi.map((v, i) => div(mul(v, escape[i]), activity));
  let elapsed = zero, integratedEscape = zero, rateProduct = one, reverseRateProduct = one, finalRatio = one;
  const occupied = [zero, zero, zero], first = p.states[0];
  const path = p.states.map((state, index) => {
    const dwell = rational(p.dwellNumerators[index], p.timeDenominator);
    elapsed = add(elapsed, dwell); occupied[state] = add(occupied[state], dwell);
    integratedEscape = add(integratedEscape, mul(escape[state], dwell));
    const endpointRatio = div(pi[first], pi[state]), jumpRatio = div(rateProduct, reverseRateProduct);
    const ratio = mul(endpointRatio, jumpRatio);
    finalRatio = ratio;
    const forwardCoefficient = mul(pi[first], rateProduct), reverseCoefficient = mul(pi[state], reverseRateProduct);
    const row = { index, state, dwell: output(dwell), elapsed: output(elapsed),
      integratedEscape: output(integratedEscape), endpointRatio: output(endpointRatio), jumpRateRatio: output(jumpRatio),
      pathRatio: output(ratio), endpointLogRatio: observe(log(endpointRatio)), jumpLogRatio: observe(log(jumpRatio)),
      totalLogRatio: observe(log(ratio)),
      logForwardDensity: observe(intervalAdd(log(forwardCoefficient), point(neg(integratedEscape)))),
      logReverseDensity: observe(intervalAdd(log(reverseCoefficient), point(neg(integratedEscape)))),
      pathRatioIsExactlyOne: ratio.n === ratio.d };
    if (index + 1 < p.states.length) {
      const next = p.states[index + 1]; rateProduct = mul(rateProduct, Q[state][next]);
      reverseRateProduct = mul(reverseRateProduct, Q[next][state]);
    }
    return row;
  });
  const final = path[path.length - 1];
  return { parameters: p, precisionBits: 256, generator: Q.map(row => row.map(output)), stationary: pi.map(output),
    treeWeights: treeWeights.map(output), treeNormalization: output(normalization), escapeRates: escape.map(output),
    reversedGenerator: reversed.map(row => row.map(output)), orientedEdges,
    stationaryResidual: pi.map((_, j) => output(sum(pi.map((v, i) => mul(v, Q[i][j]))))),
    generatorRowResidual: Q.map(row => output(sum(row))),
    reversedRowResidual: reversed.map(row => output(sum(row))),
    forwardCycleRateProduct: output(forwardProduct), backwardCycleRateProduct: output(backwardProduct),
    cycleRatio: output(cycleRatio), cycleCurrent: output(current), cycleAffinity: observe(affinity),
    pathRelativeEntropyRate: observe(relativeEntropyRate), stationaryActivity: output(activity),
    jumpEpochStationary: jumpStationary.map(output), totalTime: output(elapsed),
    chosenPathOccupation: elapsed.n === ZERO ? null : occupied.map(t => output(div(t, elapsed))),
    path, current: final,
    classification: { irreducible: true, allStatesEvenUnderTimeReversal: true,
      detailedBalance: equal(forwardProduct, backwardProduct), stationaryIsUniform: pi.every(q => equal(q, pi[0])),
      clockwiseCurrent: current.n > ZERO, counterclockwiseCurrent: current.n < ZERO,
      forwardPathFavored: finalRatio.n > finalRatio.d,
      pathRatioIsExactlyOne: final.pathRatioIsExactlyOne, chosenPathIsSampled: false,
      thermalInterpretationSpecified: false },
    approximationContract: 'Positive three-state continuous-time rates in1/s, row convention p prime=p Q, all states even under reversal, stationary initialization. Six exact integer numerators share a denominator; path dwell times are exact rational seconds with strictly positive pre-jump waits and nonnegative final wait. Stationary probabilities, currents, reversed rates and entire selected-path likelihood ratio are exact rational; logarithms are directed256bit enclosures with ln(1) exactly0. Log forward/reverse path densities use1/s reference rate and1s reference time, and are densities in ordered jump times, not point-path probabilities. Current prefix compares its own time-reversed path including dwell times; endpoint factors are retained. Chosen paths are specified examples, not Monte Carlo samples. Stationary relative-entropy rate is an ensemble path distinguishability, not the sign of every chosen path. No thermal bath, local detailed balance, heat, infinite/zero rate, odd variable or hidden-channel inference.' };
}
