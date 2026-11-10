/** Exactly two allowed spatial configurations: all empty and all occupied.
 * Rational finite observables and directed logarithm intervals; complex root
 * coordinates are drawing projections, not numerical root certificates.
 */
import { markovFraction as f, exactMarkovObservation as exact, type MarkovFraction as F } from './learning-finite-markov-exact';
import { createTranscendentalContext } from './learning-interval-transcendentals';
import { partitionObservationFromRange, type PartitionObservation as Q } from './learning-partition-exact-core';
export interface TwoPhaseRoundingParameters {
 siteCount: number;
 activityNumerator: number;
 activityDenominator: number;
 sizePath: 'fixed' | 'inverse-size' | 'inverse-size-squared';
 approachSide: -1 | 1;
 offsetNumerator: number;
 offsetDenominator: number;
}
const Z = BigInt(0), O = BigInt(1), T = BigInt(2);
interface I { lower: F; upper: F }
const add = (a: F, b: F) => f(a.n * b.d + b.n * a.d, a.d * b.d);
const multiply = (a: F, b: F) => f(a.n * b.n, a.d * b.d);
const divide = (a: F, b: F) => f(a.n * b.d, a.d * b.n);
const subtract = (a: F, b: F) => add(a, f(-b.n, b.d));
const obs = (a: F) => exact(a.n, a.d);
const point = (q: F): I => ({ lower: q, upper: q });
const scale = (q: I, a: F): I => ({ lower: multiply(q.lower, a), upper: multiply(q.upper, a) });
const sum = (a: I, b: I): I => ({ lower: add(a.lower, b.lower), upper: add(a.upper, b.upper) });
const dyadic = (n: bigint) => ({ coefficient: n, exponent: 0 });
const observe = (q: I): Q => q.lower.n * q.upper.d === q.upper.n * q.lower.d ? obs(q.lower) : partitionObservationFromRange({ lower: { n: dyadic(q.lower.n), d: dyadic(q.lower.d) }, upper: { n: dyadic(q.upper.n), d: dyadic(q.upper.d) } });
const integer = (v: unknown, low: number, high = 1000000): v is number => typeof v === 'number' && Number.isInteger(v) && v >= low && v <= high;
export function isTwoPhaseRoundingParameters(value: unknown): value is TwoPhaseRoundingParameters {
 if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
 const p = value as Record<string, unknown>, keys = ['siteCount', 'activityNumerator', 'activityDenominator', 'sizePath', 'approachSide', 'offsetNumerator', 'offsetDenominator'];
 return Object.keys(p).length === keys.length && keys.every(k => Object.prototype.hasOwnProperty.call(p, k))
  && integer(p.siteCount, 1, 512) && integer(p.activityNumerator, 1) && integer(p.activityDenominator, 1)
  && integer(p.offsetNumerator, 0) && integer(p.offsetDenominator, 1) && p.offsetNumerator <= p.offsetDenominator
  && (p.approachSide === 1 || p.approachSide === -1) && ['fixed', 'inverse-size', 'inverse-size-squared'].includes(p.sizePath as string);
}
/** ln(1+t)=2q H(q²), q=t/(2+t), H(u)=sum u^j/(2j+1).
 * Keep q as an exact rational, however small. Only H is quantized, with
 * directed products and tail <= 9/[8(2J+3)9^(J+1)] for J=72, u<=1/9.
 */
function positiveLogOnePlus(t: F): I {
 const q = f(t.n, T * t.d + t.n), S = O << BigInt(192), uN = q.n * q.n, uD = q.d * q.d;
 const ceil = (n: bigint, d: bigint) => (n + d - O) / d;
 const uLo = uN * S / uD, uHi = ceil(uN * S, uD);
 let powerLo = S, powerHi = S, lo = S, hi = S;
 const terms = 72;
 for (let j = 1; j <= terms; j++) { powerLo = powerLo * uLo / S; powerHi = ceil(powerHi * uHi, S); const d = BigInt(2 * j + 1); lo += powerLo / d; hi += ceil(powerHi, d); }
 hi += ceil(BigInt(9) * S, BigInt(8 * (2 * terms + 3)) * BigInt(9) ** BigInt(terms + 1));
 return { lower: f(T * q.n * lo, q.d * S), upper: f(T * q.n * hi, q.d * S) };
}
export function twoPhaseRoundingModel(input: TwoPhaseRoundingParameters) {
 if (!isTwoPhaseRoundingParameters(input)) throw new RangeError('Expected exactly seven two-phase parameters: 1..512 sites, positive activity weights <=1000000, 0<=offset numerator<=positive denominator<=1000000, a supported size path and approach side ±1');
 const parameters = Object.freeze({ ...input, offsetNumerator: input.offsetNumerator + 0 });
 const ctx = createTranscendentalContext(192), C = BigInt(input.offsetNumerator), D = BigInt(input.offsetDenominator);
 const log = (q: F): I => { const r = ctx.logRatio(dyadic(q.n), dyadic(q.d)); return { lower: f(r.lower, ctx.scale), upper: f(r.upper, ctx.scale) }; };
 const logTwo = log(f(T)), logNine = log(f(BigInt(9)));
 function calculate(n: number) {
  const N = BigInt(n), power = input.sizePath === 'inverse-size' ? O : T;
  const pathBase = f(D * N ** power + C, D * N ** power);
  const z = input.sizePath === 'fixed' ? f(BigInt(input.activityNumerator), BigInt(input.activityDenominator)) : input.approachSide === 1 ? pathBase : f(pathBase.d, pathBase.n);
  const a = z.n ** N, b = z.d ** N, norm = a + b, p = f(a, norm), empty = f(b, norm), v = multiply(p, empty);
  const small = a < b ? a : b, large = a < b ? b : a, tailRatio = f(small, large);
  const field = log(z), branch = z.n > z.d ? field : point(f(Z));
  const correction = scale(positiveLogOnePlus(tailRatio), f(O, N));
  return { siteCount: n, activity: obs(z), activityPower: exact(a, b), partitionFunction: exact(norm, b), occupiedProbability: obs(p), emptyProbability: obs(empty),
   meanDensity: obs(p), densityVariance: obs(v), meanCount: obs(multiply(f(N), p)), countVariance: obs(multiply(f(N * N), v)), densityResponseToField: obs(multiply(f(N), v)), countResponseToField: obs(multiply(f(N * N), v)),
   minorityProbability: exact(small, norm), minorityToMajorityWeight: obs(tailRatio), exponentialPressureCorrection: exact(large + small, large),
   field: observe(field), scaledField: observe(scale(field, f(N))), limitingPressureAtCurrentField: observe(branch), pressure: observe(sum(branch, correction)), pressureCorrection: observe(correction), pressureCorrectionUpperBound: observe(scale(logTwo, f(O, N))),
   tenToNinetyFieldWidth: observe(scale(logNine, f(T, N))), allowedConfigurationCount: 2,
   distribution: [{ count: 0, density: exact(Z), probability: obs(empty) }, { count: n, density: exact(O), probability: obs(p) }],
   densityVarianceFromNamedStates: obs(add(multiply(multiply(p, p), empty), multiply(multiply(empty, empty), p))),
   approximationContract: 'Exactly two allowed configurations, each with reference weight one: all empty (weight 1) and all occupied (weight z^N). Derivatives are with respect to g=ln z at fixed N, not total derivatives along the selected N path. Pressure is ln(Z)/N and has no physical units or temperature assigned. The width is the g interval from occupied probability 0.1 to 0.9 for this fixed-N law, not a width measured in z. The limiting branch at the current field is max(0,g), not necessarily the limit along an N-dependent path.' };
 }
 const current = calculate(input.siteCount), sizes: number[] = []; for (let n = 1; n <= input.siteCount; n *= 2) sizes.push(n); if (sizes.at(-1) !== input.siteCount) sizes.push(input.siteCount);
 const history = sizes.map(n => n === input.siteCount ? current : calculate(n));
 const fixed = input.sizePath === 'fixed', fieldSign = Math.sign(input.activityNumerator - input.activityDenominator), coexist = fixed && fieldSign === 0;
 let densityLimit: I, varianceLimit: I;
 if (fixed) { const p = f(BigInt(fieldSign < 0 ? 0 : fieldSign > 0 ? 1 : 1), BigInt(fieldSign === 0 ? 2 : 1)); densityLimit = point(p); varianceLimit = point(multiply(p, subtract(f(O), p))); }
 else if (input.sizePath === 'inverse-size' && C > Z) {
  const e = ctx.expNegativeRatio(dyadic(C), dyadic(D)), lo = f(e.lower, ctx.scale), hi = f(e.upper, ctx.scale);
  const upperP = divide(f(O), add(f(O), lo)), lowerP = divide(f(O), add(f(O), hi));
  densityLimit = input.approachSide === 1 ? { lower: lowerP, upper: upperP } : { lower: subtract(f(O), upperP), upper: subtract(f(O), lowerP) };
  const variance = (x: F) => divide(x, multiply(add(f(O), x), add(f(O), x)));
  varianceLimit = { lower: variance(lo), upper: variance(hi) };
 } else { densityLimit = point(f(O, T)); varianceLimit = point(f(O, BigInt(4))); }
 const roots = Array.from({ length: input.siteCount }, (_, j) => {
  const numerator = BigInt(2 * j + 1), denominator = BigInt(2 * input.siteCount), negativeReal = 2 * j + 1 === input.siteCount;
  const angle = 2 * Math.PI * (2 * j + 1) / (2 * input.siteCount);
  return { index: j, turns: exact(numerator, denominator), poweredTurnsModuloOne: exact(O, T), conjugateIndex: input.siteCount - 1 - j,
   isNegativeRealRoot: negativeReal, plotReal: negativeReal ? -1 : Math.cos(angle), plotImaginary: negativeReal ? 0 : Math.sin(angle) };
 });
 return { parameters, current, history, roots,
  limits: { pressure: fixed && fieldSign > 0 ? observe(log(f(BigInt(input.activityNumerator), BigInt(input.activityDenominator)))) : exact(Z), meanDensity: observe(densityLimit), densityVariance: observe(varianceLimit),
   densityResponseToField: fixed && !coexist ? { kind: 'finite' as const, value: exact(Z) } : { kind: 'positive-infinity' as const },
   scaledField: fixed && !coexist ? { kind: fieldSign > 0 ? 'positive-infinity' as const : 'negative-infinity' as const } : { kind: 'finite' as const, value: exact(!fixed && input.sizePath === 'inverse-size' ? BigInt(input.approachSide) * C : Z, D) },
   densityLawAtoms: [{ density: exact(Z), probability: observe({ lower: subtract(f(O), densityLimit.upper), upper: subtract(f(O), densityLimit.lower) }) }, { density: exact(O), probability: observe(densityLimit) }].filter(atom => atom.probability.sign !== 0) },
  rootGeometry: { polynomial: '1+z^N', degree: input.siteCount, nearestPositiveAxisTurns: exact(O, T * BigInt(input.siteCount)), nearestLogFieldImaginaryPiFactor: exact(O, BigInt(input.siteCount)), negativeRealRootExists: input.siteCount % 2 === 1, positiveRealRootExists: false,
   closestRootDistanceToOnePlot: 2 * Math.sin(Math.PI / (2 * input.siteCount)), coordinatesAreApproximate: true },
  classification: { selfAveragesAlongPath: fixed && !coexist, nonzeroLimitingDensityVariance: !(fixed && !coexist), responseDivergesAlongPath: !(fixed && !coexist), finitePressureSmoothForPositiveActivity: true },
  fullyResolved: [...history.flatMap(r => [r.field, r.scaledField, r.pressure, r.pressureCorrection]), observe(densityLimit), observe(varianceLimit)].every(q => q.resolution !== 'unresolved-at-precision-ceiling'),
  approximationContract: 'Positive rational activity, two allowed states only; this is not (1+z)^N, an Ising/Curie-Weiss state sum or a general Lee-Yang theorem. Fixed path uses activity weights. Approach paths use 0<=c<=1 from offset weights and z_N=1+c/N or 1+c/N², or their reciprocals; activity weights are retained inputs but unused on those paths. Side is unused on the fixed path. Exact finite probabilities and moments retain rational point bounds. Logarithms have directed rational intervals at 192 bits; a tiny ln(1+t) factors out exact t/(2+t) before quantizing the remaining bounded series. No rounded 1+t subtraction is used. Complex root angles and polynomial identities are exact symbolic data; sin/cos and distances are drawing approximations. History is a spatial size path, not time. Smooth finite pressure and O(1/N) pressure error do not imply uniform derivative convergence or self-averaging at coexistence. No thermodynamic temperature, free energy in physical units, or sampler guarantees are inferred.' };
}
