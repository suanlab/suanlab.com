/** Certified time-domain extension of the protocol-dependent three-state lesson.
 * Positive uniformization with scaling/squaring; never normalize a truncated law.
 * Static/frequency references remain in their independent exact-rational model. */
import { finiteStateLinearResponseModel, isFiniteStateResponseParameters, type FiniteStateResponseParameters } from './learning-finite-state-linear-response';
import { markovFraction as f, exactMarkovObservation as exact, type MarkovFraction as F } from './learning-finite-markov-exact';
import { createTranscendentalContext, type FixedInterval as I } from './learning-interval-transcendentals';
import { partitionObservationFromRange, type PartitionObservation as Observation } from './learning-partition-exact-core';

export interface FiniteStateResponseExperimentParameters extends FiniteStateResponseParameters {
  timeNumerator: number; timeDenominator: number; sampleSteps: number;
  amplitudeNumerator: number; amplitudeDenominator: number;
}
const baseKeys = ['a', 'b', 'c', 'd', 'e', 'f', 'rateDenominator', 'kineticNumerator', 'kineticDenominator', 'frequencyNumerator', 'frequencyDenominator'] as const;
const extraKeys = ['timeNumerator', 'timeDenominator', 'sampleSteps', 'amplitudeNumerator', 'amplitudeDenominator'] as const;
const allKeys = [...baseKeys, ...extraKeys];
const Z = BigInt(0), O = BigInt(1), TWO = BigInt(2), BITS = 256, S = O << BigInt(BITS);
const zero = f(Z), half = f(O, TWO);
const add = (a: F, b: F) => f(a.n * b.d + b.n * a.d, a.d * b.d);
const neg = (a: F) => f(-a.n, a.d);
const sub = (a: F, b: F) => add(a, neg(b));
const mul = (a: F, b: F) => f(a.n * b.n, a.d * b.d);
const floor = (n: bigint, d: bigint) => n >= Z ? n / d : -((-n + d - O) / d);
const ceil = (n: bigint, d: bigint) => -floor(-n, d);
const min = (a: bigint, b: bigint) => a < b ? a : b;
const max = (a: bigint, b: bigint) => a > b ? a : b;
const point = (x: F): I => ({ lower: floor(x.n * S, x.d), upper: ceil(x.n * S, x.d) });
const plus = (a: I, b: I): I => ({ lower: a.lower + b.lower, upper: a.upper + b.upper });
const minus = (a: I, b: I): I => ({ lower: a.lower - b.upper, upper: a.upper - b.lower });
const scale = (a: I, x: F): I => x.n >= Z
  ? { lower: floor(a.lower * x.n, x.d), upper: ceil(a.upper * x.n, x.d) }
  : { lower: floor(a.upper * x.n, x.d), upper: ceil(a.lower * x.n, x.d) };
const positiveProduct = (a: I, b: I): I => ({ lower: floor(a.lower * b.lower, S), upper: ceil(a.upper * b.upper, S) });
const probability = (a: I): I => ({ lower: max(Z, a.lower), upper: min(S, a.upper) });
const binary = (x: bigint) => ({ coefficient: x, exponent: 0 });
const observe = (a: I): Observation => a.lower === a.upper ? exact(a.lower, S) : partitionObservationFromRange({
  lower: { n: binary(a.lower), d: binary(S) }, upper: { n: binary(a.upper), d: binary(S) },
});
const observeRational = (x: F) => exact(x.n, x.d);
const decode = (x: Observation): F => {
  const b = x.bounds!.lower;
  return f(BigInt(b.numerator.coefficient), BigInt(b.denominator.coefficient));
};
const identity = (): I[][] => Array.from({ length: 3 }, (_, i) => Array.from({ length: 3 }, (_, j) => ({ lower: i === j ? S : Z, upper: i === j ? S : Z })));
const matrixProduct = (a: I[][], b: I[][]): I[][] => a.map(row => b[0].map((_, j) => probability(row.reduce((sum, x, k) => plus(sum, positiveProduct(x, b[k][j])), { lower: Z, upper: Z }))));
const rowProduct = (row: F[], matrix: I[][]): I[] => matrix[0].map((_, j) => row.reduce((sum, x, i) => plus(sum, scale(matrix[i][j], x)), { lower: Z, upper: Z }));

export function isFiniteStateResponseExperimentParameters(value: unknown): value is FiniteStateResponseExperimentParameters {
  if (!value || typeof value !== 'object' || Object.getPrototypeOf(value) !== Object.prototype || Reflect.ownKeys(value).length !== allKeys.length) return false;
  const p = value as Record<string, unknown>;
  if (!allKeys.every(k => Object.prototype.hasOwnProperty.call(p, k)) || !isFiniteStateResponseParameters(Object.fromEntries(baseKeys.map(k => [k, p[k]])))) return false;
  if (!extraKeys.every(k => typeof p[k] === 'number' && Number.isSafeInteger(p[k]) && (p[k] as number) >= (k === 'amplitudeNumerator' ? -1000000 : k === 'timeNumerator' ? 0 : 1) && (p[k] as number) <= (k === 'sampleSteps' ? 64 : 1000000))) return false;
  const v = p as unknown as FiniteStateResponseExperimentParameters;
  return v.timeNumerator <= 4 * v.timeDenominator && Math.abs(v.amplitudeNumerator) <= v.amplitudeDenominator;
}

/** Each true row is stochastic. Clipping interval entries to [0,1] is an
 * intersection with that analytic bound, never a renormalization of a partial sum.
 * Even uncertain off-diagonal rates describe a single fixed generator within
 * their intervals; every matrix product encloses that same true semigroup. */
function propagate(rates: readonly I[], time: F) {
  if (time.n === Z) return { matrix: identity(), proof: { squarings: 0, retainedPoissonTerms: 1, basePoissonMean: observeRational(zero), baseTailUpperBound: observeRational(zero), method: 'exact-zero-time' as const } };
  const [a, b, c, d, e, rateF] = rates;
  const edges = [[a, rateF], [d, b], [c, e]];
  const clock = edges.reduce((m, row) => max(m, row[0].upper + row[1].upper), Z);
  const numerator = clock * time.n;
  let denominator = S * time.d, squarings = 0;
  while (TWO * numerator > denominator) { denominator *= TWO; squarings++; }
  const mu = f(numerator, denominator);
  const K = edges.map((row, i) => {
    const diagonal = { lower: S - ceil((row[0].upper + row[1].upper) * S, clock), upper: S - floor((row[0].lower + row[1].lower) * S, clock) };
    const off = row.map(x => ({ lower: floor(x.lower * S, clock), upper: ceil(x.upper * S, clock) }));
    return i === 0 ? [diagonal, off[0], off[1]] : i === 1 ? [off[0], diagonal, off[1]] : [off[0], off[1], diagonal];
  });
  const ctx = createTranscendentalContext(BITS);
  let weight = ctx.expNegativeRatio(binary(mu.n), binary(mu.d)), power = identity();
  let result = power.map(row => row.map(x => positiveProduct(x, weight)));
  const cutoff = 64;
  for (let k = 1; k <= cutoff; k++) {
    power = matrixProduct(power, K);
    weight = scale(weight, f(mu.n, mu.d * BigInt(k)));
    result = result.map((row, i) => row.map((x, j) => plus(x, positiveProduct(power[i][j], weight))));
  }
  // mu<=1/2: sum_{k=N+1}∞ mu^k/k! <=
  // 2(N+2) / [2^(N+1)(N+1)! (2(N+2)-1)]. exp(-mu)<=1.
  let factorial = O;
  for (let k = 2; k <= cutoff + 1; k++) factorial *= BigInt(k);
  const tail = f(BigInt(2 * (cutoff + 2)), (O << BigInt(cutoff + 1)) * factorial * BigInt(2 * (cutoff + 2) - 1));
  const tailUpper = ceil(tail.n * S, tail.d);
  result = result.map(row => row.map(x => probability({ lower: x.lower, upper: x.upper + tailUpper })));
  for (let k = 0; k < squarings; k++) result = matrixProduct(result, result);
  return { matrix: result, proof: { squarings, retainedPoissonTerms: cutoff + 1, basePoissonMean: observeRational(mu), baseTailUpperBound: observeRational(tail), method: 'positive-uniformization-scaling-squaring' as const } };
}

/** exp(x) is derived by directed inversion for positive x. If exp(-x)'s
 * lower endpoint is zero at this precision, its reciprocal is unavailable.
 * No zero/Infinity rate is substituted for a finite positive mathematical rate. */
function perturbRate(rate: F, exponent: F): I | null {
  if (exponent.n === Z) return point(rate);
  const ctx = createTranscendentalContext(BITS), absolute = exponent.n < Z ? -exponent.n : exponent.n;
  const negative = ctx.expNegativeRatio(binary(absolute), binary(exponent.d));
  if (exponent.n < Z) return scale(negative, rate);
  if (negative.lower === Z) return null;
  return scale({ lower: floor(S * S, negative.upper), upper: ceil(S * S, negative.lower) }, rate);
}

export function finiteStateResponseExperiment(input: FiniteStateResponseExperimentParameters) {
  if (!isFiniteStateResponseExperimentParameters(input)) throw new RangeError('Sixteen owned integer fields, positive rates/denominators, time0..4s, sampleSteps1..64, amplitude-1..1');
  const parameters = Object.freeze(Object.fromEntries(allKeys.map(k => [k, input[k] + 0]))) as unknown as Readonly<FiniteStateResponseExperimentParameters>;
  const p = parameters, reference = finiteStateLinearResponseModel(Object.fromEntries(baseKeys.map(k => [k, p[k]])) as unknown as FiniteStateResponseParameters);
  const rates = [p.a, p.b, p.c, p.d, p.e, p.f].map(x => f(BigInt(x), BigInt(p.rateDenominator)));
  const pi = reference.stationaryLaw.map(decode), source = reference.signedSource.map(decode), prime = reference.observables.map(o => decode(o.stationaryDerivative));
  const generator = reference.generator.map(row => row.map(decode));
  const evolveRow = (row: F[]) => generator[0].map((_, j) => row.reduce((total, x, i) => add(total, mul(x, generator[i][j])), zero));
  const first = evolveRow(source), second = evolveRow(first);
  // Cayley-Hamilton: if s Q^k o=0 for k=0,1,2, the complete
  // exponential response for this observable is identically zero.
  const responseZeroStates = source.map((x, j) => x.n === Z && first[j].n === Z && second[j].n === Z);
  const sourceZero = source.every(x => x.n === Z), h = f(BigInt(p.amplitudeNumerator), BigInt(p.amplitudeDenominator));
  const eta = f(BigInt(p.kineticNumerator), BigInt(p.kineticDenominator));
  const perturbed = rates.map((rate, i) => i === 0 ? perturbRate(rate, mul(add(eta, half), h)) : i === 3 ? perturbRate(rate, mul(sub(eta, half), h)) : point(rate));
  const finiteRatesAvailable = perturbed.every((x): x is I => x !== null);
  const history = Array.from({ length: p.sampleSteps + 1 }, (_, step) => {
    const time = f(BigInt(p.timeNumerator) * BigInt(step), BigInt(p.timeDenominator) * BigInt(p.sampleSteps));
    const baseline = propagate(rates.map(point), time);
    const kernel = (time.n === Z ? source.map(point) : rowProduct(source, baseline.matrix)).map((x, j) => responseZeroStates[j] ? point(zero) : x);
    const stepResponse = rowProduct(prime, baseline.matrix).map((x, j) => time.n === Z || responseZeroStates[j] ? point(zero) : minus(point(prime[j]), x));
    const predicted = stepResponse.map((x, j) => plus(point(pi[j]), scale(x, h)));
    const unchanged = time.n === Z || h.n === Z;
    const finite = !unchanged && finiteRatesAvailable ? propagate(perturbed as I[], time) : null;
    const actual = unchanged ? pi.map(point) : finite ? rowProduct(pi, finite.matrix).map(probability) : null;
    const differences = actual ? actual.map((x, j) => minus(x, predicted[j])) : null;
    return {
      step, time: observeRational(time), kernel: kernel.map((x, j) => time.n === Z ? observeRational(source[j]) : observe(x)),
      stepSusceptibility: stepResponse.map(observe), predictedOccupancy: predicted.map((x, j) => unchanged || responseZeroStates[j] ? observeRational(pi[j]) : observe(x)),
      actualOccupancy: actual?.map((x, j) => unchanged ? observeRational(pi[j]) : observe(x)) ?? null,
      actualMinusLinearPrediction: differences?.map(x => unchanged ? observeRational(zero) : observe(x)) ?? null,
      finiteInputStatus: unchanged ? 'exact-unchanged-law' as const : finite ? 'certified-enclosure' as const : 'unresolved-positive-rate-reciprocal' as const,
      baselineProof: baseline.proof, finiteInputProof: finite?.proof ?? null,
    };
  });
  return {
    parameters, precisionBits: BITS, reference, amplitude: observeRational(h), history, current: history[history.length - 1],
    perturbedRates: perturbed.map(x => x ? observe(x) : null),
    classification: { timeDomainImplemented: true, finiteAmplitudeImplemented: true, stationaryInitialization: true, allMathematicalFiniteInputRatesPositive: true, finiteInputRatesEnclosed: finiteRatesAvailable, sourceExactlyZero: sourceZero, identicallyZeroResponseStates: responseZeroStates, normalizationOfTruncation: false, sampledPaths: false, eulerTimeStepping: false },
    approximationContract: 'Three-state CTMC with exact stationary initialization and row-law convention. Kernel s exp(Qt), step response piPrime(I-exp(Qt)), and finite-step law pi exp(Q(h)t). Signed tangents are never clipped to probability bounds. Directed256-bit positive uniformization at base Poisson mean<=1/2,65 retained terms and an explicit positive geometric tail, followed by stochastic matrix scaling/squaring. Matrix entry intersections with[0,1] use the analytic stochastic bound; no truncated law is normalized. Every displayed physical time is calculated independently, not an Euler step or sampled path. Finite-input exponential rates have directed enclosures; a positive rate whose reciprocal cannot be bounded at256bits is unavailable, not zero or infinity. At time0 or h0 the unchanged law and zero difference are exact identities. Exact sQ^k o=0 for k0,1,2 certifies an identically zero linear response by Cayley-Hamilton; this does not generally imply a zero finite-input effect. Actual-minus-tangent enclosures can be unresolved through cancellation; no unproved error order is returned. Preparation has no worker or reader registration yet.',
  };
}
