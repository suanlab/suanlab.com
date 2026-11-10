/** Exact binomial multiplicities and directed finite Curie–Weiss sums.
 * Dimensionless inputs: K=J/(kBT), b=h/(kBT); self interaction is included.
 * No finite-N peak, mode or nonzero |m| is a thermodynamic singularity.
 */
import { binaryProduct } from './learning-binary-products';
import { createTranscendentalContext, type FixedInterval } from './learning-interval-transcendentals';
import { partitionObservationFromRange, type PartitionObservation } from './learning-partition-exact-core';
import { rationalGreatestCommonDivisor as gcd } from './learning-rational-gcd';
export interface CurieWeissParameters { spinCount: number; couplingK: number; fieldB: number }
export interface CurieWeissOptions { minimumPrecision?: number; maximumPrecision?: number }
interface Q { n: bigint; d: bigint }
interface I { lower: Q; upper: Q }
const Z = BigInt(0), O = BigInt(1), TWO = BigInt(2);
const abs = (n: bigint) => n < Z ? -n : n;
function q(n: bigint, d = O): Q {
  if (d <= Z) throw new RangeError('Expected a positive rational denominator');
  const g = gcd(abs(n), d); return { n: n / g, d: d / g };
}
function binary(x: number): Q { const b = binaryProduct(x); return b.exponent >= 0 ? q(b.coefficient << BigInt(b.exponent)) : q(b.coefficient, O << BigInt(-b.exponent)); }
const add = (a: Q, b: Q) => q(a.n * b.d + b.n * a.d, a.d * b.d);
const neg = (a: Q): Q => ({ n: -a.n, d: a.d });
const sub = (a: Q, b: Q) => add(a, neg(b));
const mul = (a: Q, b: Q) => q(a.n * b.n, a.d * b.d);
const cmp = (a: Q, b: Q) => a.n * b.d - b.n * a.d;
const point = (a: Q): I => ({ lower: a, upper: a });
const sum = (a: I, b: I): I => ({ lower: add(a.lower, b.lower), upper: add(a.upper, b.upper) });
const negative = (a: I): I => ({ lower: neg(a.upper), upper: neg(a.lower) });
const difference = (a: I, b: I) => sum(a, negative(b));
function times(a: I, b: I): I {
  const v = [mul(a.lower, b.lower), mul(a.lower, b.upper), mul(a.upper, b.lower), mul(a.upper, b.upper)];
  return { lower: v.reduce((x, y) => cmp(x, y) < Z ? x : y), upper: v.reduce((x, y) => cmp(x, y) > Z ? x : y) };
}
const ivAdd = (a: FixedInterval, b: FixedInterval): FixedInterval => ({ lower: a.lower + b.lower, upper: a.upper + b.upper });
const ivScale = (a: FixedInterval, n: bigint): FixedInterval => n >= Z ? { lower: a.lower * n, upper: a.upper * n } : { lower: a.upper * n, upper: a.lower * n };
const ivZero = (): FixedInterval => ({ lower: Z, upper: Z });
const dyadic = (coefficient: bigint) => ({ coefficient, exponent: 0 });
const observe = (a: I): PartitionObservation => partitionObservationFromRange({ lower: { n: dyadic(a.lower.n), d: dyadic(a.lower.d) }, upper: { n: dyadic(a.upper.n), d: dyadic(a.upper.d) } });
const precisions = [128, 256, 512, 1024, 2048, 4096, 8192];
export function finiteCurieWeissModel(input: CurieWeissParameters, options: CurieWeissOptions = {}) {
  if (!input || !Number.isInteger(input.spinCount) || input.spinCount < 1 || input.spinCount > 128
    || !Number.isFinite(input.couplingK) || input.couplingK < 0 || input.couplingK > 8
    || !Number.isFinite(input.fieldB) || Math.abs(input.fieldB) > 4)
    throw new RangeError('Expected N=1..128, ferromagnetic K=0..8 and dimensionless b=-4..4');
  const minimum = options.minimumPrecision ?? 128, maximum = options.maximumPrecision ?? 8192;
  if (!precisions.includes(minimum) || !precisions.includes(maximum) || minimum > maximum) throw new RangeError('Invalid precision range');
  const parameters = Object.freeze({ spinCount: input.spinCount, couplingK: input.couplingK, fieldB: input.fieldB }), N = input.spinCount, n = BigInt(N), K = binary(input.couplingK), b = binary(input.fieldB);
  const M = Array.from({ length: N + 1 }, (_, i) => BigInt(2 * i - N));
  const g = [O]; for (let i = 0; i < N; i++) g.push(g[i] * BigInt(N - i) / BigInt(i + 1));
  const scores = M.map(m => add(mul(K, q(m * m, TWO * n)), mul(b, q(m))));
  // A convex quadratic with K>=0 has its maximum at an endpoint.
  const anchor = add(mul(K, q(n, TWO)), mul(q(abs(b.n), b.d), q(n)));
  const costs = scores.map(s => sub(anchor, s));
  // Equal binomial factors share the same exact log(g). Cancel that common
  // term before comparing a subnormal field splitting, at every precision.
  const bestScoreByMultiplicity = new Map<string, Q>();
  g.forEach((value, i) => { const key = value.toString(), previous = bestScoreByMultiplicity.get(key); if (!previous || cmp(scores[i], previous) > Z) bestScoreByMultiplicity.set(key, scores[i]); });
  const attempts: number[] = [];
  function calculate(P: number) {
    const ctx = createTranscendentalContext(P), S = O << BigInt(P);
    const expCache = new Map<string, FixedInterval>(), logCache = new Map<string, I>();
    const fixed = (v: FixedInterval): I => ({ lower: q(v.lower, S), upper: q(v.upper, S) });
    const log = (a: Q): I => fixed(ctx.logRatio(dyadic(a.n), dyadic(a.d)));
    const logInterval = (a: I): I => ({ lower: log(a.lower).lower, upper: log(a.upper).upper });
    const raw = costs.map((cost, i) => {
      const key = `${cost.n}/${cost.d}`; let v = expCache.get(key);
      if (!v) { v = ctx.expNegativeRatio(dyadic(cost.n), dyadic(cost.d)); expCache.set(key, v); }
      return ivScale(v, g[i]);
    });
    const W = raw.reduce(ivAdd, ivZero());
    const normalize = (v: FixedInterval, denominator = O): I => {
      const quotients = [q(v.lower, W.lower * denominator), q(v.lower, W.upper * denominator), q(v.upper, W.lower * denominator), q(v.upper, W.upper * denominator)];
      return { lower: quotients.reduce((a, b) => cmp(a, b) < Z ? a : b), upper: quotients.reduce((a, b) => cmp(a, b) > Z ? a : b) };
    };
    const probabilities = raw.map(w => normalize(w));
    const logW = logInterval({ lower: q(W.lower, S), upper: q(W.upper, S) });
    const logG = g.map(value => {
      const key = value.toString(); let v = logCache.get(key); if (!v) { v = log(q(value)); logCache.set(key, v); } return v;
    });
    const logProbabilities = raw.map((w, i) => {
      if (w.lower > Z) {
        const quotient: I = { lower: q(W.lower, w.upper), upper: q(W.upper, w.lower) };
        if (cmp(quotient.lower, q(O)) < Z) quotient.lower = q(O); // proven W/w>=1
        return negative(logInterval(quotient));
      }
      return difference(difference(logG[i], point(costs[i])), logW);
    });
    const weightedMoment = (f: (m: bigint) => bigint, denominator: bigint) => normalize(raw.reduce((a, w, i) => ivAdd(a, ivScale(w, f(M[i]))), ivZero()), denominator);
    const meanM = input.fieldB === 0 ? point(q(Z)) : weightedMoment(m => m, O);
    const mean = times(meanM, point(q(O, n)));
    const meanAbsolute = N === 1 ? point(q(O)) : weightedMoment(abs, n);
    const meanM2 = N === 1 ? point(q(O)) : weightedMoment(m => m * m, O);
    const meanSquare = times(meanM2, point(q(O, n * n)));
    // Positive pair variance in O(N) using successive distance moments.
    // A=sum(w_i), D=sum(w_i*(M_j-M_i)), E=sum(w_i*(M_j-M_i)^2).
    let A = ivZero(), D = ivZero(), E = ivZero(), pair = ivZero();
    for (let j = 1; j <= N; j++) {
      A = ivAdd(A, raw[j - 1]);
      E = ivAdd(E, ivAdd(ivScale(D, BigInt(4)), ivScale(A, BigInt(4))));
      D = ivAdd(D, ivScale(A, TWO));
      pair = ivAdd(pair, { lower: raw[j].lower * E.lower, upper: raw[j].upper * E.upper });
    }
    const variance: I = { lower: q(pair.lower, W.upper * W.upper * n * n), upper: q(pair.upper, W.lower * W.lower * n * n) };
    const susceptibility = times(variance, point(q(n)));
    const macroEntropy = negative(probabilities.reduce((a, p, i) => sum(a, times(p, logProbabilities[i])), point(q(Z))));
    const conditionalEntropy = probabilities.reduce((a, p, i) => sum(a, times(p, logG[i])), point(q(Z)));
    const microEntropy = sum(macroEntropy, conditionalEntropy);
    const logPartition = sum(point(anchor), logW), pressure = times(logPartition, point(q(O, n)));
    const meanEnergy = negative(sum(times(point(K), times(meanM2, point(q(O, TWO * n)))), times(point(b), meanM)));
    const withoutSelf = difference(logPartition, point(mul(K, q(O, TWO))));
    const meanEnergyWithoutSelf = sum(meanEnergy, point(mul(K, q(O, TWO))));
    const logScores = scores.map((s, i) => sum(point(s), logG[i]));
    const highestLower = logScores.map(v => v.lower).reduce((a, b) => cmp(a, b) > Z ? a : b);
    const candidates = M.map((_, i) => i).filter(i => cmp(logScores[i].upper, highestLower) >= Z && cmp(scores[i], bestScoreByMultiplicity.get(g[i].toString())!) === Z);
    const first = candidates[0], certifiedModes = candidates.every(i => g[i] === g[first] && cmp(scores[i], scores[first]) === Z);
    const states = M.map((m, i) => {
      const probability = observe(probabilities[i]), logProbability = observe(logProbabilities[i]);
      probability.sign = 1; logProbability.sign = -1;
      probability.logMagnitude = logProbability.value;
      probability.logResolution = logProbability.resolution === 'unresolved-at-precision-ceiling' ? 'unresolved' : logProbability.value === null ? 'outside-number-range' : 'finite';
      return { plusSpinCount: i, totalMagnetization: Number(m), magnetizationPerSpin: observe(point(q(m, n))), multiplicity: g[i].toString(), probability, logProbability };
    });
    const observations = {
      meanMagnetization: observe(mean), meanAbsoluteMagnetization: observe(meanAbsolute), meanSquareMagnetization: observe(meanSquare),
      magnetizationVariance: observe(variance), fieldSusceptibility: observe(susceptibility),
      macroEntropyNats: observe(macroEntropy), conditionalMicrostateEntropyNats: observe(conditionalEntropy), microstateEntropyNats: observe(microEntropy),
      logPartition: observe(logPartition), pressurePerSpin: observe(pressure), freeEnergyInThermalUnits: observe(negative(logPartition)),
      freeEnergyPerSpinInThermalUnits: observe(negative(pressure)), meanEnergyInThermalUnits: observe(meanEnergy),
      logPartitionWithoutSelfInteraction: observe(withoutSelf), meanEnergyWithoutSelfInteractionInThermalUnits: observe(meanEnergyWithoutSelf),
      logZFieldDerivative: observe(meanM), logZCouplingDerivative: observe(times(meanM2, point(q(O, TWO * n)))),
    };
    if (input.fieldB !== 0) { observations.meanMagnetization.sign = Math.sign(input.fieldB); observations.logZFieldDerivative.sign = Math.sign(input.fieldB); }
    for (const key of ['meanAbsoluteMagnetization', 'meanSquareMagnetization', 'magnetizationVariance', 'fieldSusceptibility', 'macroEntropyNats', 'microstateEntropyNats'] as const) observations[key].sign = 1;
    if (N > 1) observations.conditionalMicrostateEntropyNats.sign = 1;
    if (input.couplingK !== 0 || input.fieldB !== 0) observations.meanEnergyInThermalUnits.sign = -1;
    // At b=0, tilting the binomial law by exp(K*M²/(2N)) increases E[M²].
    // A nonzero |b| tilts |M| further by cosh(b*M); b*E[M] is positive.
    // Hence the self-excluded ferro energy is strictly negative unless there
    // is no field and either N=1 or K=0. Retain this sign at a precision cap.
    if (input.fieldB !== 0 || N > 1 && input.couplingK > 0) observations.meanEnergyWithoutSelfInteractionInThermalUnits.sign = -1;
    const fullyResolved = certifiedModes && [...Object.values(observations), ...states.flatMap(s => [s.probability, s.logProbability])].every(v => v.resolution !== 'unresolved-at-precision-ceiling');
    return { ...observations, states, modes: { certified: certifiedModes, candidateTotalMagnetizations: candidates.map(i => Number(M[i])) }, precisionBits: P, fullyResolved };
  }
  let latest: ReturnType<typeof calculate> | undefined;
  for (const P of precisions.filter(v => v >= minimum && v <= maximum)) { attempts.push(P); latest = calculate(P); if (latest.fullyResolved) break; }
  return { scope: 'finite-ferromagnetic-curie-weiss' as const, parameters, macroStateCount: N + 1, microstateCount: (O << n).toString(),
    spinFlipSymmetric: input.fieldB === 0, ...latest!, attempts,
    approximationContract: 'Finite classical N=1..128; exact binomial integers and supplied binary64 dimensionless K=0..8,b=-4..4. Directed rational enclosures resolve at relative width <=2^-48. Self interaction is included in -K*M^2/(2N)-b*M; removing it shifts beta*E by K/2 and logZ by -K/2. Susceptibility is d<m>/db=N*Var(m), not a kelvin or physical-field derivative. General magnitude-log displays are approximate; the separate log-probability has a directed enclosure. Modes are certified by log-weight intervals and exact symmetry, not rounded probabilities. Unresolved is not zero. No thermodynamic-limit transition or physical dynamics is inferred.' };
}
