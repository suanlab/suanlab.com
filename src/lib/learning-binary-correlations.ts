/** Two named classical spins with independent dimensionless source fields.
 * Fixed-point exponentials and rational interval propagation retain tiny
 * connected correlations. No subtraction of rounded reader observations.
 */
import { binaryProduct } from './learning-binary-products';
import { createTranscendentalContext } from './learning-interval-transcendentals';
import { partitionObservationFromRange, type PartitionObservation } from './learning-partition-exact-core';
import { rationalGreatestCommonDivisor as gcd } from './learning-rational-gcd';

export interface BinaryCorrelationsParameters {
  couplingK: number;
  fieldOne: number;
  fieldTwo: number;
  probeOne: number;
  probeTwo: number;
}
export interface BinaryCorrelationsOptions { minimumPrecision?: number; maximumPrecision?: number }
interface Q { n: bigint; d: bigint }
interface I { lower: Q; upper: Q }
const Z = BigInt(0), O = BigInt(1);
const abs = (n: bigint) => n < Z ? -n : n;
function q(n: bigint, d = O): Q { const g = gcd(abs(n), d); return { n: n / g, d: d / g }; }
function binary(v: number): Q {
  const b = binaryProduct(v);
  return b.exponent >= 0 ? q(b.coefficient << BigInt(b.exponent)) : q(b.coefficient, O << BigInt(-b.exponent));
}
const add = (a: Q, b: Q) => q(a.n * b.d + b.n * a.d, a.d * b.d);
const neg = (a: Q): Q => ({ n: -a.n, d: a.d });
const sub = (a: Q, b: Q) => add(a, neg(b));
const mul = (a: Q, b: Q) => q(a.n * b.n, a.d * b.d);
const cmp = (a: Q, b: Q) => a.n * b.d - b.n * a.d;
const point = (a: Q): I => ({ lower: a, upper: a });
const plus = (a: I, b: I): I => ({ lower: add(a.lower, b.lower), upper: add(a.upper, b.upper) });
const negative = (a: I): I => ({ lower: neg(a.upper), upper: neg(a.lower) });
const minus = (a: I, b: I) => plus(a, negative(b));
function times(a: I, b: I): I {
  const v = [mul(a.lower, b.lower), mul(a.lower, b.upper), mul(a.upper, b.lower), mul(a.upper, b.upper)];
  return { lower: v.reduce((x, y) => cmp(x, y) < Z ? x : y), upper: v.reduce((x, y) => cmp(x, y) > Z ? x : y) };
}
function divide(a: I, b: I): I {
  if (b.lower.n <= Z) throw new RangeError('Positive interval denominator required');
  return times(a, { lower: q(b.upper.d, b.upper.n), upper: q(b.lower.d, b.lower.n) });
}
const integer = (v: number) => point(q(BigInt(v)));
const dyadic = (coefficient: bigint) => ({ coefficient, exponent: 0 });
const observe = (v: I): PartitionObservation => partitionObservationFromRange({
  lower: { n: dyadic(v.lower.n), d: dyadic(v.lower.d) }, upper: { n: dyadic(v.upper.n), d: dyadic(v.upper.d) },
});
const zero = () => point(q(Z));
const precisions = [128, 256, 512, 1024, 2048, 4096, 8192];
const spins = [[-1, -1], [-1, 1], [1, -1], [1, 1]] as const;

export function binaryConnectedCorrelationsModel(input: BinaryCorrelationsParameters, options: BinaryCorrelationsOptions = {}) {
  if (!input || ![input.couplingK, input.fieldOne, input.fieldTwo].every(v => Number.isFinite(v) && Math.abs(v) <= 8)
    || ![input.probeOne, input.probeTwo].every(v => Number.isFinite(v) && Math.abs(v) <= 4))
    throw new RangeError('Expected dimensionless K,h1,h2 in [-8,8] and probe coefficients in [-4,4]');
  const minimum = options.minimumPrecision ?? 128, maximum = options.maximumPrecision ?? 8192;
  if (!precisions.includes(minimum) || !precisions.includes(maximum) || minimum > maximum) throw new RangeError('Invalid precision range');
  const parameters = Object.freeze({ couplingK: input.couplingK, fieldOne: input.fieldOne, fieldTwo: input.fieldTwo, probeOne: input.probeOne, probeTwo: input.probeTwo });
  const K = binary(input.couplingK), h1 = binary(input.fieldOne), h2 = binary(input.fieldTwo), a1 = binary(input.probeOne), a2 = binary(input.probeTwo);
  const scores = spins.map(([s1, s2]) => add(add(mul(K, q(BigInt(s1 * s2))), mul(h1, q(BigInt(s1)))), mul(h2, q(BigInt(s2)))));
  const anchor = scores.reduce((a, b) => cmp(a, b) > Z ? a : b), costs = scores.map(s => sub(anchor, s));
  const probes = spins.map(([s1, s2]) => add(mul(a1, q(BigInt(s1))), mul(a2, q(BigInt(s2)))));
  const attempts: number[] = [];
  function calculate(P: number) {
    const ctx = createTranscendentalContext(P), S = O << BigInt(P), cache = new Map<string, I>();
    const weights = costs.map(c => {
      const key = `${c.n}/${c.d}`; let v = cache.get(key);
      if (!v) { const w = ctx.expNegativeRatio(dyadic(c.n), dyadic(c.d)); v = { lower: q(w.lower, S), upper: q(w.upper, S) }; cache.set(key, v); }
      return v;
    });
    const W = weights.reduce(plus, zero()), W2 = times(W, W), W3 = times(W2, W);
    const fixedLog = (v: Q): I => { const l = ctx.logRatio(dyadic(v.n), dyadic(v.d)); return { lower: q(l.lower, S), upper: q(l.upper, S) }; };
    const logW: I = { lower: fixedLog(W.lower).lower, upper: fixedLog(W.upper).upper };
    // Group identical exact exponents before interval multiplication. This
    // proves cancellation by symmetry, rather than inferring it from a plot.
    function expectation(values: Q[]) {
      const groups = new Map<string, { weight: I; coefficient: Q }>();
      costs.forEach((c, i) => { const key = `${c.n}/${c.d}`, old = groups.get(key); groups.set(key, { weight: weights[i], coefficient: add(old?.coefficient ?? q(Z), values[i]) }); });
      return divide([...groups.values()].reduce((s, g) => plus(s, times(g.weight, point(g.coefficient))), zero()), W);
    }
    const means = [0, 1].map(j => expectation(spins.map(s => q(BigInt(s[j])))));
    const pairMean = expectation(spins.map(s => q(BigInt(s[0] * s[1]))));
    const diagOne = divide(times(integer(4), times(plus(weights[0], weights[1]), plus(weights[2], weights[3]))), W2);
    const diagTwo = divide(times(integer(4), times(plus(weights[0], weights[2]), plus(weights[1], weights[3]))), W2);
    // Cov(s1,s2)=4*(w--*w++ - w-+*w+-)/W². K=0
    // factorizes exactly for every field, including subnormal fields.
    const cross = input.couplingK === 0 ? zero() : divide(times(integer(4), minus(times(weights[0], weights[3]), times(weights[1], weights[2]))), W2);
    // det Cov =16*sum_{i<j<k} p_i*p_j*p_k, a positive
    // representation avoiding cancellation near saturated alignment.
    let triples = zero();
    for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) for (let k = j + 1; k < 4; k++) triples = plus(triples, times(weights[i], times(weights[j], weights[k])));
    const determinant = divide(times(integer(16), triples), W3);
    function variance(values: Q[]) {
      let total = zero();
      for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) {
        const d = sub(values[i], values[j]); total = plus(total, times(point(mul(d, d)), times(weights[i], weights[j])));
      }
      return divide(total, W2);
    }
    const probeMean = expectation(probes), probeVariance = variance(probes);
    const response = times(point(q(O, BigInt(2))), variance(spins.map(([s1, s2]) => q(BigInt(s1 + s2)))));
    const states = spins.map((s, i) => ({ spins: [...s], probability: observe(divide(weights[i], W)), logProbability: observe(negative(plus(point(costs[i]), logW))) }));
    const result = { means: means.map(observe), pairMean: observe(pairMean), covariance: [[observe(diagOne), observe(cross)], [observe(cross), observe(diagTwo)]],
      covarianceDeterminant: observe(determinant), logPartition: observe(plus(point(anchor), logW)), probeMean: observe(probeMean), probeVariance: observe(probeVariance), commonFieldResponsePerSpin: observe(response), states };
    // Exact sign theorem from exp(2K)-exp(-2K), even if capped.
    result.covariance[0][1].sign = result.covariance[1][0].sign = Math.sign(input.couplingK) || 0;
    const observations = [result.pairMean, result.covarianceDeterminant, result.logPartition, result.probeMean, result.probeVariance, result.commonFieldResponsePerSpin,
      ...result.means, ...result.covariance.flat(), ...states.flatMap(s => [s.probability, s.logProbability])];
    return { ...result, precisionBits: P, fullyResolved: observations.every(v => v.resolution !== 'unresolved-at-precision-ceiling') };
  }
  let latest: ReturnType<typeof calculate> | undefined;
  for (const P of precisions.filter(v => v >= minimum && v <= maximum)) { attempts.push(P); latest = calculate(P); if (latest.fullyResolved) break; }
  return { scope: 'two-spin-connected-covariance-and-source-hessian' as const, parameters, ...latest!, attempts,
    approximationContract: 'Four named classical states with p proportional to exp(K*s1*s2+h1*s1+h2*s2). Supplied binary64 parameters are exact model inputs. Directed exponential/logarithm and rational enclosures resolve observations at relative width <=2^-48. C is the Hessian of logZ with respect to dimensionless local fields; physical field derivatives need 1/(kBT). Cross covariance can be negative; C is strictly positive definite at all finite parameters. Probe A=a1*s1+a2*s2 has variance and source response a^T*C*a. Positive pair sums avoid cancellation in probe variance and determinant. General magnitude-log displays are approximate. Precision-capped unavailable values are not zero. No dynamics, causal effect, infinite-size transition, spatial correlation length or Monte Carlo uncertainty is inferred.' };
}
