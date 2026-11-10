/** Positive finite-distribution KL kernel for the Helmholtz variational model.
 * Each supplied weight is a binary64 input; normalization stays rational until
 * observation. A positive target vector defines q, NOT an independently
 * specified Hamiltonian's exact canonical distribution. Logarithms approximate
 * transcendental values. Near equality uses nonnegative Bregman contributions.
 */
import { binaryProduct as product, binarySum as sum, binaryNegate as negate,
  binaryMultiply as multiply, binaryRatioObservation as observe,
  type BinaryValue, type BinaryObservation } from './learning-binary-products';

interface Fraction { numerator: BinaryValue; denominator: BinaryValue }
export interface RelativeEntropyMeasures {
  divergence: BinaryObservation;
  trialEntropy: BinaryObservation;
  targetEntropy: BinaryObservation;
  crossEntropy: BinaryObservation;
}
const one = product(1);
const zero = (): Fraction => ({ numerator: product(0), denominator: one });
const difference = (a: BinaryValue, b: BinaryValue) => sum([a, negate(b)]);
function smallRatio(n: BinaryValue, d: BinaryValue) {
  const absolute = n.coefficient < BigInt(0) ? negate(n) : n;
  return difference(multiply(absolute, product(8)), d).coefficient <= BigInt(0);
}
function gcd(a: bigint, b: bigint) {
  if (a < BigInt(0)) a = -a;
  while (b !== BigInt(0)) { const next = a % b; a = b; b = next; }
  return a;
}
function reduce(f: Fraction): Fraction {
  if (f.numerator.coefficient === BigInt(0)) return zero();
  const common = gcd(f.numerator.coefficient, f.denominator.coefficient);
  let n = f.numerator.coefficient / common, d = f.denominator.coefficient / common;
  let exponent = f.numerator.exponent - f.denominator.exponent;
  while (n % BigInt(2) === BigInt(0)) { n /= BigInt(2); exponent++; }
  while (d % BigInt(2) === BigInt(0)) { d /= BigInt(2); exponent--; }
  return { numerator: { coefficient: n, exponent }, denominator: { coefficient: d, exponent: 0 } };
}
function add(a: Fraction, b: Fraction): Fraction {
  if (a.numerator.coefficient === BigInt(0)) return reduce(b);
  if (b.numerator.coefficient === BigInt(0)) return a;
  return reduce({ numerator: sum([multiply(a.numerator, b.denominator), multiply(b.numerator, a.denominator)]),
    denominator: multiply(a.denominator, b.denominator) });
}
function observation(f: Fraction, scale = one) {
  const normalized = reduce({ numerator: multiply(f.numerator, scale), denominator: f.denominator });
  const result = observe(normalized.numerator, normalized.denominator);
  if (result.sign === 0) return result;
  const absolute = result.sign < 0 ? negate(normalized.numerator) : normalized.numerator;
  const log = logarithm(absolute, normalized.denominator);
  return { ...result, logMagnitude: observe(log.numerator, log.denominator).value };
}
/** ln(N/D) retains the pre-projection difference when a ratio rounds to one. */
function logarithm(n: BinaryValue, d: BinaryValue): Fraction {
  const gap = difference(n, d), delta = observe(gap, d);
  if (gap.coefficient === BigInt(0)) return zero();
  if (smallRatio(gap, d)) {
    const value = delta.value ?? 0;
    const gain = value === 0 ? 1 : Math.log1p(value) / value;
    return { numerator: multiply(gap, product(gain)), denominator: d };
  }
  const ratio = observe(n, d);
  const value = delta.value !== null && delta.value > -.75 && delta.value < 1
    ? Math.log1p(delta.value) : ratio.value === null || ratio.value < 2 ** -1022
      ? ratio.logMagnitude! : Math.log(ratio.value);
  return { numerator: product(value), denominator: one };
}
/** [(1+t)ln(1+t)-t]/t². With |t|<=1/8, omitting terms k>=33
 * has absolute error <= |t|^33/[35*34*(1-|t|)] <2e-33.
 * This truncation bound does not remove binary64 evaluation error.
 */
function bregmanGain(t: number) {
  let gain = 0;
  for (let k = 32; k >= 0; k--) gain = 1 / ((k + 1) * (k + 2)) - t * gain;
  return gain;
}
function weights(v: readonly number[], positive: boolean) {
  if (!Array.isArray(v) || v.length < 1 || v.length > 32
    || !Array.from(v).every(x => typeof x === 'number' && Number.isFinite(x) && (positive ? x > 0 : x >= 0))
    || !v.some(x => x > 0)) throw new RangeError('Invalid finite probability weights');
}
/** p_i=w_i/W and q_i=v_i/V. Scales multiply raw rational measures BEFORE
 * projection; factors must be finite and nonnegative, up to eight per scale.
 * Thus neither an underflowed KL nor an overflowing intermediate scale product
 * has to destroy a representable final result. No thermodynamic units are
 * inferred: the caller must establish q=pi and supply the intended n,kB,T.
 */
export function relativeEntropyFromWeights(trialWeights: readonly number[], targetWeights: readonly number[],
  scales: Readonly<Record<string, readonly number[]>> = {}) {
  weights(trialWeights, false); weights(targetWeights, true);
  if (trialWeights.length !== targetWeights.length) throw new RangeError('Mismatched microstate counts');
  if (scales === null || typeof scales !== 'object' || Array.isArray(scales)) throw new RangeError('Invalid measure scales');
  const factors = Object.entries(scales).map(([name, values]) => {
    if (!Array.isArray(values) || values.length > 8
      || !Array.from(values).every(x => typeof x === 'number' && Number.isFinite(x) && x >= 0))
      throw new RangeError('Invalid measure scale factors');
    return [name, product(...values)] as const;
  });
  const w = trialWeights.map(x => product(x)), v = targetWeights.map(x => product(x));
  const W = sum(w), V = sum(v), common = multiply(W, V);
  let divergence = zero(), trialEntropy = zero(), targetEntropy = zero(), crossEntropy = zero();
  let sameDistribution = true;
  const coordinates = w.map((wi, i) => {
    const vi = v[i], normalizedTrial = multiply(wi, V), normalizedTarget = multiply(vi, W);
    const gap = difference(normalizedTrial, normalizedTarget), delta = observe(gap, normalizedTarget);
    const isTrialZero = wi.coefficient === BigInt(0), isEqual = gap.coefficient === BigInt(0);
    sameDistribution = sameDistribution && isEqual;
    let term: Fraction, logRatio: Fraction | null;
    if (isTrialZero) { term = { numerator: vi, denominator: V }; logRatio = null; }
    else if (isEqual) { term = zero(); logRatio = zero(); }
    else if (smallRatio(gap, normalizedTarget)) {
      term = { numerator: multiply(multiply(gap, gap), product(bregmanGain(delta.value ?? 0))),
        denominator: multiply(multiply(vi, W), common) };
      logRatio = logarithm(normalizedTrial, normalizedTarget);
    } else {
      logRatio = logarithm(normalizedTrial, normalizedTarget);
      term = { numerator: difference(multiply(normalizedTrial, logRatio.numerator), multiply(gap, logRatio.denominator)),
        denominator: multiply(common, logRatio.denominator) };
    }
    if (!isEqual && term.numerator.coefficient <= BigInt(0)) throw new RangeError('Failed positive divergence evaluation');
    divergence = add(divergence, term);
    const lp = isTrialZero ? null : logarithm(wi, W), lq = logarithm(vi, V);
    const pEntropy = lp === null ? zero() : { numerator: negate(multiply(wi, lp.numerator)), denominator: multiply(W, lp.denominator) };
    const qEntropy = { numerator: negate(multiply(vi, lq.numerator)), denominator: multiply(V, lq.denominator) };
    const cross = { numerator: negate(multiply(wi, lq.numerator)), denominator: multiply(W, lq.denominator) };
    trialEntropy = add(trialEntropy, pEntropy); targetEntropy = add(targetEntropy, qEntropy); crossEntropy = add(crossEntropy, cross);
    return { index: i, trialProbability: observation({ numerator: wi, denominator: W }), targetProbability: observation({ numerator: vi, denominator: V }),
      logTrialProbability: lp === null ? null : observation(lp), logTargetProbability: observation(lq),
      difference: observation({ numerator: gap, denominator: common }), relativeDisplacement: observation({ numerator: gap, denominator: normalizedTarget }),
      probabilityRatio: observation({ numerator: normalizedTrial, denominator: normalizedTarget }), logProbabilityRatio: logRatio === null ? null : observation(logRatio),
      divergenceContribution: observation(term), trialEntropyContribution: observation(pEntropy), targetEntropyContribution: observation(qEntropy), crossEntropyContribution: observation(cross),
      regime: isEqual ? 'equal' as const : isTrialZero ? 'zero-trial-probability' as const : delta.sign > 0 ? 'trial-above-target' as const : 'trial-below-target' as const };
  });
  const measures = (scale = one): RelativeEntropyMeasures => ({ divergence: observation(divergence, scale),
    trialEntropy: observation(trialEntropy, scale), targetEntropy: observation(targetEntropy, scale), crossEntropy: observation(crossEntropy, scale) });
  const scaledMeasures: Record<string, RelativeEntropyMeasures> = Object.fromEntries(factors.map(([name, scale]) => [name, measures(scale)]));
  return { scope: 'exactly-normalized-finite-weight-distributions' as const,
    trialWeights: [...trialWeights], targetWeights: [...targetWeights], sameDistribution, coordinates,
    ...measures(), scaledMeasures };
}
