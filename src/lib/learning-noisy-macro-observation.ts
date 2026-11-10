/** Finite classical observation channel. Preparation and each channel row are
 * normalized separately as exact rationals of supplied binary64 inputs. Logs
 * and information measures approximate transcendental values; no dynamics,
 * measurement heat or quantum back-action is inferred from the channel.
 */
import { binaryProduct, binaryRatioObservation, type BinaryObservation } from './learning-binary-products';
import { rationalGreatestCommonDivisor as gcd } from './learning-rational-gcd';
import { roundInformationTerm, informationArithmeticBits } from './learning-information-arithmetic';
interface Rational { n: bigint; d: bigint }
const Z = BigInt(0), O = BigInt(1), zero = (): Rational => ({ n: Z, d: O });
function fraction(n: bigint, d: bigint): Rational {
  if (d === Z) throw new RangeError('Zero rational denominator');
  if (n === Z) return zero();
  if (d < Z) { n = -n; d = -d; }
  const common = gcd(n, d); return { n: n / common, d: d / common };
}
function input(value: number): Rational {
  const b = binaryProduct(value);
  return b.exponent >= 0 ? fraction(b.coefficient << BigInt(b.exponent), O)
    : fraction(b.coefficient, O << BigInt(-b.exponent));
}
function add(a: Rational, b: Rational) {
  if (a.n === Z) return b; if (b.n === Z) return a;
  const common = gcd(a.d, b.d);
  const n = a.n * (b.d / common) + b.n * (a.d / common);
  if (n === Z) return zero();
  // For reduced inputs only factors in the shared denominator can cancel.
  // Avoid a full gcd of a very large numerator and the product denominator.
  const cancel = gcd(n, common);
  return { n: n / cancel, d: (a.d / common) * (b.d / cancel) };
}
const negative = (a: Rational): Rational => ({ n: -a.n, d: a.d });
const difference = (a: Rational, b: Rational) => add(a, negative(b));
function multiply(a: Rational, b: Rational): Rational {
  if (a.n === Z || b.n === Z) return zero();
  const crossA = gcd(a.n, b.d), crossB = gcd(b.n, a.d);
  return { n: (a.n / crossA) * (b.n / crossB), d: (a.d / crossB) * (b.d / crossA) };
}
const divide = (a: Rational, b: Rational) => {
  if (b.n === Z) throw new RangeError('Zero rational divisor');
  return multiply(a, b.n > Z ? { n: b.d, d: b.n } : { n: -b.d, d: -b.n });
};
const sum = (values: Rational[]) => values.reduce(add, zero());
const observe = (a: Rational): BinaryObservation => binaryRatioObservation({ coefficient: a.n, exponent: 0 }, { coefficient: a.d, exponent: 0 });
const equal = (a: Rational, b: Rational) => a.n === b.n && a.d === b.d;
const isOne = (a: Rational) => a.n === a.d;
/** ln(a), with its small displacement retained as a rational before projection. */
function logarithm(a: Rational): Rational {
  const gap = a.n - a.d;
  if (gap === Z) return zero();
  // gcd(a.n-a.d,a.d)=1 because a is already reduced.
  const delta = { n: gap, d: a.d }, d = observe(delta);
  if ((gap < Z ? -gap : gap) * BigInt(8) <= a.d) {
    const t = d.value ?? 0;
    return multiply(delta, input(t === 0 ? 1 : Math.log1p(t) / t));
  }
  const r = observe(a);
  return input(d.value !== null && d.value > -.75 && d.value < 1 ? Math.log1p(d.value)
    : r.value === null || r.value < 2 ** -1022 ? r.logMagnitude! : Math.log(r.value));
}
// Round approximate information terms before summing them exactly as dyadics.
const informationTerm = (a: Rational) => roundInformationTerm(a.n, a.d);
const informationSum = (values: Rational[]) => sum(values.map(informationTerm));
function entropy(probabilities: Rational[]) {
  return informationSum(probabilities.map(p => {
    if (p.n === Z) return zero();
    const log = logarithm(p);
    // This term is immediately rounded at 192 bits, so avoid reducing the
    // product's unrelated large denominators first. Probabilities stay reduced.
    return { n: -p.n * log.n, d: p.d * log.d };
  }));
}
function bregmanGain(t: number) {
  let gain = 0;
  for (let k = 32; k >= 0; k--) gain = 1 / ((k + 1) * (k + 2)) - t * gain;
  return gain;
}
/** Normalized finite laws with source support contained in target support.
 * Direct positive contributions preserve tiny MI; no entropy subtraction or
 * negative-value clamp. Series truncation at |t|<=1/8 is below 2e-33;
 * binary64 evaluation and approximate logarithms still contribute error.
 */
function divergence(p: Rational[], q: Rational[]) {
  let result = zero();
  for (let i = 0; i < p.length; i++) {
    if (q[i].n === Z) { if (p[i].n !== Z) throw new RangeError('Unexpected mutual-information support violation'); continue; }
    if (p[i].n === Z) { result = add(result, informationTerm(q[i])); continue; }
    const gap = difference(p[i], q[i]);
    if (gap.n === Z) continue;
    const delta = divide(gap, q[i]);
    const small = (delta.n < Z ? -delta.n : delta.n) * BigInt(8) <= delta.d;
    let term: Rational;
    if (small) {
      const gain = input(bregmanGain(observe(delta).value ?? 0));
      term = { n: gap.n * gap.n * q[i].d * gain.n, d: gap.d * gap.d * q[i].n * gain.d };
    } else {
      const log = logarithm(divide(p[i], q[i]));
      term = { n: p[i].n * log.n * gap.d - gap.n * p[i].d * log.d,
        d: p[i].d * log.d * gap.d };
    }
    if (term.n <= Z) throw new RangeError('Failed positive mutual-information evaluation');
    result = add(result, informationTerm(term));
  }
  return result;
}
function weights(values: readonly number[]) {
  if (!Array.isArray(values) || values.length < 1 || values.length > 16
    || !Array.from(values).every(x => typeof x === 'number' && Number.isFinite(x) && x >= 0)
    || !values.some(x => x > 0)) throw new RangeError('Invalid finite channel weights');
  const raw = values.map(input), total = sum(raw);
  return raw.map(x => divide(x, total));
}
export function noisyMacroObservationModel(preparationWeights: readonly number[], channelWeights: readonly (readonly number[])[]) {
  const p = weights(preparationWeights);
  if (!Array.isArray(channelWeights) || channelWeights.length !== p.length
    || !Array.from(channelWeights).every(row => Array.isArray(row))) throw new RangeError('Invalid observation channel rows');
  const channel = channelWeights.map(weights), outputs = channel[0].length;
  if (!channel.every(row => row.length === outputs)) throw new RangeError('Mismatched observation channel widths');
  const joint = channel.map((row, i) => row.map(k => multiply(p[i], k)));
  const marginal = Array.from({ length: outputs }, (_, j) => sum(joint.map(row => row[j])));
  const posterior = marginal.map((r, j) => r.n === Z ? null : joint.map(row => divide(row[j], r)));
  const rowEntropy = channel.map(entropy), posteriorEntropy = posterior.map(row => row === null ? null : entropy(row));
  const noiseContributions = rowEntropy.map((h, i) => multiply(p[i], h));
  const hiddenContributions = posteriorEntropy.map((h, j) => h === null ? zero() : multiply(marginal[j], h));
  const independent = joint.map((row, i) => row.map((_, j) => multiply(p[i], marginal[j])));
  const flattened = joint.flat(), target = independent.flat();
  const normalization = { input: isOne(sum(p)), channelRows: channel.every(row => isOne(sum(row))),
    joint: isOne(sum(flattened)), output: isOne(sum(marginal)),
    posteriors: posterior.every(row => row === null || isOne(sum(row))) };
  if (!Object.values(normalization).every(Boolean)) throw new RangeError('Failed exact channel normalization');
  return { scope: 'finite-classical-noisy-observation' as const,
    parameters: { preparationWeights: [...preparationWeights], channelWeights: channelWeights.map(row => [...row]) },
    inputProbabilities: p.map(observe), channelProbabilities: channel.map(row => row.map(observe)),
    jointProbabilities: joint.map(row => row.map(observe)), outputProbabilities: marginal.map(observe),
    posteriorProbabilities: posterior.map(row => row === null ? null : row.map(observe)),
    rowEntropies: rowEntropy.map(observe), posteriorEntropies: posteriorEntropy.map(h => h === null ? null : observe(h)),
    noiseEntropyContributions: noiseContributions.map(observe), hiddenEntropyContributions: hiddenContributions.map(observe),
    inputEntropy: observe(entropy(p)), outputEntropy: observe(entropy(marginal)), jointEntropy: observe(entropy(flattened)),
    noiseConditionalEntropy: observe(informationSum(noiseContributions)), hiddenConditionalEntropy: observe(informationSum(hiddenContributions)),
    mutualInformation: observe(divergence(flattened, target)),
    independentExactly: flattened.every((x, i) => equal(x, target[i])),
    deterministicOnPreparationSupport: channel.every((row, i) => p[i].n === Z || row.filter(k => k.n !== Z).length === 1),
    normalization,
    informationArithmeticBits,
    approximationContract: 'Exact input, row, joint and posterior rational normalization before projection. Approximate nonnegative entropy/KL terms use 192 significant bits with unrestricted exponents before exact dyadic summation; each term rounding has relative error at most 2^-192. Transcendental error remains. No measurement thermodynamics is specified.' };
}
