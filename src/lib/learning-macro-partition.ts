/** Finite weighted microstates, deterministic observations and an explicitly
 * chosen uniform-within-group reconstruction. No dynamics or heat is implied.
 * Input binary64 weights, sums and ratios stay exact until observation;
 * logarithms and their derived measures are numerical approximations.
 */
import { binaryProduct as product, binarySum as sum, binaryNegate as negate,
  binaryMultiply as multiply, binaryRatioObservation as observe,
  type BinaryValue, type BinaryObservation } from './learning-binary-products';

interface Fraction { numerator: BinaryValue; denominator: BinaryValue }
type RawDivergence = Fraction | 'infinite';
export type MacroPartitionDivergence = BinaryObservation & { status: 'finite' | 'infinite' };
const one = product(1), zero = (): Fraction => ({ numerator: product(0), denominator: one });
const difference = (a: BinaryValue, b: BinaryValue) => sum([a, negate(b)]);
const isZero = (a: BinaryValue) => a.coefficient === BigInt(0);
function gcd(a: bigint, b: bigint) {
  if (a < BigInt(0)) a = -a;
  while (b !== BigInt(0)) { const next = a % b; a = b; b = next; }
  return a;
}
function reduce(f: Fraction): Fraction {
  if (isZero(f.numerator)) return zero();
  const common = gcd(f.numerator.coefficient, f.denominator.coefficient);
  let n = f.numerator.coefficient / common, d = f.denominator.coefficient / common;
  let exponent = f.numerator.exponent - f.denominator.exponent;
  while (n % BigInt(2) === BigInt(0)) { n /= BigInt(2); exponent++; }
  while (d % BigInt(2) === BigInt(0)) { d /= BigInt(2); exponent--; }
  return { numerator: { coefficient: n, exponent }, denominator: { coefficient: d, exponent: 0 } };
}
function add(a: Fraction, b: Fraction): Fraction {
  if (isZero(a.numerator)) return reduce(b);
  if (isZero(b.numerator)) return a;
  return reduce({ numerator: sum([multiply(a.numerator, b.denominator), multiply(b.numerator, a.denominator)]),
    denominator: multiply(a.denominator, b.denominator) });
}
const scaled = (a: Fraction, n: BinaryValue, d: BinaryValue): Fraction =>
  reduce({ numerator: multiply(a.numerator, n), denominator: multiply(a.denominator, d) });
const observation = (a: Fraction) => observe(a.numerator, a.denominator);
function smallRatio(n: BinaryValue, d: BinaryValue) {
  const absolute = n.coefficient < BigInt(0) ? negate(n) : n;
  return difference(multiply(absolute, product(8)), d).coefficient <= BigInt(0);
}
/** Preserve ln(1+delta) before projection, even when delta itself underflows. */
function logarithm(n: BinaryValue, d: BinaryValue): Fraction {
  const gap = difference(n, d);
  if (isZero(gap)) return zero();
  const delta = observe(gap, d);
  if (smallRatio(gap, d)) {
    const value = delta.value ?? 0;
    return { numerator: multiply(gap, product(value === 0 ? 1 : Math.log1p(value) / value)), denominator: d };
  }
  const ratio = observe(n, d);
  const log = delta.value !== null && delta.value > -.75 && delta.value < 1
    ? Math.log1p(delta.value) : ratio.value === null || ratio.value < 2 ** -1022
      ? ratio.logMagnitude! : Math.log(ratio.value);
  return { numerator: product(log), denominator: one };
}
/** Stable positive Bregman gain, as in the existing finite-weight KL kernel.
 * Series truncation for |t|<=1/8 is below 2e-33; binary64 error remains.
 */
function bregmanGain(t: number) {
  let gain = 0;
  for (let k = 32; k >= 0; k--) gain = 1 / ((k + 1) * (k + 2)) - t * gain;
  return gain;
}
function entropy(weights: BinaryValue[], total: BinaryValue): Fraction {
  let result = zero();
  for (const w of weights) if (!isZero(w)) {
    const log = logarithm(w, total);
    result = add(result, { numerator: negate(multiply(w, log.numerator)), denominator: multiply(total, log.denominator) });
  }
  return result;
}
/** Sum q_i[(p_i/q_i)ln(p_i/q_i)-(p_i/q_i)+1]. This nonnegative
 * representation avoids subtracting nearly equal entropy observations.
 * Zero targets and exact aggregate weights are supported, unlike projecting
 * group weights into the number-only positive-target KL teaching interface.
 */
function divergence(w: BinaryValue[], v: BinaryValue[]): RawDivergence {
  const W = sum(w), V = sum(v), common = multiply(W, V);
  let result = zero();
  for (let i = 0; i < w.length; i++) {
    const wi = w[i], vi = v[i];
    if (isZero(vi)) { if (!isZero(wi)) return 'infinite'; continue; }
    if (isZero(wi)) { result = add(result, { numerator: vi, denominator: V }); continue; }
    const p = multiply(wi, V), q = multiply(vi, W), gap = difference(p, q);
    if (isZero(gap)) continue;
    let term: Fraction;
    if (smallRatio(gap, q)) {
      const gain = bregmanGain(observe(gap, q).value ?? 0);
      term = { numerator: multiply(multiply(gap, gap), product(gain)), denominator: multiply(q, common) };
    } else {
      const log = logarithm(p, q);
      term = { numerator: difference(multiply(p, log.numerator), multiply(gap, log.denominator)),
        denominator: multiply(common, log.denominator) };
    }
    if (term.numerator.coefficient <= BigInt(0)) throw new RangeError('Failed positive partition divergence');
    result = add(result, term);
  }
  return result;
}
function divergenceObservation(raw: RawDivergence): MacroPartitionDivergence {
  return raw === 'infinite' ? { status: 'infinite', value: null, sign: 1, logMagnitude: null }
    : { status: 'finite', ...observation(raw) };
}
function addDivergence(a: RawDivergence, b: RawDivergence): RawDivergence {
  return a === 'infinite' || b === 'infinite' ? 'infinite' : add(a, b);
}
function weights(values: readonly number[]) {
  if (!Array.isArray(values) || values.length < 1 || values.length > 32
    || !Array.from(values).every(x => typeof x === 'number' && Number.isFinite(x) && x >= 0)
    || !values.some(x => x > 0)) throw new RangeError('Invalid finite microstate weights');
  return values.map(x => product(x));
}

/** Group labels include declared empty groups. Positive probabilities outside
 * the display range retain sign/log evidence; they are never support zeros.
 * Null conditionals mean undefined, while KL status=infinite is mathematical.
 */
export function macroPartitionModel(trialWeights: readonly number[], targetWeights: readonly number[],
  mapping: readonly number[], groupCount: number) {
  const w = weights(trialWeights), v = weights(targetWeights);
  if (w.length !== v.length || !Array.isArray(mapping) || mapping.length !== w.length
    || !Number.isInteger(groupCount) || groupCount < 1 || groupCount > 32
    || !Array.from(mapping).every(g => Number.isInteger(g) && g >= 0 && g < groupCount))
    throw new RangeError('Invalid finite macrostate partition');
  const W = sum(w), V = sum(v);
  const members = Array.from({ length: groupCount }, (_, g) => mapping.flatMap((label, i) => label === g ? [i] : []));
  const groupW = members.map(ids => sum(ids.map(i => w[i]))), groupV = members.map(ids => sum(ids.map(i => v[i])));
  let conditional = zero(), gain = zero(), reconstructionWithin = zero();
  let hidden: RawDivergence | null = zero();
  const groups = members.map((ids, group) => {
    const wg = groupW[group], vg = groupV[group], ws = ids.map(i => w[i]), vs = ids.map(i => v[i]);
    const conditionalEntropy = isZero(wg) ? null : entropy(ws, wg);
    const entropyContribution = conditionalEntropy === null ? zero() : scaled(conditionalEntropy, wg, W);
    conditional = add(conditional, entropyContribution);
    let gainContribution = zero();
    if (!isZero(wg)) {
      const rawGain = divergence(ws, ids.map(() => one));
      if (rawGain === 'infinite') throw new RangeError('Uniform reconstruction must have full within-group support');
      gainContribution = scaled(rawGain, wg, W);
      reconstructionWithin = add(reconstructionWithin, { numerator: multiply(wg, product(Math.log(ids.length))), denominator: W });
    }
    gain = add(gain, gainContribution);
    let inside: RawDivergence | null = zero();
    if (!isZero(wg)) {
      if (isZero(vg)) inside = null;
      else {
        const raw = divergence(ws, vs);
        inside = raw === 'infinite' ? 'infinite' : scaled(raw, wg, W);
      }
    }
    hidden = inside === null || hidden === null ? null : addDivergence(hidden, inside);
    return { group, members: [...ids], count: ids.length,
      probability: observe(wg, W), targetProbability: observe(vg, V),
      conditionalProbabilities: isZero(wg) ? null : ws.map(x => observe(x, wg)),
      targetConditionalProbabilities: isZero(vg) ? null : vs.map(x => observe(x, vg)),
      conditionalEntropy: conditionalEntropy === null ? null : observation(conditionalEntropy),
      conditionalEntropyContribution: observation(entropyContribution), reconstructionGainContribution: observation(gainContribution),
      conditionalDivergenceContribution: inside === null ? null : divergenceObservation(inside) };
  });
  const macroEntropy = entropy(groupW, W);
  return { scope: 'finite-weighted-deterministic-macro-observation' as const,
    parameters: { trialWeights: [...trialWeights], targetWeights: [...targetWeights], mapping: [...mapping], groupCount },
    reconstructionRule: 'uniform-within-each-declared-nonempty-group' as const,
    probabilities: w.map(x => observe(x, W)), targetProbabilities: v.map(x => observe(x, V)),
    reconstructedProbabilities: mapping.map(g => observe(groupW[g], multiply(W, product(members[g].length)))), groups,
    microEntropy: observation(entropy(w, W)), macroEntropy: observation(macroEntropy),
    conditionalEntropy: observation(conditional), reconstructedEntropy: observation(add(macroEntropy, reconstructionWithin)),
    reconstructionGain: observation(gain), microDivergence: divergenceObservation(divergence(w, v)),
    macroDivergence: divergenceObservation(divergence(groupW, groupV)),
    hiddenDivergence: hidden === null ? null : divergenceObservation(hidden),
    normalizationContract: 'Exact binary-rational normalization and grouping before projection; logarithms and entropy/KL approximations are not universally correctly rounded.' };
}
