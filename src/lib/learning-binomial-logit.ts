/** Internal finite Bernoulli product kernel. The supplied log odds are a ratio
 * of exact binary products; probabilities involve approximate exponentials.
 * Never use a rounded zero minority probability as a support constraint.
 */
import { binaryProduct as product, binaryMultiply as multiply, binarySum as sum,
  binaryNegate as negate, binaryObservation as observe, binaryRatioObservation as ratio,
  type BinaryValue } from './learning-binary-products';
import { binaryMultiplicityRow } from './learning-binary-macrostates';
import type { SpectrumQuantity } from './learning-gapped-spectrum';
const Z = BigInt(0);
export const binomialOne = product(1);
export const binomialZeroQuantity = (): SpectrumQuantity => ({ value: 0, sign: 0, logMagnitude: null,
  logNegativeLogMagnitude: null, logMagnitudeStatus: 'not-applicable', logNegativeLogMagnitudeStatus: 'not-applicable' });
/** Logarithms of numerical SI values are approximate, not equality proofs. */
export function binomialRawQuantity(n: BinaryValue, d = binomialOne): SpectrumQuantity {
  const q = ratio(n, d);
  if (q.sign === 0) return binomialZeroQuantity();
  const gap = sum([n, negate(d)]);
  if (gap.coefficient === Z) return { ...q, logMagnitude: 0, logMagnitudeStatus: 'finite',
    logNegativeLogMagnitude: null, logNegativeLogMagnitudeStatus: 'not-applicable' };
  const absGap = gap.coefficient < Z ? negate(gap) : gap;
  if (sum([multiply(absGap, product(8)), negate(d)]).coefficient <= Z) {
    const delta = ratio(gap, d), log = delta.value === null ? null : Math.log1p(delta.value);
    return { ...q, logMagnitude: log, logMagnitudeStatus: log === null ? 'outside-number-range' : 'finite',
      logNegativeLogMagnitude: delta.sign < 0 ? log === null ? delta.logMagnitude : Math.log(-log) : null,
      logNegativeLogMagnitudeStatus: delta.sign < 0 ? 'finite' : 'not-applicable' };
  }
  return { ...q, logMagnitudeStatus: 'finite',
    logNegativeLogMagnitude: q.logMagnitude! < 0 ? Math.log(-q.logMagnitude!) : null,
    logNegativeLogMagnitudeStatus: q.logMagnitude! < 0 ? 'finite' : 'not-applicable' };
}
/** exp(L), preserving a nonzero L even outside its own linear range. */
export function binomialExponential(n: BinaryValue, d = binomialOne): SpectrumQuantity {
  const log = ratio(n, d);
  const displayed = log.value === null ? log.logMagnitude! < 0 ? 1 : null : Math.exp(log.value);
  return { value: displayed === 0 || !Number.isFinite(displayed) ? null : displayed, sign: 1,
    logMagnitude: log.value, logMagnitudeStatus: log.value === null ? 'outside-number-range' : 'finite',
    logNegativeLogMagnitude: log.sign < 0 ? log.logMagnitude : null,
    logNegativeLogMagnitudeStatus: log.sign < 0 ? 'finite' : 'not-applicable' };
}
/** exp(-loss), when ln(loss) is very negative. A displayed 1 is not an exact
 * point mass; first and second logarithm availability are explicit. */
function nearOne(logLoss: BinaryValue, d: BinaryValue): SpectrumQuantity {
  const lossLog = ratio(logLoss, d);
  return { value: 1, sign: 1, logMagnitude: null, logMagnitudeStatus: 'outside-number-range',
    logNegativeLogMagnitude: lossLog.value,
    logNegativeLogMagnitudeStatus: lossLog.value === null ? 'outside-number-range' : 'finite' };
}
export function binomialLogit(sites: number, logitNumerator: BinaryValue, logitDenominator: BinaryValue) {
  if (!Number.isInteger(sites) || sites < 0 || sites > 4096) throw new RangeError('Invalid binomial site count');
  if (logitDenominator.coefficient <= Z) throw new RangeError('Expected positive logit denominator');
  const magnitude = logitNumerator.coefficient < Z ? negate(logitNumerator) : logitNumerator;
  const u = ratio(magnitude, logitDenominator), positive = logitNumerator.coefficient > Z;
  const counts = binaryMultiplicityRow(sites), n = product(sites);
  const extreme = u.value === null && u.sign > 0 && u.logMagnitude! > 0 || u.value !== null && u.value > 4096;
  const expression = (constant: number, factor = 1) => sum([
    multiply(logitDenominator, product(constant)), negate(multiply(magnitude, product(factor)))
  ]);
  // For |logit|>4096, omitted log(1+exp(-|logit|)) corrections are smaller
  // than exp(-4096). No accepted physical prefactor can rescue them linearly.
  if (extreme) {
    const minority = binomialExponential(negate(magnitude), logitDenominator);
    const majority = nearOne(negate(magnitude), logitDenominator);
    const distribution = counts.map((count, k) => {
      const j = positive ? sites - k : k;
      return { excitations: k, multiplicity: count.toString(),
        probability: sites === 0 ? binomialExponential(product(0)) : j === 0
          ? nearOne(expression(Math.log(sites)), logitDenominator)
          : binomialExponential(expression(observe({ coefficient: count, exponent: 0 }).logMagnitude!, j), logitDenominator) };
    });
    return { sites, positive, magnitude, denominator: logitDenominator, scaledMagnitude: u,
      minority, majority, distribution, raw: null,
      oneProbability: positive ? majority : minority,
      zeroProbability: positive ? minority : majority,
      logPartitionCorrection: sites === 0 ? binomialZeroQuantity() : binomialExponential(expression(Math.log(sites)), logitDenominator),
      tailLog: expression, extreme: true as const };
  }
  const x = u.value ?? 0;
  // Eight moderate exponentials give a binary product with unrestricted
  // exponent, retaining exp(-4096) before multiplication by physical scales.
  const root = Math.exp(-x / 8), w = product(root, root, root, root, root, root, root, root);
  const W = sum([binomialOne, w]);
  const correction = x > 37 ? w : product(Math.log1p(Math.exp(-x)));
  const correctionNumerator = multiply(correction, logitDenominator);
  const minority = { ...binomialExponential(negate(sum([magnitude, correctionNumerator])), logitDenominator), value: ratio(w, W).value };
  const majority = magnitude.coefficient === Z ? minority
    : { ...binomialExponential(negate(correction)), value: ratio(binomialOne, W).value };
  const distribution = counts.map((count, k) => {
    const j = positive ? sites - k : k;
    const numerator = sum([multiply(logitDenominator, product(observe({ coefficient: count, exponent: 0 }).logMagnitude!)),
      negate(multiply(magnitude, product(j))), negate(multiply(correctionNumerator, n))]);
    return { excitations: k, multiplicity: count.toString(), probability: binomialExponential(numerator, logitDenominator) };
  });
  return { sites, positive, magnitude, denominator: logitDenominator, scaledMagnitude: u,
    minority, majority, distribution, raw: { w, W, correction },
    oneProbability: positive ? majority : minority,
    zeroProbability: positive ? minority : majority,
    logPartitionCorrection: binomialRawQuantity(multiply(n, correction)),
    tailLog: expression, extreme: false as const };
}
