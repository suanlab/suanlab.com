/** Trial finite-state thermodynamic functionals. Probability normalization and
 * energy sums use exact binary inputs; logarithms remain approximations. No
 * rounded canonical weights are substituted into an asserted Helmholtz–KL gap.
 */
import { canonicalFiniteSpectrum, canonicalBoltzmannConstant as kB } from './learning-canonical-spectrum';
import { binaryProduct as product, binarySum as sum, binaryMultiply as multiply,
  binaryNegate as negate, binaryRatioObservation as ratio,
  type BinaryValue, type BinaryObservation } from './learning-binary-products';
export interface TrialFreeEnergyObservation extends BinaryObservation {
  zeroStatus: 'structural-zero' | 'unresolved-transcendental-cancellation' | 'not-zero-approximation';
}
const subtract = (a: BinaryValue, b: BinaryValue) => sum([a, negate(b)]);
/** Numerator of p ln(W/w), with common denominator W. Close to p=1,
 * w ln(1+(W-w)/w)=(W-w) log1p(t)/t retains subnormal tails.
 */
function entropyNumerator(w: BinaryValue, W: BinaryValue): BinaryValue {
  if (w.coefficient === BigInt(0)) return product(0);
  const gap = subtract(W, w);
  if (gap.coefficient === BigInt(0)) return product(0);
  if (subtract(multiply(gap, product(8)), w).coefficient <= BigInt(0)) {
    const t = ratio(gap, w).value ?? 0;
    return multiply(gap, product(t === 0 ? 1 : Math.log1p(t) / t));
  }
  const p = ratio(w, W);
  // Using the logarithm of a rounded subnormal probability loses information.
  const log = p.value === null || p.value < 2 ** -1022 ? p.logMagnitude! : Math.log(p.value);
  return multiply(w, product(-log));
}
/** Copies are independent, identically distributed, with additive energies.
 * Zero copies is allowed; correlated preparations require a joint distribution.
 */
export function finiteTrialThermodynamics(energies: readonly number[], temperature: number,
  trialWeights: readonly number[], copies = 1) {
  const canonical = canonicalFiniteSpectrum(energies, temperature);
  if (!Array.isArray(trialWeights) || trialWeights.length !== energies.length
    || !Array.from(trialWeights).every(w => typeof w === 'number' && Number.isFinite(w) && w >= 0)
    || !trialWeights.some(w => w > 0)) throw new RangeError('Expected nonnegative finite weights with positive sum');
  if (typeof copies !== 'number' || !Number.isFinite(copies) || !Number.isInteger(copies) || copies < 0)
    throw new RangeError('Expected finite nonnegative integer independent-copy count');
  const weights = trialWeights.map(w => product(w)), W = sum(weights);
  const thermal = product(kB, temperature), scale = product(copies);
  const energyTerms = weights.map((w, i) => multiply(w, product(energies[i])));
  const entropyTerms = weights.map(w => entropyNumerator(w, W));
  const energyNumerator = sum(energyTerms), H = sum(entropyTerms);
  const freeNumerator = subtract(energyNumerator, multiply(H, thermal));
  const free = (factor: BinaryValue): TrialFreeEnergyObservation => {
    const observation = ratio(multiply(freeNumerator, factor), W);
    const structural = factor.coefficient === BigInt(0)
      || energyNumerator.coefficient === BigInt(0) && H.coefficient === BigInt(0);
    return { ...observation, zeroStatus: structural ? 'structural-zero'
      : observation.sign === 0 ? 'unresolved-transcendental-cancellation' : 'not-zero-approximation' };
  };
  const measures = (factor: BinaryValue) => ({
    meanEnergy: ratio(multiply(energyNumerator, factor), W),
    informationEntropy: ratio(multiply(H, factor), W),
    entropy: ratio(multiply(multiply(H, product(kB)), factor), W),
    temperatureTimesEntropy: ratio(multiply(multiply(H, thermal), factor), W),
    helmholtz: free(factor),
  });
  return { scope: 'finite-state-trial-functionals-with-fixed-canonical-target' as const,
    energies: [...energies], temperature, trialWeights: [...trialWeights], copies, canonical,
    perCopy: measures(product(1)), total: measures(scale),
    states: weights.map((w, index) => ({ index, probability: ratio(w, W),
      meanEnergyContribution: ratio(energyTerms[index], W), informationEntropyContribution: ratio(entropyTerms[index], W) })),
    units: { meanEnergy: 'J', informationEntropy: 'nat', entropy: 'J/K', temperatureTimesEntropy: 'J', helmholtz: 'J' } as const,
    approximationContract: 'Exact input normalization and mean energies; approximate entropy logarithms and anchored Helmholtz values. No equilibrium free-energy gap certificate yet.' as const };
}
