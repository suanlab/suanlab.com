/** Canonical independent binary sites with a fixed nonnegative SI energy gap.
 * Negative T describes population inversion of this bounded spectrum only.
 * T=0 is the positive-side equilibrium limit, not finite-time cooling.
 */
import { binaryProduct as product, binaryMultiply as multiply, binarySum as sum,
  binaryNegate as negate } from './learning-binary-products';
import { canonicalBoltzmannConstant as kB } from './learning-canonical-spectrum';
import { binomialLogit, binomialExponential as exponential, binomialRawQuantity as rawQuantity,
  binomialZeroQuantity as zero, binomialOne as one } from './learning-binomial-logit';
export interface CanonicalBinaryParameters { sites: number; energyGap: number; temperature: number }
export function canonicalBinaryModel(parameters: CanonicalBinaryParameters) {
  const { sites: n, energyGap: epsilon, temperature: T } = parameters;
  if (!Number.isInteger(n) || n < 0 || n > 4096) throw new RangeError('Expected 0 to 4096 sites');
  if (!Number.isFinite(epsilon) || epsilon < 0 || !Number.isFinite(T))
    throw new RangeError('Expected finite nonnegative gap and finite temperature');
  const coldLimit = T === 0 && epsilon > 0, negativeTemperature = T < 0;
  const thermal = coldLimit || epsilon === 0 ? one : product(kB, Math.abs(T));
  const gap = product(epsilon), kernel = binomialLogit(n, coldLimit ? product(0) : negativeTemperature ? gap : negate(gap), thermal);
  const scaled = negativeTemperature ? { ...kernel.scaledMagnitude, sign: kernel.scaledMagnitude.sign === 0 ? 0 : -kernel.scaledMagnitude.sign,
    value: kernel.scaledMagnitude.value === null ? null : kernel.scaledMagnitude.value === 0 ? 0 : -kernel.scaledMagnitude.value } : kernel.scaledMagnitude;
  const common = { scope: 'canonical-distinguishable-independent-binary-sites' as const,
    parameters: { ...parameters }, boltzmannConstant: kB,
    mode: T === 0 ? 'zero-temperature-limit' as const : negativeTemperature ? 'bounded-negative-temperature' as const : 'positive-temperature' as const,
    spectrum: epsilon === 0 ? 'exact-zero-gap' as const : 'positive-gap' as const,
    totalMicrostates: (BigInt(1) << BigInt(n)).toString(),
    approximationContract: 'Exact binary input products and state multiplicities; approximate exponentials and logarithms. Finite-temperature probabilities stay positive outside display ranges. Nested logs diagnose tails, not certified departures from unit-valued moments. Examples are independent preparations, not physical time.' as const };
  if (coldLimit) return { ...common, scaledEnergyGap: null,
    oneProbability: zero(), zeroProbability: rawQuantity(one),
    distribution: kernel.distribution.map(row => ({ ...row, probability: row.excitations === 0 ? rawQuantity(one) : zero() })),
    meanExcitations: zero(), excitationVariance: zero(), meanEnergy: zero(), energyVariance: zero(),
    entropy: zero(), entropyInKB: zero(), heatCapacity: zero(), heatCapacityInKB: zero(), logPartition: zero(),
    relativeCountFluctuation: null, relativeEnergyFluctuation: null };
  const relative = n === 0 ? null : exponential(sum([negativeTemperature ? negate(gap) : gap, negate(multiply(thermal, product(Math.log(n))))]), multiply(product(2), thermal));
  if (n === 0) return { ...common, scaledEnergyGap: scaled,
    oneProbability: kernel.oneProbability, zeroProbability: kernel.zeroProbability, distribution: kernel.distribution,
    meanExcitations: zero(), excitationVariance: zero(), meanEnergy: zero(), energyVariance: zero(),
    entropy: zero(), entropyInKB: zero(), heatCapacity: zero(), heatCapacityInKB: zero(), logPartition: zero(),
    relativeCountFluctuation: null, relativeEnergyFluctuation: null };
  if (kernel.raw === null) {
    const q = (constant: number) => exponential(kernel.tailLog(constant), thermal);
    const logN = Math.log(n), logE = Math.log(epsilon), logKB = Math.log(kB), logU = kernel.scaledMagnitude.logMagnitude!;
    const entropyFactor = kernel.scaledMagnitude.value === null ? logU : Math.log1p(kernel.scaledMagnitude.value);
    return { ...common, scaledEnergyGap: scaled,
      oneProbability: kernel.oneProbability, zeroProbability: kernel.zeroProbability, distribution: kernel.distribution,
      meanExcitations: negativeTemperature ? rawQuantity(product(n)) : q(logN), excitationVariance: q(logN),
      meanEnergy: negativeTemperature ? rawQuantity(product(n, epsilon)) : q(logN + logE), energyVariance: q(logN + 2 * logE),
      entropy: q(logN + logKB + entropyFactor), entropyInKB: q(logN + entropyFactor),
      heatCapacity: q(logN + logKB + 2 * logU), heatCapacityInKB: q(logN + 2 * logU),
      logPartition: negativeTemperature ? rawQuantity(product(n, epsilon), thermal) : kernel.logPartitionCorrection, relativeCountFluctuation: relative, relativeEnergyFluctuation: relative };
  }
  const { w, W, correction } = kernel.raw, W2 = multiply(W, W), nRaw = product(n);
  const mean = multiply(nRaw, negativeTemperature ? one : w), variance = multiply(nRaw, w);
  const entropyNumerator = multiply(nRaw, sum([multiply(multiply(correction, thermal), W), multiply(gap, w)]));
  const entropyDenominator = multiply(thermal, W);
  const capacityNumerator = multiply(multiply(multiply(nRaw, gap), gap), w);
  const capacityDenominator = multiply(multiply(thermal, thermal), W2);
  return { ...common, scaledEnergyGap: scaled,
    oneProbability: kernel.oneProbability, zeroProbability: kernel.zeroProbability, distribution: kernel.distribution,
    meanExcitations: rawQuantity(mean, W), excitationVariance: rawQuantity(variance, W2),
    meanEnergy: rawQuantity(multiply(mean, gap), W), energyVariance: rawQuantity(multiply(multiply(variance, gap), gap), W2),
    entropy: rawQuantity(multiply(entropyNumerator, product(kB)), entropyDenominator), entropyInKB: rawQuantity(entropyNumerator, entropyDenominator),
    heatCapacity: rawQuantity(multiply(capacityNumerator, product(kB)), capacityDenominator), heatCapacityInKB: rawQuantity(capacityNumerator, capacityDenominator),
    logPartition: negativeTemperature ? rawQuantity(multiply(nRaw, sum([multiply(correction, thermal), gap])), thermal) : kernel.logPartitionCorrection, relativeCountFluctuation: relative,
    relativeEnergyFluctuation: epsilon === 0 ? null : relative };
}
