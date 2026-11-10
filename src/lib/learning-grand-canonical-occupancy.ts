/** Classical labeled exclusion sites, each empty or singly occupied.
 * Particle number varies; this is not the continuum ideal-gas state count.
 * T=0 denotes the positive-side grand-canonical equilibrium limit.
 */
import { binaryProduct as product, binaryMultiply as multiply, binarySum as sum,
  binaryNegate as negate, binaryObservation as observe, binaryRatioObservation as ratio } from './learning-binary-products';
import { canonicalBoltzmannConstant as kB } from './learning-canonical-spectrum';
import { binomialLogit, binomialExponential as exponential, binomialRawQuantity as rawQuantity,
  binomialZeroQuantity as zero, binomialOne as one } from './learning-binomial-logit';
import type { SpectrumQuantity } from './learning-gapped-spectrum';
export interface GrandOccupancyParameters { sites: number; energyGap: number; chemicalPotential: number; temperature: number }
const Z = BigInt(0);
const signed = (q: SpectrumQuantity, sign: number): SpectrumQuantity => q.sign === 0 ? q : {
  ...q, sign, value: q.value === null ? null : sign * q.value
};
export function grandCanonicalOccupancyModel(parameters: GrandOccupancyParameters) {
  const { sites: M, energyGap: epsilon, chemicalPotential: mu, temperature: T } = parameters;
  if (!Number.isInteger(M) || M < 0 || M > 4096) throw new RangeError('Expected 0 to 4096 sites');
  if (!Number.isFinite(epsilon) || epsilon < 0 || !Number.isFinite(mu) || !Number.isFinite(T))
    throw new RangeError('Expected finite energy, chemical potential and temperature');
  const e = product(epsilon), delta = sum([e, negate(product(mu))]);
  const temperatureSign = T < 0 ? -1 : 1, thermal = T === 0 ? one : product(kB, Math.abs(T));
  const bias = T < 0 ? delta : negate(delta);
  const kernel = binomialLogit(M, T === 0 ? product(0) : bias, thermal);
  const common = { scope: 'grand-canonical-classical-labeled-exclusion-sites' as const,
    parameters: { ...parameters }, boltzmannConstant: kB,
    mode: T === 0 ? 'positive-side-zero-temperature-limit' as const : T < 0 ? 'bounded-negative-temperature' as const : 'positive-temperature' as const,
    occupationCostMinusChemicalPotential: observe(delta), scaledChemicalBias: T === 0 ? null : ratio(bias, thermal),
    totalMicrostates: (BigInt(1) << BigInt(M)).toString(),
    approximationContract: 'Exact binary input differences/products and multiplicities; approximate transcendental quantities. Finite-temperature tails stay positive. Nested logs diagnose tails, not certified departures from unit-valued moments. No kinetics or particle flow is specified.' as const };
  const decorate = (rows: typeof kernel.distribution) => rows.map(row => ({ particles: row.excitations,
    multiplicity: row.multiplicity, probability: row.probability,
    conditionalPatternProbability: row.probability.sign === 0 ? null : rawQuantity(one, { coefficient: BigInt(row.multiplicity), exponent: 0 }) }));
  if (T === 0 && delta.coefficient !== Z) {
    const full = delta.coefficient < Z, particles = full ? M : 0;
    const distribution = decorate(kernel.distribution.map(row => ({ ...row,
      probability: row.excitations === particles ? rawQuantity(one) : zero() })));
    return { ...common, oneProbability: full ? rawQuantity(one) : zero(), zeroProbability: full ? zero() : rawQuantity(one), distribution,
      meanParticles: rawQuantity(product(particles)), meanHoles: rawQuantity(product(M - particles)),
      numberVariance: zero(), energyVariance: zero(), energyNumberCovariance: zero(),
      meanEnergy: rawQuantity(product(particles, epsilon)), entropy: zero(), entropyInKB: zero(),
      numberSusceptibility: zero(), relativeNumberFluctuation: full && M > 0 ? zero() : null,
      grandPotential: full ? signed(rawQuantity(negate(multiply(product(M), delta))), -1) : zero(),
      logGrandPartition: full && M > 0 ? null : zero(),
      logGrandPartitionStatus: full && M > 0 ? 'infinite-zero-temperature-limit' as const : 'finite-zero-temperature-limit' as const,
      susceptibilityStatus: M === 0 ? 'empty-system' as const : 'zero-temperature-away-from-threshold' as const };
  }
  const distribution = decorate(kernel.distribution);
  const relative = M === 0 ? null : exponential(sum([negate(bias), negate(multiply(thermal, product(Math.log(M))))]), multiply(product(2), thermal));
  if (M === 0) return { ...common, oneProbability: kernel.oneProbability, zeroProbability: kernel.zeroProbability, distribution,
    meanParticles: zero(), meanHoles: zero(), numberVariance: zero(), meanEnergy: zero(), energyVariance: zero(),
    energyNumberCovariance: zero(), entropy: zero(), entropyInKB: zero(), numberSusceptibility: zero(),
    relativeNumberFluctuation: null, grandPotential: zero(), logGrandPartition: zero(),
    logGrandPartitionStatus: T === 0 ? 'finite-zero-temperature-limit' as const : 'finite-temperature' as const,
    susceptibilityStatus: 'empty-system' as const };
  if (kernel.raw === null) {
    const logM = Math.log(M), thermalLog = observe(thermal).logMagnitude!;
    const q = (constant: number) => exponential(kernel.tailLog(constant), thermal);
    const meanParticles = kernel.positive ? rawQuantity(product(M)) : q(logM);
    const meanHoles = kernel.positive ? q(logM) : rawQuantity(product(M));
    const entropyFactor = kernel.scaledMagnitude.value === null ? kernel.scaledMagnitude.logMagnitude! : Math.log1p(kernel.scaledMagnitude.value);
    const logXi = kernel.positive ? rawQuantity(multiply(product(M), kernel.magnitude), thermal) : kernel.logPartitionCorrection;
    const potentialMagnitude = kernel.positive ? rawQuantity(multiply(product(M), kernel.magnitude)) : q(logM + thermalLog);
    return { ...common, oneProbability: kernel.oneProbability, zeroProbability: kernel.zeroProbability, distribution,
      meanParticles, meanHoles, numberVariance: q(logM),
      meanEnergy: epsilon === 0 ? zero() : kernel.positive ? rawQuantity(product(M, epsilon)) : q(logM + Math.log(epsilon)),
      energyVariance: epsilon === 0 ? zero() : q(logM + 2 * Math.log(epsilon)),
      energyNumberCovariance: epsilon === 0 ? zero() : q(logM + Math.log(epsilon)),
      entropy: q(logM + Math.log(kB) + entropyFactor), entropyInKB: q(logM + entropyFactor),
      numberSusceptibility: signed(q(logM - thermalLog), temperatureSign), relativeNumberFluctuation: relative,
      grandPotential: signed(potentialMagnitude, -temperatureSign), logGrandPartition: logXi,
      logGrandPartitionStatus: 'finite-temperature' as const, susceptibilityStatus: 'finite-temperature' as const };
  }
  const { w, W, correction } = kernel.raw, m = product(M), W2 = multiply(W, W);
  const particlesNumerator = multiply(m, kernel.positive ? one : w), holesNumerator = multiply(m, kernel.positive ? w : one);
  const varianceNumerator = multiply(m, w);
  const entropyNumerator = multiply(m, sum([multiply(multiply(correction, thermal), W), multiply(kernel.magnitude, w)]));
  const entropyDenominator = multiply(thermal, W);
  const partitionNumerator = multiply(m, sum([multiply(correction, thermal), kernel.positive ? kernel.magnitude : product(0)]));
  return { ...common, oneProbability: kernel.oneProbability, zeroProbability: kernel.zeroProbability, distribution,
    meanParticles: rawQuantity(particlesNumerator, W), meanHoles: rawQuantity(holesNumerator, W),
    numberVariance: rawQuantity(varianceNumerator, W2), meanEnergy: rawQuantity(multiply(particlesNumerator, e), W),
    energyVariance: rawQuantity(multiply(multiply(varianceNumerator, e), e), W2),
    energyNumberCovariance: rawQuantity(multiply(varianceNumerator, e), W2),
    entropy: rawQuantity(multiply(entropyNumerator, product(kB)), entropyDenominator), entropyInKB: rawQuantity(entropyNumerator, entropyDenominator),
    numberSusceptibility: T === 0 ? null : signed(rawQuantity(varianceNumerator, multiply(thermal, W2)), temperatureSign),
    relativeNumberFluctuation: relative,
    grandPotential: T === 0 ? zero() : signed(rawQuantity(partitionNumerator), -temperatureSign),
    logGrandPartition: rawQuantity(partitionNumerator, thermal),
    logGrandPartitionStatus: T === 0 ? 'finite-zero-temperature-limit' as const : 'finite-temperature' as const,
    susceptibilityStatus: T === 0 ? 'undefined-at-zero-temperature-threshold' as const : 'finite-temperature' as const };
}
