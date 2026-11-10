/** Adaptive enclosure of D(p||pi) from the ORIGINAL finite Hamiltonian.
 * This does not subtract two displayed free energies or normalize rounded
 * canonical probabilities as replacement weights. Uniform cold ground-manifold
 * preparations use a separate enclosed logarithmic tail calculation.
 */
import { finiteTrialThermodynamics } from './learning-trial-thermodynamics';
import { finiteEquilibriumFreeEnergy } from './learning-equilibrium-free-energy';
import { canonicalBoltzmannConstant as kB } from './learning-canonical-spectrum';
import { createTranscendentalContext, type FixedInterval } from './learning-interval-transcendentals';
import { canonicalGroundTail } from './learning-canonical-ground-tail';
import { binaryProduct as product, binarySum as sum, binaryMultiply as multiply,
  binaryNegate as negate, binaryObservation as observe, type BinaryValue } from './learning-binary-products';
const ZERO = BigInt(0), ONE = BigInt(1);
const floor = (n: bigint, d: bigint) => n >= ZERO ? n / d : -((-n + d - ONE) / d);
const ceil = (n: bigint, d: bigint) => -floor(-n, d);
function scaledFraction(n: BinaryValue, d: BinaryValue, bits: number): FixedInterval {
  let a = n.coefficient, b = d.coefficient;
  const shift = n.exponent - d.exponent + bits;
  if (shift >= 0) a <<= BigInt(shift); else b <<= BigInt(-shift);
  return { lower: floor(a, b), upper: ceil(a, b) };
}
function weighted(v: FixedInterval, n: BinaryValue, d: BinaryValue): FixedInterval {
  const lower = scaledFraction(multiply({ coefficient: v.lower, exponent: 0 }, n), d, 0);
  const upper = scaledFraction(multiply({ coefficient: v.upper, exponent: 0 }, n), d, 0);
  return { lower: lower.lower, upper: upper.upper };
}
const add = (a: FixedInterval, b: FixedInterval): FixedInterval => ({ lower: a.lower + b.lower, upper: a.upper + b.upper });
export function finiteCanonicalHelmholtzGap(energies: readonly number[], temperature: number,
  trialWeights: readonly number[], copies = 1) {
  const trial = finiteTrialThermodynamics(energies, temperature, trialWeights, copies);
  const equilibrium = finiteEquilibriumFreeEnergy(energies, temperature, copies);
  const weights = trialWeights.map(w => product(w)), W = sum(weights), thermal = product(kB, temperature);
  const minimum = product(trial.canonical.minimumEnergy);
  const gaps = energies.map(e => sum([product(e), negate(minimum)]));
  const structuralEquilibrium = gaps.every(g => g.coefficient === ZERO)
    && trialWeights.every(w => w === trialWeights[0]);
  const zero = observe(product(0));
  if (structuralEquilibrium) return { trial, equilibrium, resolution: 'structural-equilibrium' as const,
    precisionBits: 0, divergence: zero, perCopyGap: zero, totalGap: zero,
    bounds: { lower: zero, upper: zero }, exactInterval: { lowerCoefficient: '0', upperCoefficient: '0', exponent: 0 }, attempts: [] as number[] };
  const groundTail = canonicalGroundTail(gaps, thermal, trialWeights, copies);
  if (groundTail !== null) return { trial, equilibrium, ...groundTail };
  const energyNumerator = sum(weights.map((w, i) => multiply(w, gaps[i])));
  const energyDenominator = multiply(W, thermal);
  const attempts: number[] = [];
  let latest: FixedInterval = { lower: ZERO, upper: ZERO }, precisionBits = 0;
  for (const P of [128, 256, 512, 1024, 2048, 4096, 8192]) {
    attempts.push(P); precisionBits = P;
    const c = createTranscendentalContext(P);
    let partition: FixedInterval = { lower: ZERO, upper: ZERO };
    for (const gap of gaps) partition = add(partition, c.expNegativeRatio(gap, thermal));
    const lowerLogZ = c.logRatio({ coefficient: partition.lower, exponent: -P }, product(1));
    const upperLogZ = c.logRatio({ coefficient: partition.upper, exponent: -P }, product(1));
    let result = add(scaledFraction(energyNumerator, energyDenominator, P), { lower: lowerLogZ.lower, upper: upperLogZ.upper });
    for (const w of weights) {
      if (w.coefficient === ZERO) continue;
      result = add(result, weighted(c.logRatio(w, W), w, W));
    }
    latest = result;
    // Relative interval width <=2^-40, with a STRICTLY positive lower bound.
    // This criterion selects numerical precision; it never defines equilibrium.
    if (result.lower > ZERO && ((result.upper - result.lower) << BigInt(40)) <= result.lower) {
      const midpoint = { coefficient: result.lower + result.upper, exponent: -P - 1 };
      return { trial, equilibrium, resolution: 'positive-interval-resolved' as const, precisionBits: P,
        divergence: observe(midpoint), perCopyGap: observe(multiply(midpoint, thermal)),
        totalGap: observe(multiply(multiply(midpoint, thermal), product(copies))),
        bounds: { lower: observe({ coefficient: result.lower, exponent: -P }), upper: observe({ coefficient: result.upper, exponent: -P }) },
        exactInterval: { lowerCoefficient: result.lower.toString(), upperCoefficient: result.upper.toString(), exponent: -P }, attempts };
    }
  }
  // null here means the numerical question is unresolved, NOT D=0 or equilibrium.
  return { trial, equilibrium, resolution: 'unresolved-at-precision-ceiling' as const, precisionBits,
    divergence: null, perCopyGap: null, totalGap: copies === 0 ? zero : null,
    bounds: { lower: observe({ coefficient: latest.lower, exponent: -precisionBits }), upper: observe({ coefficient: latest.upper, exponent: -precisionBits }) },
    exactInterval: { lowerCoefficient: latest.lower.toString(), upperCoefficient: latest.upper.toString(), exponent: -precisionBits }, attempts };
}
