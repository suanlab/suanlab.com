/** Fixed finite Hamiltonian at positive temperature. Exact rational centered
 * logits are retained; transcendental observations are approximations. This
 * module does not yet calculate trial free energies or certify a Helmholtz gap.
 */
import { binaryProduct as product, binarySum as sum, binaryNegate as negate,
  binaryRatioObservation as ratio, binaryObservation as observe,
  type BinaryValue, type BinaryObservation } from './learning-binary-products';
export const canonicalBoltzmannConstant = 1.380649e-23;
const subtract = (a: BinaryValue, b: BinaryValue) => sum([a, negate(b)]);
const positiveExponential = (log: BinaryObservation, groundState: boolean): BinaryObservation => {
  const projected = log.value === null ? groundState ? 1 : null : Math.exp(log.value);
  return { value: projected === 0 ? null : projected, sign: 1, logMagnitude: log.value };
};
const serialized = (v: BinaryValue) => ({ coefficient: v.coefficient.toString(), exponent: v.exponent });
/** Ordinary arrays of 1..32 finite microstate energies, in joules. Temperature
 * is any finite positive binary64 kelvin input. Degeneracy is represented by
 * repeated microstates, not an additional multiplicity factor.
 */
export function canonicalFiniteSpectrum(energies: readonly number[], temperature: number) {
  if (!Array.isArray(energies) || energies.length < 1 || energies.length > 32
    || !Array.from(energies).every(x => typeof x === 'number' && Number.isFinite(x)))
    throw new RangeError('Expected 1..32 finite microstate energies');
  if (typeof temperature !== 'number' || !Number.isFinite(temperature) || temperature <= 0)
    throw new RangeError('Expected finite positive reservoir temperature');
  const minimumEnergy = Math.min(...energies), thermal = product(canonicalBoltzmannConstant, temperature);
  const gaps = energies.map(e => subtract(product(e), product(minimumEnergy)));
  const groundCount = gaps.filter(g => g.coefficient === BigInt(0)).length;
  const excited = gaps.filter(g => g.coefficient !== BigInt(0));
  const minimumGap = excited.reduce<BinaryValue | null>((a, b) => a === null || subtract(b, a).coefficient < BigInt(0) ? b : a, null);
  let logExcitedWeightSum: BinaryObservation | null = null;
  let centeredLogPartition: BinaryObservation;
  if (minimumGap === null) centeredLogPartition = observe(product(Math.log(groundCount)));
  else {
    // Center the excited sum independently: no subtraction of rounded, enormous
    // logits, and at least one term is exactly one in this comparison sum.
    const terms = excited.map(g => {
      const d = ratio(subtract(g, minimumGap), thermal);
      if (d.sign === 0) return 1;
      return d.value === null ? d.logMagnitude! < 0 ? 1 : 0 : Math.exp(-d.value);
    });
    const logMultiplicity = Math.log(terms.reduce((a, b) => a + b, 0));
    const scaledMinimum = ratio(minimumGap, thermal);
    // If the minimum logit underflows, it is negligible relative to the finite
    // log of the number of excited states. An overflowing logit is retained.
    logExcitedWeightSum = scaledMinimum.value === null && scaledMinimum.logMagnitude! > 0
      ? { value: null, sign: -1, logMagnitude: scaledMinimum.logMagnitude }
      : observe(product(logMultiplicity - (scaledMinimum.value ?? 0)));
    const tail = logExcitedWeightSum.value === null ? 0 : Math.exp(logExcitedWeightSum.value);
    if (groundCount === 1 && (logExcitedWeightSum.value === null || logExcitedWeightSum.value < -36)) {
      // log(1+tail)>0 even when the exponential cannot be displayed. Its log
      // is asymptotic to log(tail); this is an observation, not an exact zero.
      centeredLogPartition = { value: tail === 0 ? null : tail, sign: 1, logMagnitude: logExcitedWeightSum.value };
    } else {
      centeredLogPartition = observe(product(Math.log(groundCount) + Math.log1p(tail / groundCount)));
    }
  }
  const states = gaps.map((gap, index) => {
    const scaledEnergyGap = ratio(gap, thermal);
    let logProbability: BinaryObservation;
    if (gap.coefficient === BigInt(0)) {
      logProbability = { value: centeredLogPartition.value === null ? null : -centeredLogPartition.value,
        sign: centeredLogPartition.sign === 0 ? 0 : -centeredLogPartition.sign, logMagnitude: centeredLogPartition.logMagnitude };
      if (logProbability.sign === 0) logProbability.value = 0;
    } else if (centeredLogPartition.value === null) {
      logProbability = { ...scaledEnergyGap, value: scaledEnergyGap.value === null ? null : -scaledEnergyGap.value, sign: -1 };
    } else {
      const correction = product(centeredLogPartition.value, canonicalBoltzmannConstant, temperature);
      logProbability = ratio(negate(sum([gap, correction])), thermal);
    }
    return { index, energy: energies[index], groundState: gap.coefficient === BigInt(0),
      energyGap: observe(gap), scaledEnergyGap, logProbability,
      probability: positiveExponential(logProbability, gap.coefficient === BigInt(0)),
      exactScaledEnergyGap: { numerator: serialized(gap), denominator: serialized(thermal) } };
  });
  return { scope: 'fixed-finite-hamiltonian-positive-temperature' as const,
    energies: [...energies], temperature, boltzmannConstant: canonicalBoltzmannConstant,
    minimumEnergy, groundCount, thermalEnergy: observe(thermal), centeredLogPartition,
    logExcitedWeightSum, states,
    normalization: 'exp(-scaledEnergyGap)/sum(exp(-scaledEnergyGap))' as const,
    observationContract: 'Approximate transcendental observations; positive probabilities remain positive when the linear or logarithmic display is out of range.' as const };
}
