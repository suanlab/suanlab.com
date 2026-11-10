/** Anchored equilibrium Helmholtz energy from a fixed finite Hamiltonian.
 * Centered partition corrections are multiplied before projection, including
 * corrections whose centered logarithm itself has no binary64 linear value.
 */
import { canonicalFiniteSpectrum, canonicalBoltzmannConstant as kB } from './learning-canonical-spectrum';
import { binaryProduct as product, binaryMultiply as multiply, binarySum as sum,
  binaryNegate as negate, binaryObservation as observe,
  type BinaryValue, type BinaryObservation } from './learning-binary-products';
import type { TrialFreeEnergyObservation } from './learning-trial-thermodynamics';
/** Recover exp(log) as a scaled binary approximation, without projecting its
 * power of two. This is approximate transcendental arithmetic, not a correctly
 * rounded arbitrary-precision exponential.
 */
function exponential(log: number): BinaryValue {
  const exponent = Math.floor(log / Math.LN2);
  const mantissa = product(Math.exp(log - exponent * Math.LN2));
  return { coefficient: mantissa.coefficient, exponent: mantissa.exponent + exponent };
}
export function finiteEquilibriumFreeEnergy(energies: readonly number[], temperature: number, copies = 1) {
  const canonical = canonicalFiniteSpectrum(energies, temperature);
  if (typeof copies !== 'number' || !Number.isFinite(copies) || !Number.isInteger(copies) || copies < 0)
    throw new RangeError('Expected finite nonnegative integer independent-copy count');
  const L = canonical.centeredLogPartition, thermal = product(kB, temperature);
  const logThermal = observe(thermal).logMagnitude!;
  // Below this branch every correction, even multiplied by MAX_VALUE copies,
  // is far below the smallest energy-input quantum. Its negative direction can
  // still decide an EXACT rounding tie of Emin*copies; do not discard it.
  const infinitesimal = L.sign !== 0 && (L.logMagnitude === null || L.logMagnitude < -10000);
  const rawL = L.sign === 0 ? product(0) : infinitesimal ? null
    : L.value !== null && L.value >= 2 ** -1022 ? product(L.value) : exponential(L.logMagnitude!);
  const rawCorrection = rawL === null ? null : multiply(thermal, rawL);
  const correction = (count: number): BinaryObservation => {
    if (count === 0 || L.sign === 0) return observe(product(0));
    if (rawCorrection !== null) {
      const result = observe(multiply(rawCorrection, product(count)));
      // Preserve the pre-projection logarithm for reconstructed subnormals.
      return { ...result, logMagnitude: logThermal + L.logMagnitude! + Math.log(count) };
    }
    return { value: null, sign: 1,
      logMagnitude: L.logMagnitude === null ? null : logThermal + L.logMagnitude + Math.log(count) };
  };
  const freeEnergy = (count: number): TrialFreeEnergyObservation => {
    if (count === 0) return { ...observe(product(0)), zeroStatus: 'structural-zero' };
    const anchor = product(canonical.minimumEnergy, count);
    if (infinitesimal && canonical.minimumEnergy === 0) {
      const c = correction(count);
      return { value: null, sign: -1, logMagnitude: c.logMagnitude, zeroStatus: 'not-zero-approximation' };
    }
    // This sentinel is ONLY a directed rounding witness for a nonzero rational
    // anchor. It is never returned as the physical partition correction. Both
    // it and the actual infinitesimal lie below every relevant rounding quantum.
    const c = rawCorrection === null
      ? multiply({ coefficient: BigInt(1), exponent: -20000 }, product(count))
      : multiply(rawCorrection, product(count));
    const result = observe(sum([anchor, negate(c)]));
    const structural = L.sign === 0 && canonical.minimumEnergy === 0;
    return { ...result, zeroStatus: structural ? 'structural-zero'
      : result.sign === 0 ? 'unresolved-transcendental-cancellation' : 'not-zero-approximation' };
  };
  return { scope: 'fixed-finite-hamiltonian-equilibrium-helmholtz' as const,
    energies: [...energies], temperature, copies, canonical,
    perCopy: { helmholtz: freeEnergy(1), centeredPartitionEnergyCorrection: correction(1) },
    total: { helmholtz: freeEnergy(copies), centeredPartitionEnergyCorrection: correction(copies) },
    correctionRegime: L.sign === 0 ? 'structural-zero' as const : infinitesimal ? 'logarithmic-infinitesimal' as const
      : L.value === null || L.value < 2 ** -1022 ? 'reconstructed-before-scaling' as const : 'finite-centered-log-partition' as const,
    approximationContract: 'Feq=Emin-kBT*log(sum(exp(-(Ei-Emin)/kBT))). Anchored transcendental approximations do not certify the positive trial free-energy gap.' as const };
}
