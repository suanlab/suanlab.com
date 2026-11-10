/** Classical ideal-gas states at fixed n,T with a constant external pressure.
 * Fref is supplied at this run's n,T,Vref. Its temperature dependence is NOT
 * specified, so only isothermal entropy/energy differences are determined.
 * No piston trajectory, elapsed time or relaxation rate is defined.
 * Products/balances use supplied binary inputs; logarithms remain approximate.
 */
import { binaryProduct as product, binarySum as sum, binaryNegate as negate,
  binaryMultiply as multiply, binaryRatioObservation as observe,
  type BinaryValue, type BinaryObservation } from './learning-binary-products';
import { idealGasConstant as R } from './learning-ideal-gas-paths';

export interface GasFreeEnergyParameters {
  moles: number;
  temperature: number;
  externalPressure: number;
  volume: number;
  referenceVolume: number;
  /** Total Helmholtz value at the supplied n,T,Vref, in joules. */
  referenceHelmholtz: number;
}
export interface AnchoredFreeEnergy extends BinaryObservation {
  /** A cancellation of rounded logarithms is not proof of physical zero. */
  zeroStatus: 'not-zero-approximation' | 'structural-zero' | 'unresolved-transcendental-cancellation';
}
interface Fraction { numerator: BinaryValue; denominator: BinaryValue }
const one = product(1);
const difference = (a: BinaryValue, b: BinaryValue) => sum([a, negate(b)]);
const zero = () => observe(product(0), one);
function bound(value: number, low: number, high: number, name: string) {
  if (!Number.isFinite(value) || value < low || value > high) throw new RangeError(`Invalid ${name}`);
}
function validate(p: GasFreeEnergyParameters) {
  bound(p.moles, .25, 2, 'moles'); bound(p.temperature, 250, 800, 'temperature');
  bound(p.externalPressure, 25000, 200000, 'external pressure');
  bound(p.volume, .001, 1, 'volume'); bound(p.referenceVolume, .001, 1, 'reference volume');
  bound(p.referenceHelmholtz, -10000, 10000, 'reference Helmholtz energy');
}
/** Preserve ln(N/D) near one using the exact pre-projection displacement.
 * A displayed ratio of one need not make this logarithm zero.
 */
function logRatio(numerator: BinaryValue, denominator: BinaryValue): Fraction {
  const gap = difference(numerator, denominator);
  if (gap.coefficient === BigInt(0)) return { numerator: product(0), denominator: one };
  const delta = observe(gap, denominator).value!;
  if (Math.abs(delta) <= .125) {
    const gain = delta === 0 ? 1 : Math.log1p(delta) / delta;
    return { numerator: multiply(gap, product(gain)), denominator };
  }
  return { numerator: product(Math.log(observe(numerator, denominator).value!)), denominator: one };
}
/** x-1-ln(x) = delta² sum[k>=0] (-delta)^k/(k+2).
 * With |delta|<=1/8 the omitted tail after k=32 is bounded by
 * |delta|^33/(35(1-|delta|)), below 6e-32. Binary64 rounding still applies.
 */
function remainderGain(delta: number) {
  let value = 0;
  for (let k = 32; k >= 0; k--) value = 1 / (k + 2) - delta * value;
  return value;
}
export function gasVolumeFreeEnergy(p: GasFreeEnergyParameters) {
  validate(p);
  const A = product(p.moles, R, p.temperature), pressure = product(p.externalPressure), volume = product(p.volume);
  const pressureVolume = product(p.externalPressure, p.volume), balance = difference(pressureVolume, A);
  const displacement = observe(balance, A), logX = logRatio(pressureVolume, A);
  const logReference = logRatio(volume, product(p.referenceVolume));
  const logEquilibriumReference = logRatio(A, product(p.externalPressure, p.referenceVolume));
  let gap: Fraction;
  if (balance.coefficient === BigInt(0)) gap = { numerator: product(0), denominator: one };
  else if (Math.abs(displacement.value!) <= .125) gap = {
    numerator: multiply(multiply(balance, balance), product(remainderGain(displacement.value!))), denominator: A,
  };
  else gap = {
    numerator: difference(multiply(balance, logX.denominator), multiply(A, logX.numerator)),
    denominator: logX.denominator,
  };
  const availability = observe(gap.numerator, gap.denominator);
  // Never turn a numerical failure into a purported equilibrium by clamping.
  if (balance.coefficient !== BigInt(0) && availability.sign !== 1)
    throw new RangeError('Failed positive availability evaluation');
  function anchored(logarithm: Fraction, extra: BinaryValue): AnchoredFreeEnergy {
    const base = sum([product(p.referenceHelmholtz), extra]);
    const numerator = difference(multiply(base, logarithm.denominator), multiply(A, logarithm.numerator));
    const result = observe(numerator, logarithm.denominator);
    return { ...result, zeroStatus: result.sign !== 0 ? 'not-zero-approximation'
      : logarithm.numerator.coefficient === BigInt(0) ? 'structural-zero' : 'unresolved-transcendental-cancellation' };
  }
  const helmholtzChangeFromEquilibrium = observe(negate(multiply(A, logX.numerator)), logX.denominator);
  // Build the entropy numerator from raw factors, not the rounded observation.
  const gasEntropy = observe(multiply(product(-p.moles, R), logX.numerator), logX.denominator);
  return {
    parameters: { ...p }, scope: 'fixed-temperature-branch-with-supplied-reference-anchor' as const,
    regime: balance.coefficient === BigInt(0) ? 'equilibrium' as const : displacement.sign < 0 ? 'below-equilibrium-volume' as const : 'above-equilibrium-volume' as const,
    equilibriumConditionSatisfied: balance.coefficient === BigInt(0),
    thermalEnergyScale: observe(A, one), equilibriumVolume: observe(A, pressure),
    volumeRatioToEquilibrium: observe(pressureVolume, A), relativeVolumeDisplacement: displacement,
    volumeDisplacement: observe(balance, pressure), pressureVolumeMinusThermalEnergy: observe(balance, one),
    gasPressure: observe(A, volume), externalMinusGasPressure: observe(balance, volume),
    constrainedPotentialCurvature: observe(A, multiply(volume, volume)),
    logVolumeRatioFromReference: observe(logReference.numerator, logReference.denominator),
    logVolumeRatioToEquilibrium: observe(logX.numerator, logX.denominator),
    helmholtzChangeFromReference: observe(negate(multiply(A, logReference.numerator)), logReference.denominator),
    entropyChangeFromReference: observe(multiply(product(p.moles, R), logReference.numerator), logReference.denominator),
    internalEnergyChangeFromReference: zero(),
    helmholtzFreeEnergy: anchored(logReference, product(0)),
    gasGibbsFreeEnergy: anchored(logReference, A),
    constrainedPotential: anchored(logReference, pressureVolume),
    gibbsAtExternalPressureEquilibrium: anchored(logEquilibriumReference, A),
    constrainedMinusGasGibbs: observe(balance, one),
    helmholtzChangeFromEquilibrium,
    externalPressureVolumePotentialChangeFromEquilibrium: observe(balance, one),
    availability, availabilityPerThermalEnergy: observe(gap.numerator, multiply(gap.denominator, A)),
    /** Conditional endpoint budget only: fixed thermal reservoir T, constant
     * external-pressure work, no other work, zero initial/final kinetic store.
     * It provides neither a realizable schedule nor a convergence-time claim.
     */
    conditionalEquilibrationBudget: {
      energyChangeGas: zero(), heatIntoGas: observe(negate(balance), one), workByGas: observe(negate(balance), one),
      entropyChangeGas: gasEntropy, entropyChangeThermalReservoir: observe(balance, product(p.temperature)),
      entropyProduction: observe(gap.numerator, multiply(gap.denominator, product(p.temperature))),
      // Reversible comparison uses a varying quasistatic load between the same
      // endpoints; it is not the constant-external-pressure process above.
      maximumReversibleWorkByGas: helmholtzChangeFromEquilibrium,
      workDeficitFromReversible: availability,
    },
  };
}
