const elementaryCharge = 1.602176634e-19;
const boltzmannConstant = 1.380649e-23;

/** Output noise for a passive RC readout, with one-sided spectra and SI units. */
export function calculatePhotodiodeNoise(resistance: number, capacitance: number, current: number, temperature: number) {
  if (![resistance, capacitance, current, temperature].every(Number.isFinite) ||
      resistance <= 0 || capacitance <= 0 || current < 0 || temperature <= 0) {
    throw new RangeError('Expected positive resistance, capacitance, temperature, and nonnegative current');
  }
  const cutoff = 1 / (2 * Math.PI * resistance * capacitance);
  const equivalentBandwidth = 1 / (4 * resistance * capacitance);
  const shotVariance = elementaryCharge * current * resistance / (2 * capacitance);
  const thermalVariance = boltzmannConstant * temperature / capacitance;
  return { cutoff, equivalentBandwidth, shotVariance, thermalVariance, rmsVoltage: Math.sqrt(shotVariance + thermalVariance) };
}
