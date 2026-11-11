/** Passive photodiode readout using SI units and a fixed small-signal operating point. */
export function calculatePhotodiodeResponse(
  resistance: number,
  capacitance: number,
  responsivity: number,
  opticalStep: number,
  frequency: number,
) {
  if (![resistance, capacitance, responsivity, opticalStep, frequency].every(Number.isFinite) ||
      resistance <= 0 || capacitance <= 0 || responsivity < 0 || frequency < 0) {
    throw new RangeError('Expected positive resistance and capacitance, and nonnegative responsivity and frequency');
  }
  const tau = resistance * capacitance;
  const dcGain = resistance * responsivity;
  const scaledFrequency = 2 * Math.PI * frequency * tau;
  return {
    tau,
    dcGain,
    cutoff: 1 / (2 * Math.PI * tau),
    riseTime: tau * Math.log(9),
    finalVoltage: dcGain * opticalStep,
    relativeAmplitude: 1 / Math.hypot(1, scaledFrequency),
    phaseDegrees: scaledFrequency === 0 ? 0 : -Math.atan(scaledFrequency) * 180 / Math.PI,
  };
}

export function calculatePhotodiodeStep(finalVoltage: number, tau: number, seconds: number) {
  return seconds <= 0 ? 0 : -finalVoltage * Math.expm1(-seconds / tau);
}
