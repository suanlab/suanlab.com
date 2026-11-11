/** Candidate paths agreeing with integer-time measurements of unit-rate decay. */
export function calculateDynamicsResidual(amplitude: number, time: number) {
  if (![amplitude, time].every(Number.isFinite) || time < 0 || time > 2) {
    throw new RangeError('Expected finite amplitude and time in [0, 2]');
  }
  const baseline = Math.exp(-time);
  const perturbation = Number.isInteger(time) ? 0 : amplitude * Math.sin(Math.PI * time);
  return {
    baseline,
    candidate: baseline + perturbation,
    observationResidual: perturbation,
    equationResidual: amplitude * Math.PI * Math.cos(Math.PI * time) + perturbation,
    equationRms: Math.abs(amplitude) * Math.sqrt((Math.PI ** 2 + 1) / 2),
  };
}
