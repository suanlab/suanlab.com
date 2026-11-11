/** Orthonormal rest-frame components for signature (-,+,+,+), with κ=1. */
export function calculateEinsteinFluidCurvature(density: number, pressure: number, lambda: number) {
  if (![density, pressure, lambda].every(Number.isFinite) || density < 0) {
    throw new RangeError('Expected finite fluid parameters and nonnegative density');
  }
  const stressTrace = -density + 3 * pressure;
  const scalarCurvature = 4 * lambda - stressTrace;
  const timeRicci = (density + 3 * pressure) / 2 - lambda;
  const spatialRicci = (density - pressure) / 2 + lambda;
  return {
    stressTrace, scalarCurvature, timeRicci, spatialRicci,
    timeEinstein: timeRicci + scalarCurvature / 2,
    spatialEinstein: spatialRicci - scalarCurvature / 2,
  };
}
