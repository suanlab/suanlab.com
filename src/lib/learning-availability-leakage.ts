/** Exact linear prediction risks under a decision-time information boundary. */
export function calculateAvailabilityLeakage(correlation: number) {
  if (!Number.isFinite(correlation) || Math.abs(correlation) > 1) {
    throw new RangeError('Expected correlation in [-1, 1]');
  }
  return {
    availableMse: .64 * (1 - correlation ** 2) + .36,
    futureMse: .36,
    targetMse: 0,
    availableCoefficient: .8 * correlation,
  };
}
