/** Exact propagation for sums and differences of two correlated readings. */
export function calculateCorrelatedVariance(sigmaA: number, sigmaB: number, correlation: number) {
  if (![sigmaA, sigmaB, correlation].every(Number.isFinite) || sigmaA <= 0 || sigmaB <= 0 || Math.abs(correlation) > 1) {
    throw new RangeError('Expected positive scales and a correlation in [-1, 1]');
  }
  const cross = 2 * sigmaA * sigmaB;
  return {
    covariance: correlation * sigmaA * sigmaB,
    sumVariance: (sigmaA - sigmaB) ** 2 + cross * (1 + correlation),
    differenceVariance: (sigmaA - sigmaB) ** 2 + cross * (1 - correlation),
  };
}
