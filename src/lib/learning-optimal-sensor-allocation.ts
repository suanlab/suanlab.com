/** Enumerate A- and D-optimal integer allocations for two independent sensors. */
export function calculateSensorAllocations(total: number, sigmaA: number, sigmaB: number) {
  if (!Number.isInteger(total) || total < 2 || total > 1000 ||
      ![sigmaA, sigmaB].every(value => Number.isFinite(value) && value > 0)) {
    throw new RangeError('Expected a budget of 2–1000 and positive noise scales');
  }
  const plans = Array.from({ length: total - 1 }, (_, i) => {
    const first = i + 1, second = total - first;
    const varianceA = sigmaA ** 2 / first;
    const varianceB = sigmaB ** 2 / second;
    return { first, second, varianceA, varianceB, trace: varianceA + varianceB, determinant: varianceA * varianceB };
  });
  const traceBest = plans.reduce((best, plan) => plan.trace < best.trace ? plan : best);
  const determinantBest = plans.reduce((best, plan) => plan.determinant < best.determinant ? plan : best);
  return { plans, traceBest, determinantBest };
}
