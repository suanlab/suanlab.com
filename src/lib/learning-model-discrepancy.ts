/** Compare an offset decay with a zero-equilibrium model fitted at time one. */
export function calculateModelDiscrepancy(equilibrium: number, rate: number, time: number) {
  if (![equilibrium, rate, time].every(Number.isFinite) ||
      equilibrium < 0 || equilibrium >= 1 || rate <= 0 || time < 0) {
    throw new RangeError('Expected equilibrium in [0, 1), positive rate, and nonnegative time');
  }
  const truth = equilibrium + (1 - equilibrium) * Math.exp(-rate * time);
  const fittedRate = equilibrium === 0 ? rate : -Math.log1p((1 - equilibrium) * Math.expm1(-rate));
  const prediction = Math.exp(-fittedRate * time);
  return { truth, fittedRate, prediction, discrepancy: truth - prediction };
}
