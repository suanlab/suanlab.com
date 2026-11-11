/** Episode and observation-weighted effects, using population-form covariance. */
export function calculateAggregationEffects(counts: readonly number[], effects: readonly number[]) {
  if (counts.length === 0 || counts.length !== effects.length ||
      !counts.every(value => Number.isInteger(value) && value > 0) || !effects.every(Number.isFinite)) {
    throw new RangeError('Expected aligned nonempty effects and positive integer counts');
  }
  const total = counts.reduce((sum, value) => sum + value, 0);
  const meanCount = total / counts.length;
  const episodeMean = effects.reduce((sum, value) => sum + value, 0) / effects.length;
  const observationMean = counts.reduce((sum, value, i) => sum + value * effects[i], 0) / total;
  const covariance = counts.reduce((sum, value, i) => sum + (value - meanCount) * (effects[i] - episodeMean), 0) / counts.length;
  return { episodeMean, observationMean, covariance, meanCount };
}
