/** Scalar Gaussian posterior and an independent follow-up with the same sensor. */
export function calculateGaussianPosterior(
  priorMean: number,
  priorVariance: number,
  strength: number,
  noiseVariance: number,
  observation: number,
) {
  if (![priorMean, priorVariance, strength, noiseVariance, observation].every(Number.isFinite) ||
      priorVariance <= 0 || noiseVariance <= 0) {
    throw new RangeError('Expected finite parameters and positive variances');
  }
  const observationVariance = strength ** 2 * priorVariance + noiseVariance;
  const gain = priorVariance * strength / observationVariance;
  const mean = priorMean + gain * (observation - strength * priorMean);
  const variance = priorVariance * (noiseVariance / observationVariance);
  return {
    mean, variance,
    predictiveMean: strength * mean,
    predictiveVariance: strength ** 2 * variance + noiseVariance,
  };
}

export function gaussianDensity(x: number, mean: number, variance: number) {
  return Math.exp(-.5 * ((x - mean) / Math.sqrt(variance)) ** 2) / Math.sqrt(2 * Math.PI * variance);
}
