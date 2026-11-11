/** Population MSE for independent centered features with unit variance. */
export function calculateRandomizedControls(coefficient: number, randomCorrelation: number) {
  if (![coefficient, randomCorrelation].every(Number.isFinite) || coefficient < 0 || Math.abs(randomCorrelation) > 1) {
    throw new RangeError('Expected a nonnegative coefficient and correlation in [-1, 1]');
  }
  const signalVariance = coefficient ** 2;
  const noiseVariance = .01;
  return {
    full: noiseVariance,
    removed: signalVariance + noiseVariance,
    frozenRandom: 2 * signalVariance * (1 - randomCorrelation) + noiseVariance,
    refittedRandom: signalVariance * (1 - randomCorrelation ** 2) + noiseVariance,
    refittedCoefficient: coefficient * randomCorrelation,
  };
}
