type Observation = number | null;

/** Scalar forecast and analysis with independent process and observation noise. */
export function calculateKalmanAssimilation(
  initialMean: number,
  initialVariance: number,
  persistence: number,
  processVariance: number,
  observationVariance: number,
  observations: readonly Observation[],
) {
  if (![initialMean, initialVariance, persistence, processVariance, observationVariance].every(Number.isFinite) ||
      initialVariance < 0 || processVariance < 0 || observationVariance <= 0 ||
      !observations.every(value => value === null || Number.isFinite(value))) {
    throw new RangeError('Expected finite inputs, nonnegative state variances, and positive observation variance');
  }
  let mean = initialMean;
  let variance = initialVariance;
  return observations.map(observation => {
    const forecastMean = persistence * mean;
    const forecastVariance = persistence ** 2 * variance + processVariance;
    const innovation = observation === null ? null : observation - forecastMean;
    const gain = innovation === null ? 0 : forecastVariance / (forecastVariance + observationVariance);
    mean = forecastMean + gain * (innovation ?? 0);
    variance = innovation === null ? forecastVariance
      : forecastVariance * (observationVariance / (forecastVariance + observationVariance));
    return { observation, forecastMean, forecastVariance, innovation, gain, mean, variance };
  });
}
