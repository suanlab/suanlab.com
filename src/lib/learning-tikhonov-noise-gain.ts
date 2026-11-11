/** Zeroth-order Tikhonov filter for a singular mode with known synthetic truth. */
export function calculateTikhonovMode(strength: number, penalty: number, truth: number, sigma: number, error: number) {
  if (![strength, penalty, truth, sigma, error].every(Number.isFinite) || strength < 0 || penalty < 0 || sigma < 0) {
    throw new RangeError('Expected finite inputs and nonnegative strength, penalty, and noise scale');
  }
  const denominator = strength ** 2 + penalty;
  if (denominator === 0) return null;
  const signalRetention = strength ** 2 / denominator;
  const noiseGain = strength / denominator;
  const bias = -penalty / denominator * truth;
  const noiseDeviation = noiseGain * sigma;
  return {
    signalRetention, noiseGain, bias, noiseDeviation,
    estimate: signalRetention * truth + noiseGain * error,
    meanSquaredError: bias ** 2 + noiseDeviation ** 2,
  };
}
