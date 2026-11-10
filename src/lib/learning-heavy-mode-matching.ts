export interface HeavyModeInput {
  massRatio: number;
  mixing: number;
  frequencyRatio: number;
  order: number;
}

/** Euclidean response in units where the light mass is one. */
export function calculateHeavyModeMatching(input: HeavyModeInput) {
  const { massRatio, mixing, frequencyRatio, order } = input;
  if (!Number.isFinite(massRatio) || massRatio < 2 || massRatio > 16 ||
      !Number.isFinite(mixing) || mixing < 0 || mixing > .99 ||
      !Number.isFinite(frequencyRatio) || frequencyRatio < 0 || frequencyRatio > 2 ||
      !Number.isInteger(order) || order < 0 || order > 6) {
    throw new RangeError('Unsupported heavy-mode parameters');
  }

  const x = frequencyRatio ** 2;
  const omega = massRatio * frequencyRatio;
  const strength = mixing ** 2;
  const staticKernel = (1 - mixing) * (1 + mixing);
  const kineticCoefficient = 1 + strength / massRatio ** 2;
  const exactKernel = omega ** 2 + (staticKernel + x) / (1 + x);
  let retained = 0;
  for (let power = order; power >= 0; power--) retained = 1 - x * retained;
  const approximateKernel = omega ** 2 + 1 - strength * retained;
  const kernelError = strength * x ** (order + 1) / (1 + x);
  return {
    staticKernel,
    kineticCoefficient,
    exactKernel,
    approximateKernel,
    kernelError,
    exactResponse: 1 / exactKernel,
    approximateResponse: approximateKernel > 0 ? 1 / approximateKernel : null,
    responseError: approximateKernel > 0 ? kernelError / (exactKernel * approximateKernel) : null,
  };
}
