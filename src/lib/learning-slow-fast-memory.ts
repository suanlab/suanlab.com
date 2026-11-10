export interface SlowFastInput {
  fastRate: number;
  mixing: number;
  initialFast: number;
  time: number;
}

/** Two linear variables with x(0)=1 and a selectable hidden initial value. */
export function calculateSlowFastMemory(input: SlowFastInput) {
  const { fastRate, mixing, initialFast, time } = input;
  if (!Number.isFinite(fastRate) || fastRate < 2 || fastRate > 64 ||
      !Number.isFinite(mixing) || mixing < 0 || mixing > .95 ||
      !Number.isFinite(initialFast) || initialFast < 0 || initialFast > 2 ||
      !Number.isFinite(time) || time < 0 || time > 10) {
    throw new RangeError('Unsupported slow-fast parameters');
  }

  const gap = Math.sqrt((fastRate - 1) ** 2 + 4 * mixing * fastRate);
  const fastEigenvalue = -(1 + fastRate + gap) / 2;
  const slowEigenvalue = fastRate * (1 - mixing) / fastEigenvalue;
  const slowWeight = (initialFast - 1 - fastEigenvalue) / gap;
  const fastWeight = 1 - slowWeight;
  const slowMode = slowWeight * Math.exp(slowEigenvalue * time);
  const fastMode = fastWeight * Math.exp(fastEigenvalue * time);
  const slow = slowMode + fastMode;
  const fast = (1 + slowEigenvalue) * slowMode + (1 + fastEigenvalue) * fastMode;
  const reduced = Math.exp(-(1 - mixing) * time);
  const derivativeBound = Math.abs(slowEigenvalue * slowWeight) + Math.abs(fastEigenvalue * fastWeight);
  const initialLayer = Math.exp(-fastRate * time) * Math.abs(initialFast - mixing);
  const memoryLag = mixing * derivativeBound * (-Math.expm1(-fastRate * time)) / fastRate;

  return {
    slow, fast, reduced, slowEigenvalue, fastEigenvalue,
    reductionError: Math.abs(slow - reduced),
    closureResidual: Math.abs(fast - mixing * slow),
    closureBound: initialLayer + memoryLag,
  };
}
