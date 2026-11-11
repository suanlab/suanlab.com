/** Positive exponential cooling in excess-temperature coordinates. */
export function predictCooling(initial: number, rate: number, time: number) {
  if (![initial, rate, time].every(Number.isFinite) || initial <= 0 || rate <= 0 || time < 0) {
    throw new RangeError('Expected positive initial excess and rate, and nonnegative time');
  }
  return initial * Math.exp(-rate * time);
}

export function fitOneCoolingReading(excess: number, time: number, rate: number) {
  if (![excess, time, rate].every(Number.isFinite) || excess <= 0 || time < 0 || rate <= 0) {
    throw new RangeError('Expected a positive reading and rate, and nonnegative time');
  }
  return { initial: excess * Math.exp(rate * time), rate };
}

export function fitTwoCoolingReadings(first: number, second: number, firstTime: number, secondTime: number) {
  if (![first, second, firstTime, secondTime].every(Number.isFinite) || firstTime < 0 || secondTime <= firstTime) {
    throw new RangeError('Expected finite readings at increasing nonnegative times');
  }
  if (first <= second || second <= 0) return null;
  const relativeDrop = (first - second) / second;
  const logRatio = Number.isFinite(relativeDrop) ? Math.log1p(relativeDrop) : Math.log(first) - Math.log(second);
  return fitOneCoolingReading(first, firstTime, logRatio / (secondTime - firstTime));
}
