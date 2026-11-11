/** Summarize paired energy observations and a shared fractional meter uncertainty. */
export function calculatePairedEnergy(a: readonly number[], b: readonly number[], meterUncertainty: number) {
  if (a.length < 2 || a.length !== b.length || !a.every(Number.isFinite) || !b.every(Number.isFinite) ||
      !Number.isFinite(meterUncertainty) || meterUncertainty < 0) {
    throw new RangeError('Expected at least two aligned observations and nonnegative meter uncertainty');
  }
  let mean = 0;
  let sumSquares = 0;
  const differences = a.map((value, i) => {
    const difference = value - b[i];
    const deviation = difference - mean;
    mean += deviation / (i + 1);
    sumSquares += deviation * (difference - mean);
    return difference;
  });
  return {
    differences,
    mean,
    standardError: Math.sqrt(sumSquares / (a.length - 1) / a.length),
    calibrationUncertainty: Math.abs(mean) * meterUncertainty,
  };
}
