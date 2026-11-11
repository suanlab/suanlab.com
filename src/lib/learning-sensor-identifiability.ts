/** Two mixed observations y1 = a + b and y2 = a + r b. */
export function calculateSensorIdentifiability(r: number, sigma: number, firstError: number, secondError: number) {
  if (![r, sigma, firstError, secondError].every(Number.isFinite) || r < -1 || r > 1 || sigma < 0) {
    throw new RangeError('Expected sensor coefficient in [-1, 1] and nonnegative noise standard deviation');
  }
  const first = 5 + firstError;
  const second = 2 + 3 * r + secondError;
  if (r === 1) {
    return { first, second, state: first === second ? 'nonunique' : 'inconsistent', estimate: null, bDeviation: null } as const;
  }
  const gap = 1 - r;
  const b = 3 + (firstError - secondError) / gap;
  const a = first - b;
  return { first, second, state: 'unique', estimate: { a, b }, bDeviation: Math.SQRT2 * sigma / gap } as const;
}
