/** Two-site nearest-neighbor chain. Wave number is measured in inverse cell lengths. */
export function calculateElectronicBands(t1: number, t2: number, delta: number, q: number) {
  if (![t1, t2, delta, q].every(Number.isFinite) || t1 < 0 || t2 < 0 || Math.abs(q) > Math.PI) {
    throw new RangeError('Expected nonnegative hopping and a wave number in the first Brillouin zone');
  }
  const edge = Math.abs(q) === Math.PI;
  const halfGap = Math.hypot(delta, t1 - t2);
  const coupling = edge ? 0 : 2 * Math.sqrt(t1) * Math.sqrt(t2) * Math.cos(q / 2);
  const upper = Math.hypot(halfGap, coupling);
  return {
    lower: upper === 0 ? 0 : -upper,
    upper,
    gap: 2 * halfGap,
    upperWeightA: upper === 0 ? null : (1 + delta / upper) / 2,
    upperSlope: upper === 0 ? null : edge || q === 0 ? 0 : -(t1 / upper) * t2 * Math.sin(q),
  };
}
