/** Schwarzschild circular clock at or outside the marginally stable radius, in G=c=M=1. */
export function calculateWeakFieldClock(radius: number) {
  if (!Number.isFinite(radius) || radius < 6) throw new RangeError('Expected circular radius r/M ≥ 6');
  const u = 1 / radius;
  const a = 3 * u;
  const exact = Math.sqrt(1 - a);
  const linear = 1 - a / 2;
  return {
    exact, linear,
    error: a ** 2 / (4 * (linear + exact)),
    staticRate: Math.sqrt(1 - 2 * u),
    localSpeed: Math.sqrt(u / (1 - 2 * u)),
    angularFrequency: u ** 1.5,
  };
}
