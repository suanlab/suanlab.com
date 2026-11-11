/** Explicit Euler and exact unit-frequency oscillator at a fixed dimensionless time. */
export function calculateOscillatorConvergence(steps: number, horizon = 4) {
  if (!Number.isInteger(steps) || steps < 1 || steps > 10000 || !Number.isFinite(horizon) || horizon <= 0 || horizon > 10) {
    throw new RangeError('Expected 1–10000 steps and a horizon in (0, 10]');
  }
  const h = horizon / steps;
  let x = 1, p = 0;
  const trajectory: [number, number][] = [[x, p]];
  for (let i = 0; i < steps; i++) {
    const nextX = x + h * p;
    p -= h * x;
    x = nextX;
    trajectory.push([x, p]);
  }
  const exactX = Math.cos(horizon), exactP = -Math.sin(horizon);
  return {
    h, trajectory, x, p,
    energyRatio: x ** 2 + p ** 2,
    analyticalEnergyRatio: Math.exp(steps * Math.log1p(h ** 2)),
    stateError: Math.hypot(x - exactX, p - exactP),
  };
}
