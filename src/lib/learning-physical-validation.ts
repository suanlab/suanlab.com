type Ports = readonly [number, number];

/** Two-port reference map with common gain and symmetric readout clipping. */
export function calculatePhysicalValidation(input: Ports, gain: number, limit: number) {
  if (![...input, gain, limit].every(Number.isFinite) || gain <= 0 || limit <= 0) {
    throw new RangeError('Expected finite inputs, positive gain, and a positive readout limit');
  }
  const reference = ([x, y]: Ports): Ports => [.75 * x + .25 * y, .25 * x + .75 * y];
  const measure = (ports: Ports): Ports => {
    const [a, b] = reference(ports);
    const clip = (value: number) => Math.max(-limit, Math.min(limit, gain * value));
    return [clip(a), clip(b)];
  };
  const target = reference(input);
  const measured = measure(input);
  const first = measure([1, 0]), second = measure([0, 1]);
  const fitted: Ports = [
    input[0] * first[0] + input[1] * second[0],
    input[0] * first[1] + input[1] * second[1],
  ];
  const residual: Ports = [measured[0] - fitted[0], measured[1] - fitted[1]];
  return {
    target, measured, fitted, residual,
    absoluteError: Math.hypot(measured[0] - target[0], measured[1] - target[1]),
    calibratedError: Math.hypot(...residual),
  };
}
