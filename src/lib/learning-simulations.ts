export type Vec3 = [number, number, number];
export type Vec2 = [number, number];
export type Mat3 = [Vec3, Vec3, Vec3];

function finite(values: number[]) {
  if (!values.every(Number.isFinite)) throw new Error('Model inputs must be finite.');
}
export function add3(a: Vec3, b: Vec3): Vec3 { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
export function scale3(a: Vec3, factor: number): Vec3 { return [a[0] * factor, a[1] * factor, a[2] * factor]; }
export function norm3(a: Vec3) { return Math.hypot(...a); }
export function dot3(a: Vec3, b: Vec3) { return a.reduce((sum, x, i) => sum + x * b[i], 0); }
export function particleState(force1: Vec3, force2: Vec3, mass: number, time: number, velocity0: Vec3 = [0, 0, 0], position0: Vec3 = [0, 0, 0]) {
  finite([...force1, ...force2, mass, time, ...velocity0, ...position0]);
  if (mass <= 0 || time < 0) throw new Error('Mass must be positive and time non-negative.');
  const force = add3(force1, force2);
  const acceleration = scale3(force, 1 / mass);
  const velocity = add3(velocity0, scale3(acceleration, time));
  const displacement = add3(scale3(velocity0, time), scale3(acceleration, .5 * time * time));
  return { force, magnitude: norm3(force), acceleration, velocity, position: add3(position0, displacement),
    kineticEnergy: .5 * mass * dot3(velocity, velocity), work: dot3(force, displacement) };
}
export function transform3(matrix: Mat3, point: Vec3): Vec3 {
  finite([...matrix.flat(), ...point]);
  return matrix.map(row => dot3(row, point)) as Vec3;
}
export function determinant3(m: Mat3) {
  finite(m.flat());
  return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
    - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
    + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}
export function quadratic(a: number, b: number, point: Vec2) {
  finite([a, b, ...point]);
  if (a <= 0 || b <= 0) throw new Error('Curvatures must be positive.');
  return { loss: .5 * (a * point[0] ** 2 + b * point[1] ** 2), gradient: [a * point[0], b * point[1]] as Vec2 };
}
export function gradientStep(a: number, b: number, point: Vec2, rate: number): Vec2 {
  finite([rate]);
  if (rate <= 0) throw new Error('Learning rate must be positive.');
  const { gradient } = quadratic(a, b, point);
  const next: Vec2 = [point[0] - rate * gradient[0], point[1] - rate * gradient[1]];
  finite(next);
  return next;
}
export function gradientPath(a: number, b: number, start: Vec2, rate: number, steps: number): Vec2[] {
  if (!Number.isInteger(steps) || steps < 0 || steps > 50) throw new Error('Steps must be an integer from 0 to 50.');
  quadratic(a, b, start); finite([rate]);
  if (rate <= 0) throw new Error('Learning rate must be positive.');
  const points: Vec2[] = [[...start]];
  for (let i = 0; i < steps; i++) {
    const next = gradientStep(a, b, points[points.length - 1], rate);
    points.push(next);
    if (Math.max(...next.map(Math.abs)) > 1000) break;
  }
  return points;
}
export function formatNumber(value: number) {
  if (!Number.isFinite(value)) return '—';
  if (value === 0) return '0';
  if (Math.abs(value) < .001 || Math.abs(value) >= 1e6) return value.toExponential(3).replace('e+', 'e');
  return value.toLocaleString('en-US', { maximumFractionDigits: 3 });
}
