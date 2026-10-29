import { add3, dot3, norm3, scale3, type Vec2, type Vec3 } from './learning-simulations';
function finite(values: number[]) { if (!values.every(Number.isFinite)) throw new Error('Inputs must be finite.'); }

export function projectionState(u: Vec3, v: Vec3) {
  finite([...u, ...v]); const dot = dot3(u, v); const lengthU = norm3(u); const lengthV = norm3(v);
  const unitV = lengthV === 0 ? null : v.map(x => x / lengthV) as Vec3;
  const unitU = lengthU === 0 ? null : u.map(x => x / lengthU) as Vec3;
  const cosine = unitU && unitV ? Math.max(-1, Math.min(1, dot3(unitU, unitV))) : null;
  const along = unitV ? dot3(u, unitV) : null;
  const coefficient = along === null ? null : along / lengthV;
  const projected = unitV && along !== null ? scale3(unitV, along) : null;
  const residual = projected ? add3(u, scale3(projected, -1)) : null;
  return { dot, lengthU, lengthV, cosine, angle: cosine === null ? null : Math.acos(cosine) * 180 / Math.PI,
    coefficient, projected, residual, residualDot: residual ? dot3(v, residual) : null };
}
export const regressionInputs: Vec3 = [1, 2, 3];
/** Intercept-free regression; observations are axes in response space, not spatial positions. */
export function leastSquaresState(response: Vec3, slope: number) {
  finite([...response, slope]); const x = regressionInputs;
  const predicted = scale3(x, slope); const residual = add3(response, scale3(predicted, -1));
  const optimalSlope = dot3(x, response) / dot3(x, x);
  const optimalPredicted = scale3(x, optimalSlope); const optimalResidual = add3(response, scale3(optimalPredicted, -1));
  return { predicted, residual, squaredError: dot3(residual, residual), loss: .5 * dot3(residual, residual),
    gradient: -dot3(x, residual), normalResidual: dot3(x, residual), optimalSlope, optimalPredicted,
    optimalLoss: .5 * dot3(optimalResidual, optimalResidual) };
}
export function symmetricEigen2(a: number, b: number, d: number) {
  finite([a, b, d]); const gap = Math.hypot(a - d, 2 * b); const trace = a + d;
  const values: Vec2 = [(trace + gap) / 2, (trace - gap) / 2];
  const angle = .5 * Math.atan2(2 * b, a - d);
  const first: Vec2 = [Math.cos(angle), Math.sin(angle)]; const second: Vec2 = [-first[1], first[0]];
  // The tolerance handles roundoff in an analytically positive semidefinite matrix.
  const tolerance = 1e-12 * Math.max(1, ...values.map(Math.abs));
  const positiveSemidefinite = values[1] >= -tolerance;
  const nonnegative = values.map(v => Math.max(0, v)); const total = nonnegative[0] + nonnegative[1];
  return { values, first, second, angle: angle * 180 / Math.PI, repeated: gap === 0, positiveSemidefinite,
    varianceRatio: positiveSemidefinite && total > 0 ? nonnegative[0] / total : null };
}
export function applySymmetric2(a: number, b: number, d: number, point: Vec2): Vec2 {
  finite([a, b, d, ...point]); return [a * point[0] + b * point[1], b * point[0] + d * point[1]];
}
/** Deterministic ring, not Gaussian samples. Population covariance (denominator n) equals C. */
export function covarianceRing(a: number, b: number, d: number, count = 32): Vec2[] | null {
  if (!Number.isInteger(count) || count < 4 || count > 128) throw new Error('Use 4–128 points.');
  const e = symmetricEigen2(a, b, d); if (!e.positiveSemidefinite) return null;
  return Array.from({ length: count }, (_, i) => {
    const phase = i / count * 2 * Math.PI;
    const x = Math.sqrt(2 * Math.max(0, e.values[0])) * Math.cos(phase);
    const y = Math.sqrt(2 * Math.max(0, e.values[1])) * Math.sin(phase);
    return [x * e.first[0] + y * e.second[0], x * e.first[1] + y * e.second[1]];
  });
}

export function populationCovariance2(points: Vec2[]) {
  if (!points.length) throw new Error('At least one point is required.');
  finite(points.flat());
  const n = points.length; const mean: Vec2 = [points.reduce((s, p) => s + p[0], 0) / n, points.reduce((s, p) => s + p[1], 0) / n];
  const xx = points.reduce((s, p) => s + (p[0] - mean[0]) ** 2, 0) / n;
  const yy = points.reduce((s, p) => s + (p[1] - mean[1]) ** 2, 0) / n;
  const xy = points.reduce((s, p) => s + (p[0] - mean[0]) * (p[1] - mean[1]), 0) / n;
  return { mean, matrix: [[xx, xy], [xy, yy]] as [Vec2, Vec2] };
}
