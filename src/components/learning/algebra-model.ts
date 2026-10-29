import type { Vec3 } from '@/lib/learning-simulations';
import { applySymmetric2, covarianceRing, leastSquaresState, projectionState, populationCovariance2, regressionInputs, symmetricEigen2 } from '@/lib/learning-linear-algebra';
import type { LabScene, SceneArrow, SceneDot, SceneLine } from './lab-model';
export type AlgebraKind = 'inner-products' | 'least-squares' | 'eigenvalues';
export function isAlgebraKind(kind: string): kind is AlgebraKind { return ['inner-products', 'least-squares', 'eigenvalues'].includes(kind); }
export interface AlgebraSettings {
  u: Vec3; v: Vec3; response: Vec3; slope: number; a: number; b: number; d: number; angle: number; view: 'action' | 'covariance';
}
export const defaultAlgebra: AlgebraSettings = { u: [3, 2, 0], v: [2, 0, 0], response: [1, 2, 2], slope: .5, a: 2, b: 1, d: 2, angle: 0, view: 'action' };
export function validAlgebra(value: unknown): value is AlgebraSettings {
  if (!value || typeof value !== 'object') return false;
  const s = value as AlgebraSettings;
  const vector = (v: unknown) => Array.isArray(v) && v.length === 3 && v.every(x => typeof x === 'number' && Number.isFinite(x) && x >= -5 && x <= 5);
  const between = (x: number, min: number, max: number) => Number.isFinite(x) && x >= min && x <= max;
  return vector(s.u) && vector(s.v) && vector(s.response) && between(s.slope, -3, 3)
    && between(s.a, 0, 4) && between(s.d, 0, 4) && between(s.b, -2, 2) && between(s.angle, -90, 180) && ['action', 'covariance'].includes(s.view);
}
export const algebraPresets: Record<AlgebraKind, Partial<AlgebraSettings>[]> = {
  'inner-products': [
    { u: [3, 2, 0], v: [2, 0, 0] }, { u: [3, 0, 0], v: [0, 2, 0] },
    { u: [-3, 0, 0], v: [2, 0, 0] }, { u: [3, 2, 0], v: [0, 0, 0] },
  ],
  'least-squares': [
    { response: [1, 2, 2], slope: .5 }, { response: [1, 2, 2], slope: 11 / 14 },
    { response: [1, 2, 3], slope: 1 }, { response: [1, 2, 5], slope: 10 / 7 },
  ],
  eigenvalues: [
    { a: 2, b: 1, d: 2, angle: 45, view: 'action' }, { a: 2, b: 1, d: 2, angle: 45, view: 'covariance' },
    { a: 2, b: 0, d: 2, angle: 0, view: 'covariance' }, { a: 1, b: 2, d: 1, angle: 45, view: 'action' },
    { a: 0, b: 0, d: 0, angle: 0, view: 'covariance' },
  ],
};
export function algebraValues(s: AlgebraSettings) {
  const radians = s.angle * Math.PI / 180; const direction: [number, number] = [Math.cos(radians), Math.sin(radians)];
  const mapped = applySymmetric2(s.a, s.b, s.d, direction);
  const ring = covarianceRing(s.a, s.b, s.d);
  return { ring, sample: ring ? populationCovariance2(ring) : null, projection: projectionState(s.u, s.v), regression: leastSquaresState(s.response, s.slope), eigen: symmetricEigen2(s.a, s.b, s.d),
    direction, mapped, cross: direction[0] * mapped[1] - direction[1] * mapped[0], rayleigh: direction[0] * mapped[0] + direction[1] * mapped[1] };
}
export function buildAlgebraScene(kind: AlgebraKind, s: AlgebraSettings): LabScene {
  const lines: SceneLine[] = []; const arrows: SceneArrow[] = []; const dots: SceneDot[] = []; const zero: Vec3 = [0, 0, 0]; const result = algebraValues(s);
  if (kind === 'inner-products') {
    arrows.push({ start: zero, end: s.u, color: 'blue' }, { start: zero, end: s.v, color: 'orange' });
    if (result.projection.projected) {
      arrows.push({ start: zero, end: result.projection.projected, color: 'green' });
      lines.push({ points: [result.projection.projected, s.u], color: 'neutral', dashed: true });
    }
  } else if (kind === 'least-squares') {
    arrows.push({ start: zero, end: s.response, color: 'blue' }, { start: zero, end: regressionInputs, color: 'orange' },
      { start: zero, end: result.regression.predicted, color: 'green' });
    lines.push({ points: [result.regression.predicted, s.response], color: 'neutral', dashed: true });
  } else {
    const e = result.eigen;
    const ring = s.view === 'covariance' ? result.ring : null;
    if (s.view === 'action') {
      const circle: Vec3[] = Array.from({ length: 65 }, (_, i) => [Math.cos(i / 64 * 2 * Math.PI), Math.sin(i / 64 * 2 * Math.PI), 0]);
      lines.push({ points: circle, color: 'neutral', dashed: true }, { points: circle.map(p => [...applySymmetric2(s.a, s.b, s.d, [p[0], p[1]]), 0]), color: 'green' });
      arrows.push({ start: zero, end: [...result.direction, 0], color: 'blue' }, { start: zero, end: [...result.mapped, 0], color: 'orange' });
    } else if (ring) {
      lines.push({ points: Array.from({ length: 65 }, (_, i) => {
        const theta = i / 64 * 2 * Math.PI; const u = Math.sqrt(Math.max(0, e.values[0])) * Math.cos(theta); const v = Math.sqrt(Math.max(0, e.values[1])) * Math.sin(theta);
        return [u * e.first[0] + v * e.second[0], u * e.first[1] + v * e.second[1], 0];
      }), color: 'green' });
      ring.forEach(p => dots.push({ position: [...p, 0], color: 'blue', radius: .06 }));
      [e.first, e.second].forEach((axis, i) => {
        const radius = Math.sqrt(Math.max(0, e.values[i]));
        lines.push({ points: [[-axis[0] * radius, -axis[1] * radius, 0], [axis[0] * radius, axis[1] * radius, 0]], color: i === 0 ? 'green' : 'orange' });
      });
    }
  }
  const points = [...lines.flatMap(l => l.points), ...arrows.flatMap(a => [a.start, a.end]), ...dots.map(d => d.position)];
  const extent = Math.max(2, ...points.flat().map(Math.abs)); const scale = (p: Vec3): Vec3 => p.map(v => v * 3.5 / extent) as Vec3;
  return { kind, gridUnit: extent / 3.5, heightScale: 1,
    axisLabels: kind === 'least-squares' ? ['obs₁', 'obs₂', 'obs₃'] : undefined,
    lines: lines.map(l => ({ ...l, points: l.points.map(scale) })), arrows: arrows.map(a => ({ ...a, start: scale(a.start), end: scale(a.end) })), dots: dots.map(d => ({ ...d, position: scale(d.position) })) };
}
