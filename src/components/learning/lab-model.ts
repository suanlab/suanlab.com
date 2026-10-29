import type { LearningUnitId } from '@/data/interactive-book';
import { determinant3, gradientPath, particleState, quadratic, transform3, type Mat3, type Vec2, type Vec3 } from '@/lib/learning-simulations';

export type LabKind = 'vectors-and-forces' | 'linear-transforms' | 'gradients';
export interface LabSettings {
  force1: Vec3; force2: Vec3; velocity0: Vec3; mass: number; time: number; mode: 'forces' | 'motion';
  matrix: Mat3; point: Vec3; a: number; b: number; start: Vec2; rate: number; steps: number;
}
export const defaultLab: LabSettings = {
  force1: [3, 0, 0], force2: [0, 4, 0], velocity0: [0, 0, 0], mass: 2, time: 0, mode: 'forces',
  matrix: [[2, 0, 0], [0, .5, 0], [0, 0, 1]], point: [1, 2, 3], a: 2, b: 1, start: [2, 2], rate: .25, steps: 0,
};
function vector(value: unknown, size: number, min: number, max: number): boolean {
  return Array.isArray(value) && value.length === size && value.every(v => typeof v === 'number' && Number.isFinite(v) && v >= min && v <= max);
}
export function validLab(value: unknown): value is LabSettings {
  if (!value || typeof value !== 'object') return false;
  const s = value as LabSettings;
  return vector(s.force1, 3, -5, 5) && vector(s.force2, 3, -5, 5) && vector(s.velocity0, 3, -5, 5)
    && vector([s.mass], 1, .5, 5) && vector([s.time], 1, 0, 5) && ['forces', 'motion'].includes(s.mode)
    && Array.isArray(s.matrix) && s.matrix.length === 3 && s.matrix.every(row => vector(row, 3, -3, 3))
    && vector(s.point, 3, -3, 3) && vector([s.a, s.b], 2, .5, 5) && vector(s.start, 2, -3, 3)
    && vector([s.rate], 1, .05, 2) && Number.isInteger(s.steps) && s.steps >= 0 && s.steps <= 50;
}
export const matrixPresets: Mat3[] = [
  [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[2, 0, 0], [0, .5, 0], [0, 0, 1]],
  [[0, -1, 0], [1, 0, 0], [0, 0, 1]], [[1, 1, 0], [0, 1, 0], [0, 0, 1]],
  [[-1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 0, 0], [0, 0, 1]],
];
export function labValues(s: LabSettings) {
  const particle = particleState(s.force1, s.force2, s.mass, s.time, s.velocity0);
  const path = gradientPath(s.a, s.b, s.start, s.rate, s.steps);
  const current = path[path.length - 1];
  return { particle, transformed: transform3(s.matrix, s.point), determinant: determinant3(s.matrix), path, current,
    objective: quadratic(s.a, s.b, current), stopped: Math.max(...current.map(Math.abs)) > 1000, limit: 2 / Math.max(s.a, s.b) };
}
export type LabColor = 'blue' | 'orange' | 'green' | 'neutral';
export interface SceneLine { points: Vec3[]; color: LabColor; dashed?: boolean; }
export interface SceneArrow { start: Vec3; end: Vec3; color: LabColor; dashed?: boolean; }
export interface SceneDot { position: Vec3; color: LabColor; radius?: number; }
export interface LabScene {
  lines: SceneLine[]; arrows: SceneArrow[]; dots: SceneDot[]; surface?: { vertices: number[]; indices: number[] };
  gridUnit: number; heightScale: number; kind: LearningUnitId; axisLabels?: [string, string, string];
}
const colors: LabColor[] = ['blue', 'orange', 'green'];
export function buildLabScene(kind: LabKind, s: LabSettings): LabScene {
  const lines: SceneLine[] = []; const arrows: SceneArrow[] = []; const dots: SceneDot[] = [];
  const zero: Vec3 = [0, 0, 0]; let heightScale = 1; let surface: LabScene['surface'];
  const result = labValues(s);
  if (kind === 'vectors-and-forces') {
    if (s.mode === 'forces') {
      [s.force1, s.force2, result.particle.force].forEach((v, i) => arrows.push({ start: zero, end: v, color: colors[i] }));
      arrows.push({ start: s.force1, end: result.particle.force, color: colors[1], dashed: true });
    } else {
      lines.push({ points: Array.from({ length: 51 }, (_, i) => particleState(s.force1, s.force2, s.mass, i / 10, s.velocity0).position), color: colors[2] });
      dots.push({ position: zero, color: 'neutral', radius: .08 }, { position: result.particle.position, color: colors[1], radius: .18 });
    }
  } else if (kind === 'linear-transforms') {
    const vertices: Vec3[] = Array.from({ length: 8 }, (_, i) => [i & 1 ? 1 : 0, i & 2 ? 1 : 0, i & 4 ? 1 : 0]);
    vertices.forEach((v, i) => [1, 2, 4].forEach(bit => {
      if (!(i & bit)) {
        lines.push({ points: [v, vertices[i | bit]], color: 'neutral', dashed: true });
        lines.push({ points: [transform3(s.matrix, v), transform3(s.matrix, vertices[i | bit])], color: colors[2] });
      }
    }));
    ([ [1, 0, 0], [0, 1, 0], [0, 0, 1] ] as Vec3[]).forEach((basis, i) => arrows.push({ start: zero, end: transform3(s.matrix, basis), color: colors[i] }));
    dots.push({ position: s.point, color: 'neutral' }, { position: result.transformed, color: colors[1], radius: .15 });
  } else {
    const range = Math.max(2.5, ...result.path.flat().map(v => Math.abs(v) * 1.15));
    heightScale = 3.5 / quadratic(s.a, s.b, [range, range]).loss;
    const project = (p: Vec2): Vec3 => [p[0] / range * 3.5, quadratic(s.a, s.b, p).loss * heightScale, p[1] / range * 3.5];
    const vertices: number[] = []; const indices: number[] = []; const n = 24;
    for (let i = 0; i <= n; i++) for (let j = 0; j <= n; j++) vertices.push(...project([(i / n * 2 - 1) * range, (j / n * 2 - 1) * range]));
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const k = i * (n + 1) + j; indices.push(k, k + 1, k + n + 1, k + 1, k + n + 2, k + n + 1);
    }
    surface = { vertices, indices };
    lines.push({ points: result.path.map(project), color: 'orange' });
    dots.push({ position: project(result.current), color: 'orange', radius: .15 }, { position: zero, color: colors[2] });
    // The arrow shows the first descent direction in parameter space, not a physical velocity.
    const g = result.objective.gradient; const length = Math.hypot(...g);
    if (length > 1e-9) {
      const p = project(result.current);
      arrows.push({ start: p, end: [p[0] - g[0] / length, p[1], p[2] - g[1] / length], color: 'blue' });
    }
    return { lines, arrows, dots, surface, gridUnit: range / 3.5, heightScale, kind };
  }
  const coordinates = [...lines.flatMap(l => l.points), ...arrows.flatMap(a => [a.start, a.end]), ...dots.map(d => d.position)];
  const bound = Math.max(2, ...coordinates.flat().map(Math.abs)); const factor = 3.5 / bound;
  const scale = (p: Vec3): Vec3 => p.map(v => v * factor) as Vec3;
  return { lines: lines.map(l => ({ ...l, points: l.points.map(scale) })), arrows: arrows.map(a => ({ ...a, start: scale(a.start), end: scale(a.end) })),
    dots: dots.map(d => ({ ...d, position: scale(d.position) })), gridUnit: 1 / factor, heightScale, kind };
}
