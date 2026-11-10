/** Drawing projections and owned UI state. No exact arithmetic or calculation engine. */
import { readBrownianBridgeInput, type BrownianBridgeInput } from './learning-brownian-bridge-input';
import type { BrownianBridgeResult } from './learning-brownian-bridge';
import { scalarFieldProjection as project } from './learning-scalar-field-view';
export type BrownianBridgeMatrixKind = 'position' | 'increment' | 'precision';
export type BrownianBridgeScale = 'linear' | 'signed-log';
export interface BrownianBridgeDisplay { kind: BrownianBridgeMatrixKind; scale: BrownianBridgeScale; selectedTime: number; selectedCell: number; }
export interface BrownianBridgeLearning { step: number; experiment: number; explanationOpen: boolean; answers: readonly number[]; }
export const brownianBridgeDefaultLearning: Readonly<BrownianBridgeLearning> = Object.freeze({ step: 0, experiment: 0, explanationOpen: false, answers: Object.freeze(Array(8).fill(-1) as number[]) });
export function readBrownianBridgeLearning(value: unknown): Readonly<BrownianBridgeLearning> | null {
  try {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
    const d = Object.getOwnPropertyDescriptors(value), keys = ['step', 'experiment', 'explanationOpen', 'answers'];
    if (Reflect.ownKeys(d).length !== 4 || keys.some(k => !d[k] || !Object.hasOwn(d[k], 'value'))) return null;
    const own = Object.fromEntries(keys.map(k => [k, d[k].value]));
    if (!Number.isInteger(own.step) || own.step < 0 || own.step > 7 || !Number.isInteger(own.experiment) || own.experiment < 0 || own.experiment > 5 || typeof own.explanationOpen !== 'boolean' || !Array.isArray(own.answers)) return null;
    const a = Object.getOwnPropertyDescriptors(own.answers) as unknown as Record<string, PropertyDescriptor>, answers: number[] = [];
    if (Reflect.ownKeys(a).length !== 9 || !Object.hasOwn(a.length, 'value') || a.length.value !== 8) return null;
    for (let i = 0; i < 8; i++) { const item = a[String(i)]; if (!item || !Object.hasOwn(item, 'value') || !Number.isInteger(item.value) || item.value < -1 || item.value > 2) return null; answers.push(item.value); }
    return Object.freeze({ step: own.step, experiment: own.experiment, explanationOpen: own.explanationOpen, answers: Object.freeze(answers) });
  } catch { return null; }
}
export const brownianBridgeDefaultDisplay: Readonly<BrownianBridgeDisplay> = Object.freeze({ kind: 'position', scale: 'linear', selectedTime: 2, selectedCell: 12 });
export const brownianBridgeMatrixSize = (input: BrownianBridgeInput, kind: BrownianBridgeMatrixKind) => input.intervals + (kind === 'position' ? 1 : kind === 'precision' ? -1 : 0);
export function readBrownianBridgeDisplay(value: unknown, input: BrownianBridgeInput): Readonly<BrownianBridgeDisplay> | null {
  try {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
    const d = Object.getOwnPropertyDescriptors(value), keys = ['kind', 'scale', 'selectedTime', 'selectedCell'];
    if (Reflect.ownKeys(d).length !== keys.length || keys.some(k => !d[k] || !Object.hasOwn(d[k], 'value'))) return null;
    const own = Object.fromEntries(keys.map(k => [k, d[k].value]));
    if (!['position', 'increment', 'precision'].includes(own.kind) || !['linear', 'signed-log'].includes(own.scale)) return null;
    const n = brownianBridgeMatrixSize(input, own.kind);
    if (!Number.isInteger(own.selectedTime) || own.selectedTime < 0 || own.selectedTime > input.intervals || !Number.isInteger(own.selectedCell) || own.selectedCell < 0 || own.selectedCell >= n * n) return null;
    return Object.freeze({ kind: own.kind, scale: own.scale, selectedTime: own.selectedTime, selectedCell: own.selectedCell });
  } catch { return null; }
}
export const brownianBridgeDraft = (input: BrownianBridgeInput) => ({ ...input, intervals: String(input.intervals), probe: String(input.probe) });
export type BrownianBridgeDraft = ReturnType<typeof brownianBridgeDraft>;
export function parseBrownianBridgeDraft(draft: BrownianBridgeDraft) {
  if (!/^(?:[2-9]|[12]\d|3[0-2])$/.test(draft.intervals) || !/^(?:0|[1-9]\d?)$/.test(draft.probe)) return null;
  return readBrownianBridgeInput({ ...draft, intervals: Number(draft.intervals), probe: Number(draft.probe) });
}
export function brownianBridgeMatrixView(result: BrownianBridgeResult, display: BrownianBridgeDisplay) {
  const source = display.kind === 'position' ? result.covariance : display.kind === 'increment' ? result.increments.covariance : result.precision;
  const n = source.length, selected = Math.max(0, Math.min(n * n - 1, display.selectedCell)), row = Math.floor(selected / n), column = selected % n;
  let projectionAvailable = true;
  const values = source.map(r => r.map(value => { const v = project(value); if (v === null) projectionAvailable = false; return v ?? 0; }));
  const transform = (v: number) => display.scale === 'signed-log' ? Math.sign(v) * Math.log1p(Math.abs(v)) / Math.LN10 : v;
  const heights = values.map(r => r.map(transform)), maximum = Math.max(...heights.flat().map(Math.abs)) || 1;
  return { n, selected, row, column, kind: display.kind, scale: display.scale, source, values, heights, maximum, projectionAvailable,
    labels: source.map((_, i) => display.kind === 'position' ? i : i + 1),
    points: heights.flatMap((r, i) => r.map((value, j) => ({ x: n === 1 ? 0 : 6 * i / (n - 1) - 3, z: n === 1 ? 0 : 6 * j / (n - 1) - 3, y: 2.5 * value / maximum, value, selected: i === row && j === column }))),
  };
}
export type BrownianBridgeMatrixView = ReturnType<typeof brownianBridgeMatrixView>;
export function brownianBridgePathView(result: BrownianBridgeResult) {
  const times = result.times.map(project), mean = result.mean.map(project), base = result.baselineMean.map(project), variance = result.covariance.map((row, i) => project(row[i]));
  if ([...times, ...mean, ...base, ...variance].some(v => v === null)) return null;
  return { times: times as number[], mean: mean as number[], base: base as number[], sigma: variance.map(v => Math.sqrt(Math.max(0, v!))) };
}
export { project as brownianBridgeProjection };
