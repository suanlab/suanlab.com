/** Drawing projections and owned UI state. No exact arithmetic or calculation engine. */
import { readEuclideanKernelInput, type EuclideanKernelInput } from './learning-euclidean-kernel-input';
import type { EuclideanKernelResult } from './learning-euclidean-kernel';
import { scalarFieldProjection as project } from './learning-scalar-field-view';
export type EuclideanKernelMatrixKind = 'position' | 'precision';
export type EuclideanKernelScale = 'linear' | 'signed-log';
export interface EuclideanKernelDisplay { kind: EuclideanKernelMatrixKind; scale: EuclideanKernelScale; selectedTime: number; selectedCell: number; }
export interface EuclideanKernelLearning { step: number; experiment: number; explanationOpen: boolean; answers: readonly number[]; }
export const euclideanKernelDefaultLearning: Readonly<EuclideanKernelLearning> = Object.freeze({ step: 0, experiment: 0, explanationOpen: false, answers: Object.freeze(Array(8).fill(-1) as number[]) });
export function readEuclideanKernelLearning(value: unknown): Readonly<EuclideanKernelLearning> | null {
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
export const euclideanKernelDefaultDisplay: Readonly<EuclideanKernelDisplay> = Object.freeze({ kind: 'position', scale: 'linear', selectedTime: 1, selectedCell: 4 });
export const euclideanKernelMatrixSize = (input: EuclideanKernelInput, kind: EuclideanKernelMatrixKind) => input.intervals + (kind === 'position' ? 1 : -1);
/** A newly applied example highlights its declared physical probe,not an old flat-array index. */
export function euclideanKernelAlignDisplay(input: EuclideanKernelInput, display: EuclideanKernelDisplay) {
  const n=euclideanKernelMatrixSize(input,display.kind),i=display.kind==='position'?input.probe:Math.max(0,Math.min(n-1,input.probe-1));
  return {...display,selectedTime:input.probe,selectedCell:i*(n+1)};
}
/** Keep physical time-index pairs when switching between full and interior matrices. */
export function euclideanKernelChangeMatrix(input: EuclideanKernelInput, display: EuclideanKernelDisplay, kind: EuclideanKernelMatrixKind) {
  const oldN=euclideanKernelMatrixSize(input,display.kind),oldOffset=display.kind==='position'?0:1,n=euclideanKernelMatrixSize(input,kind),offset=kind==='position'?0:1;
  const row=Math.max(0,Math.min(n-1,Math.floor(display.selectedCell/oldN)+oldOffset-offset)),column=Math.max(0,Math.min(n-1,display.selectedCell%oldN+oldOffset-offset));
  return {...display,kind,selectedCell:row*n+column};
}
export function readEuclideanKernelDisplay(value: unknown, input: EuclideanKernelInput): Readonly<EuclideanKernelDisplay> | null {
  try {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
    const d = Object.getOwnPropertyDescriptors(value), keys = ['kind', 'scale', 'selectedTime', 'selectedCell'];
    if (Reflect.ownKeys(d).length !== keys.length || keys.some(k => !d[k] || !Object.hasOwn(d[k], 'value'))) return null;
    const own = Object.fromEntries(keys.map(k => [k, d[k].value]));
    if (!['position', 'precision'].includes(own.kind) || !['linear', 'signed-log'].includes(own.scale)) return null;
    const n = euclideanKernelMatrixSize(input, own.kind);
    if (!Number.isInteger(own.selectedTime) || own.selectedTime < 0 || own.selectedTime > input.intervals || !Number.isInteger(own.selectedCell) || own.selectedCell < 0 || own.selectedCell >= n * n) return null;
    return Object.freeze({ kind: own.kind, scale: own.scale, selectedTime: own.selectedTime, selectedCell: own.selectedCell });
  } catch { return null; }
}
export const euclideanKernelDraft = (input: EuclideanKernelInput) => ({ ...input, intervals: String(input.intervals), probe: String(input.probe) });
export type EuclideanKernelDraft = ReturnType<typeof euclideanKernelDraft>;
export function parseEuclideanKernelDraft(draft: EuclideanKernelDraft) {
  if (!/^(?:[2-9]|[12]\d|3[0-2])$/.test(draft.intervals) || !/^(?:0|[1-9]\d?)$/.test(draft.probe)) return null;
  return readEuclideanKernelInput({ ...draft, intervals: Number(draft.intervals), probe: Number(draft.probe) });
}
export function euclideanKernelMatrixView(result: EuclideanKernelResult, display: EuclideanKernelDisplay) {
  const source = display.kind === 'position' ? euclideanKernelCovariance(result) : result.precision;
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
export type EuclideanKernelMatrixView = ReturnType<typeof euclideanKernelMatrixView>;
/** Lossless dictionary lookup only; the Worker has checked the complete matrix. */
export const euclideanKernelCovariance = (result: EuclideanKernelResult) => result.covariance.indices.map(row => row.map(id => result.covariance.values[Math.floor(id / 128)][id % 128]));
export function euclideanKernelPathView(result: EuclideanKernelResult) {
  const times = result.times.map(project), mean = result.mean.map(project), covariance = euclideanKernelCovariance(result), variance = covariance.map((row, i) => project(row[i]));
  const a = project(result.input.start), b = project(result.input.end), tau = project(result.input.time), omega2 = project(result.input.omegaSquared);
  if (a === null || b === null || tau === null || omega2 === null) return null;
  const base = result.times.map((_, i) => a + (b - a) * i / result.input.intervals), z = Math.sqrt(omega2) * tau;
  const continuous = result.times.map((_, i) => i === 0 ? a : i === result.input.intervals ? b : z === 0 ? base[i] : a * Math.sinh(z * (1 - i / result.input.intervals)) / Math.sinh(z) + b * Math.sinh(z * i / result.input.intervals) / Math.sinh(z));
  if ([...times, ...mean, ...base, ...variance].some(v => v === null)) return null;
  return { times: times as number[], mean: mean as number[], base, continuous, sigma: variance.map(v => Math.sqrt(Math.max(0, v!))) };
}
/** Approximate Mehler diagnostics, separate from the exact checked Worker result.
 * The floating-resolution screen is a heuristic, not an interval certificate.
 */
export function euclideanKernelComparison(result: EuclideanKernelResult) {
  const m = project(result.input.mass), h = project(result.input.hbar), tau = project(result.input.time), w2 = project(result.input.omegaSquared), a = project(result.input.start), b = project(result.input.end);
  if ([m, h, tau, w2, a, b].some(x => x === null)) return null;
  const mass = m!, hbar = h!, time = tau!, omega = Math.sqrt(w2!), z = omega * time, free = w2 === 0;
  // Avoid cosh(z)-1 cancellation in the weak-potential branch.
  const ratio = z === 0 ? 1 : z / Math.sinh(z);
  const action = mass / (2 * hbar) * ((b! - a!) ** 2 / time * ratio + (a! ** 2 + b! ** 2) * omega * Math.tanh(z / 2));
  const logKernel = .5 * (Math.log(mass / (2 * hbar * time)) + Math.log(ratio) - Math.log(Math.PI)) - action;
  const logTrace = free ? null : -Math.log(2) - Math.log(Math.sinh(z / 2));
  const rows = result.refinements.map(row => {
    const p = project(row.prefactorSquaredTimesPi), minimum = project(row.minimumAction), correction = project(row.rightLogCorrection), trace2 = row.traceSquared === null ? null : project(row.traceSquared);
    if (p === null || minimum === null || correction === null || p <= 0) return null;
    const finiteLog = .5 * (Math.log(p) - Math.log(Math.PI)) - minimum, difference = finiteLog - logKernel, resolution = 64 * Number.EPSILON * (1 + Math.abs(finiteLog) + Math.abs(logKernel));
    const rightDifference = difference + correction;
    const symmetric = free ? 0 : Math.abs(difference) > resolution ? difference : null;
    const right = free ? 0 : Math.abs(rightDifference) > resolution ? rightDifference : null;
    const finiteTraceLog = trace2 !== null && trace2 > 0 ? Math.log(trace2) / 2 : null;
    const traceDifference = finiteTraceLog !== null && logTrace !== null ? finiteTraceLog - logTrace : null;
    const traceResolution = finiteTraceLog !== null && logTrace !== null ? 64 * Number.EPSILON * (1 + Math.abs(finiteTraceLog) + Math.abs(logTrace)) : Infinity;
    return { intervals: row.intervals, symmetric, right, finiteLog, resolution, traceLog: finiteTraceLog, traceLogDifference: traceDifference !== null && Math.abs(traceDifference) > traceResolution ? traceDifference : null, exactFreeIdentity: free };
  });
  return { logKernel, logTrace, action, free, rows };
}
export { project as euclideanKernelProjection };
