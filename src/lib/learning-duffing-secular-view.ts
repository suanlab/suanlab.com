/** Display projections of already verified samples. Never solves the oscillator. */
import type { DuffingSecularResult } from './learning-duffing-secular-protocol';
import { scalarFieldProjection as project } from './learning-scalar-field-view';
export type DuffingWindow = 'full' | 'local';
export type DuffingScale = 'linear' | 'signed-log';
export type DuffingMethod = 'exactPosition' | 'naivePosition' | 'firstPosition' | 'secondPosition';
export const duffingMethods: DuffingMethod[] = ['exactPosition', 'naivePosition', 'firstPosition', 'secondPosition'];
export function duffingSecularView(result: DuffingSecularResult, window: DuffingWindow, scale: DuffingScale, showInvalid: boolean) {
 const source = window === 'full' ? result.fullSamples : result.localSamples;
 const midpoint = (r: {lower: string; upper: string}) => (project(r.lower)! + project(r.upper)!) / 2;
 const transform = (v: number) => scale === 'signed-log' ? Math.sign(v) * Math.log1p(Math.abs(v)) / Math.LN10 : v;
 const methods = duffingMethods.filter(k => showInvalid || k !== 'secondPosition' || result.approximations[1].positiveFrequency);
 const points = source.map(p => ({source: p, time: project(p.time)!, values: Object.fromEntries(duffingMethods.map(k => [k, transform(midpoint(p[k]))])) as Record<DuffingMethod, number>}));
 const values = points.flatMap(p => methods.map(k => p.values[k])), low = Math.min(...values), high = Math.max(...values);
 const padding = high === low ? Math.max(1, Math.abs(high) * .05) : (high-low) * .06;
 const start = points[0].time, end = points[64].time;
 return {window, scale, methods, points, yMin: low-padding, yMax: high+padding, start, end,
  // An equilibrium is a constant state, not an oscillation with a resolved period.
  connect: result.mode === 'equilibrium' || window === 'local' || result.horizonDiagnostics.fullGridResolvesEightPointsPerCycle === true,
  x: (index: number) => end === start ? 55 : 55 + 640 * index / 64,
  y: (value: number) => 230 - 200 * (value-low+padding) / (high-low+2*padding),
  phase: result.horizonDiagnostics.phaseDrift ? Object.entries(result.horizonDiagnostics.phaseDrift).map(([method, interval]) => ({method, interval, end: midpoint(interval)})) : null,
 };
}
