/** Approximate drawings only; all certificates come from verified fractions. */
import type { QuarticAsymptoticResult } from './learning-quartic-asymptotic-protocol';
import { scalarFieldProjection } from './learning-scalar-field-view';
import { perturbationLogMagnitude } from './learning-regular-perturbation-view';
export type QuarticPartialScale = 'physical' | 'all';
/** The zero-coupling axis is degenerate, not an artificial extrusion. */
export const quarticSurfaceDepth = (coupling: number, fraction: number) => coupling === 0 ? 0 : 4 * fraction - 2;
export function quarticAsymptoticView(result: QuarticAsymptoticResult, order: number, width: number, partialScale: QuarticPartialScale) {
  const coupling = scalarFieldProjection(result.input.coupling)!, chosen = result.approximations[Math.max(0, Math.min(result.input.order, order))];
  const integrand = (x: number, g = coupling) => Math.exp(-x * x / 2 - g * x ** 4 / 4);
  const samples = Array.from({ length: 241 }, (_, i) => { const x = width * (2 * i / 240 - 1); return { x, value: integrand(x), gaussian: integrand(x, 0), logValue: -(x * x / 2 + coupling * x ** 4 / 4) / Math.LN10 }; });
  const surface = Array.from({ length: 21 }, (_, j) => Array.from({ length: 41 }, (_, i) => { const x = width * (2 * i / 40 - 1); return { x, value: integrand(x, coupling * j / 20) }; }));
  const partialTransform = (value: number) => partialScale === 'physical' ? value : Math.sign(value) * Math.log1p(Math.abs(value)) / Math.LN10;
  const rows = result.approximations.map(row => ({ order: row.order, partial: scalarFieldProjection(row.partial)!,
    term: perturbationLogMagnitude(row.term), bound: perturbationLogMagnitude(row.firstOmittedMagnitude), distance: perturbationLogMagnitude(row.quadratureDistance.upper), error: perturbationLogMagnitude(row.errorInterval.upper) }));
  const logs = rows.flatMap(row => [row.term, row.bound, row.distance, row.error]).filter((value): value is number => value !== null);
  const logMin = logs.length ? Math.floor(Math.min(...logs)) - 1 : -1, logMax = logs.length ? Math.max(logMin + 1, Math.ceil(Math.max(...logs))) : 1;
  const lower = scalarFieldProjection(result.quadratureInterval.lower)!, upper = scalarFieldProjection(result.quadratureInterval.upper)!;
  const yMin = partialScale === 'physical' ? 0 : Math.min(0, ...rows.map(row => partialTransform(row.partial)));
  const yMax = partialScale === 'physical' ? 1.05 : Math.max(1, ...rows.map(row => partialTransform(row.partial)));
  const partial = scalarFieldProjection(chosen.partial)!;
  return { coupling, chosen, partial, width, samples, surface, rows, logMin, logMax, lower, upper, partialScale, partialTransform, yMin, yMax,
    outsidePhysicalWindow: rows.filter(row => row.partial < 0 || row.partial > 1.05).length,
    selectedOutsidePhysicalWindow: partialScale === 'physical' && (partial < 0 || partial > 1.05),
    intervals: [result.quadratureInterval, result.asymptoticEnvelope, result.combinedInterval].map(range => ({ lower: scalarFieldProjection(range.lower)!, upper: scalarFieldProjection(range.upper)! })) };
}
export type QuarticAsymptoticView = ReturnType<typeof quarticAsymptoticView>;
