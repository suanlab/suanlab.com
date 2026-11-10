/** Drawing projections of independently verified output; no numerical engine. */
import type { SpatialSpectrumResult } from './learning-spatial-spectrum-protocol';
import { readSpatialSpectrumInput, type SpatialSpectrumInput } from './learning-spatial-spectrum-input';
import { scalarFieldProjection as project } from './learning-scalar-field-view';
export type SpatialMatrixKind = 'covariance' | 'precision' | 'fourierReal' | 'fourierImaginary';
export type SpatialScale = 'linear' | 'signed-log';
export type SpatialSpectrumDraft = Omit<SpatialSpectrumInput, 'sites' | 'origin' | 'selectedMode'> & { sites: string; origin: string; selectedMode: string };
export const spatialSpectrumDraft = (input: SpatialSpectrumInput): SpatialSpectrumDraft => ({ ...input, sites: String(input.sites), origin: String(input.origin), selectedMode: String(input.selectedMode) });
export function parseSpatialSpectrumDraft(d: SpatialSpectrumDraft) {
  if (![d.sites, d.origin, d.selectedMode].every(v => /^(?:0|[1-9]\d?)$/.test(v))) return null;
  return readSpatialSpectrumInput({ ...d, sites: Number(d.sites), origin: Number(d.origin), selectedMode: Number(d.selectedMode) });
}
export function spatialSpectrumView(r: SpatialSpectrumResult, kind: SpatialMatrixKind, scale: SpatialScale, selected = 0, modeIndex = r.input.selectedMode) {
  let projectionAvailable = true;
  const finite = (s: string) => { const v = project(s); if (v === null) { projectionAvailable = false; return 0; } return v; };
  const interval = (v: { lower: string; upper: string }) => { const lo = finite(v.lower), hi = finite(v.upper); return lo <= 0 && hi >= 0 ? 0 : lo / 2 + hi / 2; };
  const n = r.input.sites, clamp = (v: number, count: number) => Number.isInteger(v) ? Math.max(0, Math.min(count - 1, v)) : 0, entry = clamp(selected, n * n), mode = clamp(modeIndex, n);
  const ranges = kind === 'covariance' || kind === 'precision' ? r[kind].map(row => row.map(v => ({ lower: v, upper: v }))) : r.fourierCovariance.map(row => row.map(v => kind === 'fourierReal' ? v.real : v.imaginary));
  const values = ranges.map(row => row.map(interval)), transform = (v: number) => scale === 'signed-log' ? Math.sign(v) * Math.log1p(Math.abs(v)) / Math.LN10 : v;
  const heights = values.map(row => row.map(transform)), maximum = Math.max(...heights.flat().map(Math.abs)) || 1;
  const distances = r.curve.map(p => finite(p.distance)), covariance = r.curve.map(p => finite(p.covariance));
  const infinite = r.infinite.curve?.map(interval) ?? null, image = r.infinite.periodicImage?.map(interval) ?? null;
  const powers = r.modes.map(m => interval(m.power)), normalizedPowers = r.modes.map(m => interval(m.continuumNormalizedPower)), wavenumbers = r.modes.map(m => interval(m.wavenumber));
  return { projectionAvailable, kind, scale, n, selected: entry, row: Math.floor(entry / n), column: entry % n, mode, ranges, values, heights, maximum, transform, axis: kind.startsWith('fourier') ? 'mode' : 'site', distances, covariance, infinite, image, powers, normalizedPowers, wavenumbers,
    points: heights.flatMap((row, i) => row.map((v, j) => ({ x: 6 * i / (n - 1) - 3, y: 2.5 * v / maximum, z: 6 * j / (n - 1) - 3, value: v, selected: i * n + j === entry }))) };
}
export type SpatialSpectrumView = ReturnType<typeof spatialSpectrumView>;
