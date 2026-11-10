/** Drawing projections of independently checked strings. No Gaussian engine here. */
import { readThermalFieldInput, type ThermalFieldInput, type ThermalJetInput } from './learning-thermal-field-input';
import type { ThermalFieldExact } from './learning-thermal-field-protocol';
import { scalarFieldProjection } from './learning-scalar-field-view';
export type ThermalNormal = Extract<ThermalFieldExact, { kind: 'normalizable' }>;
export type ThermalMatrix = 'covariance' | 'inverseOnSpace' | 'precision';
export type ThermalScale = 'linear' | 'signed-log';
export interface ThermalDraft {
  shape: string; boundary: ThermalFieldInput['boundary']; constraint: ThermalFieldInput['constraint'];
  spacing: string; temperature: string; referenceScale: string;
  mass: ThermalJetInput; stiffness: ThermalJetInput; backgroundDensity: ThermalJetInput;
  source: ThermalJetInput;
}
export function thermalDraft(input: ThermalFieldInput): ThermalDraft {
  return { boundary: input.boundary, constraint: input.constraint, spacing: input.spacing, temperature: input.temperature, referenceScale: input.referenceScale,
    shape: input.shape.join(', '), mass: { ...input.mass }, stiffness: { ...input.stiffness }, backgroundDensity: { ...input.backgroundDensity },
    source: { value: input.sourceDensity.map(x => x.value).join(', '), first: input.sourceDensity.map(x => x.first).join(', '), second: input.sourceDensity.map(x => x.second).join(', ') } };
}
export function parseThermalDraft(d: ThermalDraft) {
  if (d.shape.length > 12 || Object.values(d.source).some(s => s.length > 1280)) return null;
  const shape = d.shape.split(',').map(x => x.trim());
  if (shape.length < 1 || shape.length > 2 || shape.some(x => !/^(?:[2-9]|[12][0-9]|3[0-2])$/.test(x))) return null;
  const values = d.source.value.split(',').map(x => x.trim()), first = d.source.first.split(',').map(x => x.trim()), second = d.source.second.split(',').map(x => x.trim());
  if (values.length !== first.length || values.length !== second.length) return null;
  const trim = (j: ThermalJetInput) => ({ value: j.value.trim(), first: j.first.trim(), second: j.second.trim() });
  return readThermalFieldInput({ shape: shape.map(Number), boundary: d.boundary, constraint: d.constraint, spacing: d.spacing.trim(), temperature: d.temperature.trim(), referenceScale: d.referenceScale.trim(),
    mass: trim(d.mass), stiffness: trim(d.stiffness), backgroundDensity: trim(d.backgroundDensity), sourceDensity: values.map((value, i) => ({ value, first: first[i], second: second[i] })) });
}
export function thermalProjection(text: string, scale: ThermalScale): number | null {
  if (scale === 'linear') return scalarFieldProjection(text);
  const [n, d = '1'] = text.split('/'), sign = n.startsWith('-') ? -1 : 1, numerator = n.replace('-', '');
  if (numerator === '0') return 0;
  const a = numerator.slice(0, 15), b = d.slice(0, 15);
  const exponent = Math.log(Number(a)) - Math.log(Number(b)) + (numerator.length - a.length - d.length + b.length) * Math.LN10;
  const value = sign * (exponent > 36 ? exponent / Math.LN10 : Math.log1p(Math.exp(exponent)) / Math.LN10);
  return Number.isFinite(value) && value !== 0 ? value : null;
}
export function thermalMatrixView(exact: ThermalNormal, kind: ThermalMatrix, scale: ThermalScale, selected: number) {
  const n = exact.sites, index = Math.max(0, Math.min(n * n - 1, selected)), row = Math.floor(index / n), column = index % n;
  const values = exact[kind].map(r => r.map(x => thermalProjection(x, scale))), available = values.every(r => r.every(x => x !== null));
  const maximum = Math.max(0, ...values.flat().filter((x): x is number => x !== null).map(Math.abs)) || 1;
  return { n, selected: index, row, column, available, maximum, values, points: available ? values.flatMap((r, i) => r.map((v, j) => ({ x: 6 * i / (n - 1) - 3, y: 2.5 * v! / maximum, z: 6 * j / (n - 1) - 3, value: v!, selected: i === row && j === column }))) : [] };
}
export const thermalApprox = (s: string) => { const n = scalarFieldProjection(s); return n === null ? '—' : n === 0 ? '0' : n.toPrecision(4); };
