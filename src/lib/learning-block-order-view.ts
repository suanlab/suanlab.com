/** Reader-side drawing and form projections of independently checked values. */
import { readBlockOrderInput, type BlockOrderInput } from './learning-block-order-input';
import type { BlockOrderWireExact } from './learning-block-order-protocol';
import { scalarFieldProjection } from './learning-scalar-field-view';
export type BlockOrderMode = 'configuration' | 'conditional' | 'profile' | 'ensemble';
export interface BlockOrderDraft {
  shape: string; blockShape: string; boundary: BlockOrderInput['boundary']; disagreementActivity: string;
  blockUpActivities: string; selectedUpCounts: string; witnessPair: string;
}
export function blockOrderDraft(input: BlockOrderInput): BlockOrderDraft {
  return { shape: input.shape.join(', '), blockShape: input.blockShape.join(', '), boundary: input.boundary,
    disagreementActivity: input.disagreementActivity, blockUpActivities: input.blockUpActivities.join(', '), selectedUpCounts: input.selectedUpCounts.join(', '), witnessPair: input.witnessPair.join(', ') };
}
export function parseBlockOrderDraft(d: BlockOrderDraft): BlockOrderInput | null {
  if (d.shape.length > 12 || d.blockShape.length > 12 || d.disagreementActivity.length > 13 || d.blockUpActivities.length > 64 || d.selectedUpCounts.length > 20 || d.witnessPair.length > 12) return null;
  const integers = (s: string, empty = false) => {
    if (empty && !s.trim()) return [];
    const parts = s.split(',').map(x => x.trim());
    return parts.every(x => /^(?:0|[1-9]\d?)$/.test(x)) ? parts.map(Number) : null;
  };
  return readBlockOrderInput({ shape: integers(d.shape), blockShape: integers(d.blockShape), boundary: d.boundary,
    disagreementActivity: d.disagreementActivity.trim(), blockUpActivities: d.blockUpActivities.split(',').map(x => x.trim()),
    selectedUpCounts: integers(d.selectedUpCounts), witnessPair: integers(d.witnessPair,true) });
}
export const blockOrderApprox = (s: string) => { const n = scalarFieldProjection(s); return n === null ? '—' : n === 0 ? '0' : n.toPrecision(4); };
export function blockOrderSpatial(input: BlockOrderInput, exact: BlockOrderWireExact, mode: BlockOrderMode, chosenExample: number) {
  const index = Math.max(0,Math.min(exact.selected.examples.length-1,chosenExample)), example = exact.selected.examples[index];
  const profile = exact.profileChunks.flat()[exact.selected.profileIndex];
  const values = Array.from({ length: exact.sites }, (_, site) => mode === 'configuration' ? String(((example.mask >> site) & 1) ? 1 : -1)
    : mode === 'conditional' ? exact.selected.conditionalSpinMeans[site] : mode === 'profile' ? profile.magnetizations[exact.blockOf[site]] : exact.meanMagnetizations[exact.blockOf[site]]);
  const scale = 6/Math.max(input.shape[0]-1,(input.shape[1] ?? 1)-1,1);
  const points = values.map((value,site) => ({ site, block: exact.blockOf[site], exact: value, value: scalarFieldProjection(value),
    x: ((site % input.shape[0])-(input.shape[0]-1)/2)*scale, z: (Math.floor(site/input.shape[0])-((input.shape[1] ?? 1)-1)/2)*scale }));
  return { points, edges: exact.edges, example, exampleIndex: index, mode, available: points.every(p => p.value !== null), spacing: scale };
}
export type BlockOrderSpatialModel = ReturnType<typeof blockOrderSpatial>;
