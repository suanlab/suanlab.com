/** Reader-safe owned state and at-most-rank budget; no numerical imports. */
export type SchmidtParameters = Record<string, number>;
export const schmidtParameterKeys: readonly string[] = Object.freeze([
  'sites', 'selectedCut', 'retainedRank',
  ...Array.from({ length: 16 }, (_, i) => [`stateReal${i}`, `stateImag${i}`]).flat(),
]);
function read(value: unknown): SchmidtParameters | null {
  try {
    if (!value || typeof value !== 'object' || Object.getPrototypeOf(value) !== Object.prototype || Reflect.ownKeys(value).length !== schmidtParameterKeys.length) return null;
    const p: SchmidtParameters = {};
    for (const key of schmidtParameterKeys) {
      const d = Object.getOwnPropertyDescriptor(value, key);
      if (!d || !Object.hasOwn(d, 'value') || !Number.isSafeInteger(d.value) || Math.abs(d.value) > 1000000) return null;
      p[key] = d.value + 0;
    }
    if (p.sites < 2 || p.sites > 4 || p.selectedCut < 1 || p.selectedCut >= p.sites || p.retainedRank < 0 || p.retainedRank > 4) return null;
    let nonzero = false;
    for (let i = 0; i < 16; i++) if (p[`stateReal${i}`] !== 0 || p[`stateImag${i}`] !== 0) { if (i >= 2 ** p.sites) return null; nonzero = true; }
    return nonzero ? p : null;
  } catch { return null; }
}
export function isSchmidtParameters(value: unknown): value is SchmidtParameters { return read(value) !== null; }
export function snapshotSchmidtParameters(value: SchmidtParameters): Readonly<SchmidtParameters> {
  const p = read(value); if (!p) throw new RangeError('35 owned bounded integers;sites2..4;valid selected cut;rank budget0..4;nonzero active complex state required');
  return Object.freeze(p);
}
export function createSchmidtParameters(sites = 4, retainedRank = 2): Readonly<SchmidtParameters> {
  if (!Number.isInteger(sites) || sites < 2 || sites > 4) throw new RangeError('Sites2..4 required');
  return snapshotSchmidtParameters({ ...Object.fromEntries(schmidtParameterKeys.map(key => [key, 0])), sites, selectedCut: Math.floor(sites / 2), retainedRank, stateReal0: 1, [`stateReal${2 ** sites - 1}`]: 1 });
}
