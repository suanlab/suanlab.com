/** Owned open-boundary MPS inputs; structural validity does not imply nonzero contraction. */
export type MpsParameters = Record<string, number>;
export const mpsParameterKeys: readonly string[] = Object.freeze([
  'sites', 'tensorDenominator', 'bond0', 'bond1', 'bond2',
  'gaugeBond', 'gaugeRow', 'gaugeColumn', 'gaugeScaleNumerator', 'gaugeScaleDenominator', 'gaugeShearReal', 'gaugeShearImag', 'gaugeShearDenominator',
  ...Array.from({ length: 4 }, (_, site) => Array.from({ length: 4 }, (_, left) => Array.from({ length: 2 }, (_, physical) => Array.from({ length: 4 }, (_, right) => [`t${site}Real${left}${physical}${right}`, `t${site}Imag${left}${physical}${right}`]).flat()).flat()).flat()).flat(),
]);
function read(value: unknown): MpsParameters | null {
  try {
    if (!value || typeof value !== 'object' || Object.getPrototypeOf(value) !== Object.prototype || Reflect.ownKeys(value).length !== mpsParameterKeys.length) return null;
    const p: MpsParameters = {};
    for (const key of mpsParameterKeys) {
      const d = Object.getOwnPropertyDescriptor(value, key);
      if (!d || !Object.hasOwn(d, 'value') || !Number.isSafeInteger(d.value) || Math.abs(d.value) > 1000000) return null;
      p[key] = d.value + 0;
    }
    if (p.sites < 1 || p.sites > 4 || p.tensorDenominator < 1 || p.gaugeScaleNumerator === 0 || p.gaugeScaleDenominator < 1 || p.gaugeShearDenominator < 1) return null;
    for (let i = 0; i < 3; i++) if (i < p.sites - 1 ? p[`bond${i}`] < 1 || p[`bond${i}`] > 4 : p[`bond${i}`] !== 0) return null;
    if (p.gaugeBond < -1 || p.gaugeBond >= p.sites - 1) return null;
    if (p.gaugeBond === -1) {
      if (p.gaugeRow !== 0 || p.gaugeColumn !== 0 || p.gaugeScaleNumerator !== 1 || p.gaugeScaleDenominator !== 1 || p.gaugeShearReal !== 0 || p.gaugeShearImag !== 0 || p.gaugeShearDenominator !== 1) return null;
    } else {
      const d = p[`bond${p.gaugeBond}`];
      if (p.gaugeRow < 0 || p.gaugeRow >= d || p.gaugeColumn < 0 || p.gaugeColumn >= d) return null;
      if (p.gaugeRow === p.gaugeColumn && (p.gaugeShearReal !== 0 || p.gaugeShearImag !== 0)) return null;
    }
    for (let site = 0; site < 4; site++) {
      const left = site === 0 ? 1 : p[`bond${site - 1}`], right = site === p.sites - 1 ? 1 : p[`bond${site}`];
      for (let l = 0; l < 4; l++) for (let s = 0; s < 2; s++) for (let r = 0; r < 4; r++) if (site >= p.sites || l >= left || r >= right) {
        if (p[`t${site}Real${l}${s}${r}`] !== 0 || p[`t${site}Imag${l}${s}${r}`] !== 0) return null;
      }
    }
    return p;
  } catch { return null; }
}
export function isMpsParameters(value: unknown): value is MpsParameters { return read(value) !== null; }
export function snapshotMpsParameters(value: MpsParameters): Readonly<MpsParameters> {
  const p = read(value); if (!p) throw new RangeError('269 owned bounded integers;sites1..4;active bonds1..4;positive tensor/gauge denominators;nonzero scale;valid shear;inactive tensor entries0 required');
  return Object.freeze(p);
}
export function createMpsParameters(sites = 3, bondDimension = 2): Readonly<MpsParameters> {
  if (!Number.isInteger(sites) || sites < 1 || sites > 4) throw new RangeError('Sites1..4 required');
  if (!Number.isInteger(bondDimension) || bondDimension < 1 || bondDimension > 4) throw new RangeError('Bond dimension1..4 required');
  const p: MpsParameters = { ...Object.fromEntries(mpsParameterKeys.map(key => [key, 0])), sites, tensorDenominator: 1, gaugeBond: sites > 1 ? 0 : -1, gaugeColumn: sites > 1 && bondDimension > 1 ? 1 : 0, gaugeScaleNumerator: sites > 1 ? 2 : 1, gaugeScaleDenominator: 1, gaugeShearReal: sites > 1 && bondDimension > 1 ? 1 : 0, gaugeShearDenominator: 1 };
  for (let i = 0; i < sites - 1; i++) p[`bond${i}`] = bondDimension;
  for (let i = 0; i < sites; i++) p[`t${i}Real000`] = 1;
  return snapshotMpsParameters(p);
}
