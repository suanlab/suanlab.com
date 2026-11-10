/** Reader-safe owned boson input. Cutoffs are maximum occupancies, not dimensions. */
export type BosonFockParameters = Record<string, number>;
export const bosonFockParameterKeys: readonly string[] = Object.freeze([
  'modes', 'cutoff0', 'cutoff1', 'hDenominator', 'offset',
  'onsite0', 'onsite1', 'selfDensity0', 'selfDensity1',
  'driveReal0', 'driveImag0', 'driveReal1', 'driveImag1',
  'hoppingReal', 'hoppingImag', 'crossDensity',
  'leftMode', 'rightMode', 'leftCreation', 'rightCreation',
  ...Array.from({ length: 16 }, (_, i) => [`stateReal${i}`, `stateImag${i}`]).flat(),
]);
function read(value: unknown): BosonFockParameters | null {
  try {
    if (!value || typeof value !== 'object' || Object.getPrototypeOf(value) !== Object.prototype || Reflect.ownKeys(value).length !== bosonFockParameterKeys.length) return null;
    const p: BosonFockParameters = {};
    for (const key of bosonFockParameterKeys) {
      const d = Object.getOwnPropertyDescriptor(value, key);
      if (!d || !Object.hasOwn(d, 'value') || !Number.isSafeInteger(d.value) || Math.abs(d.value) > 1000000) return null;
      p[key] = d.value + 0;
    }
    if (p.modes < 1 || p.modes > 2 || p.cutoff0 < 0 || p.cutoff0 > 3 || p.cutoff1 < 0 || p.cutoff1 > 3 || p.hDenominator < 1
      || p.leftMode < 0 || p.leftMode >= p.modes || p.rightMode < 0 || p.rightMode >= p.modes || ![0, 1].includes(p.leftCreation) || ![0, 1].includes(p.rightCreation)) return null;
    if (p.modes === 1 && ['cutoff1', 'onsite1', 'selfDensity1', 'driveReal1', 'driveImag1', 'hoppingReal', 'hoppingImag', 'crossDensity'].some(key => p[key] !== 0)) return null;
    const dimension = (p.cutoff0 + 1) * (p.cutoff1 + 1); let nonzero = false;
    for (let i = 0; i < 16; i++) if (p[`stateReal${i}`] !== 0 || p[`stateImag${i}`] !== 0) { if (i >= dimension) return null; nonzero = true; }
    return nonzero ? p : null;
  } catch { return null; }
}
export function isBosonFockParameters(value: unknown): value is BosonFockParameters { return read(value) !== null; }
export function snapshotBosonFockParameters(value: BosonFockParameters): Readonly<BosonFockParameters> {
  const p = read(value); if (!p) throw new RangeError('52 owned bounded integers;modes1..2;cutoffs0..3;positive denominator;nonzero active state;valid probes required');
  return Object.freeze(p);
}
export function createBosonFockParameters(modes = 1, cutoff0 = 3, cutoff1 = modes === 2 ? 3 : 0): Readonly<BosonFockParameters> {
  return snapshotBosonFockParameters({ ...Object.fromEntries(bosonFockParameterKeys.map(key => [key, 0])), modes, cutoff0, cutoff1, hDenominator: 1, stateReal0: 1, leftCreation: 0, rightCreation: 1 });
}
