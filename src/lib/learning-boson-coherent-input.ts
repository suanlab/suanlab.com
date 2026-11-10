/** Exact rational coherent amplitudes; no numerical engine dependency. */
import { bosonFockParameterKeys, isBosonFockParameters } from './learning-boson-fock-input';
export type BosonCoherentParameters = Record<string, number>;
export const bosonCoherentParameterKeys: readonly string[] = Object.freeze([
  ...bosonFockParameterKeys.filter(key => !key.startsWith('state')),
  'alphaReal0', 'alphaImag0', 'alphaDenominator0', 'alphaReal1', 'alphaImag1', 'alphaDenominator1',
]);
function read(value: unknown): BosonCoherentParameters | null {
  try {
    if (!value || typeof value !== 'object' || Object.getPrototypeOf(value) !== Object.prototype || Reflect.ownKeys(value).length !== bosonCoherentParameterKeys.length) return null;
    const p: BosonCoherentParameters = {};
    for (const key of bosonCoherentParameterKeys) {
      const d = Object.getOwnPropertyDescriptor(value, key);
      if (!d || !Object.hasOwn(d, 'value') || !Number.isSafeInteger(d.value) || Math.abs(d.value) > 1000000) return null;
      p[key] = d.value + 0;
    }
    const raw = Object.fromEntries(bosonFockParameterKeys.map(key => [key, key.startsWith('state') ? Number(key === 'stateReal0') : p[key]]));
    if (!isBosonFockParameters(raw)) return null;
    for (let i = 0; i < 2; i++) {
      const a = p[`alphaReal${i}`], b = p[`alphaImag${i}`], d = p[`alphaDenominator${i}`];
      if (d <= 0 || BigInt(a) ** BigInt(2) + BigInt(b) ** BigInt(2) > BigInt(16) * BigInt(d) ** BigInt(2)) return null;
      if (i >= p.modes && (a !== 0 || b !== 0 || d !== 1)) return null;
    }
    return p;
  } catch { return null; }
}
export function isBosonCoherentParameters(value: unknown): value is BosonCoherentParameters { return read(value) !== null; }
export function snapshotBosonCoherentParameters(value: BosonCoherentParameters): Readonly<BosonCoherentParameters> {
  const p = read(value); if (!p) throw new RangeError('26 owned bounded integers;valid finite Hamiltonian;positive alpha denominators;per-mode0<=mu<=16;inactive alpha0/1 required');
  return Object.freeze(p);
}
export function createBosonCoherentParameters(modes = 1, cutoff0 = 3, cutoff1 = modes === 2 ? 3 : 0): Readonly<BosonCoherentParameters> {
  return snapshotBosonCoherentParameters({ ...Object.fromEntries(bosonCoherentParameterKeys.map(key => [key, 0])), modes, cutoff0, cutoff1, hDenominator: 1, alphaReal0: 1, alphaDenominator0: 1, alphaDenominator1: 1, rightCreation: 1 });
}
