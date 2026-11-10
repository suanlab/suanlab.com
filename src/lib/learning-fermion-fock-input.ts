/** Owned Fock inputs; n0 is the most significant occupation bit. */
export type FermionFockParameters = Record<string, number>;
export const fermionFockParameterKeys: readonly string[] = Object.freeze([
  'modes', 'sector', 'hDenominator', 'offset',
  ...Array.from({ length: 4 }, (_, i) => `onsite${i}`),
  ...['hopping', 'pairing'].flatMap(prefix => Array.from({ length: 4 }, (_, i) =>
    Array.from({ length: 3 - i }, (_, k) => [`${prefix}Real${i}${i + k + 1}`, `${prefix}Imag${i}${i + k + 1}`]).flat()).flat()),
  ...Array.from({ length: 4 }, (_, i) => Array.from({ length: 3 - i }, (_, k) => `density${i}${i + k + 1}`)).flat(),
  'leftMode', 'rightMode', 'leftCreation', 'rightCreation',
  ...Array.from({ length: 16 }, (_, i) => [`stateReal${i}`, `stateImag${i}`]).flat(),
]);

export function fermionOccupationCount(index: number): number {
  if (!Number.isSafeInteger(index) || index < 0 || index > 15) throw new RangeError('Fock basis index0..15 required');
  let count = 0;
  for (let value = index; value > 0; value = Math.floor(value / 2)) count += value % 2;
  return count;
}

function read(value: unknown): FermionFockParameters | null {
  try {
    if (!value || typeof value !== 'object' || Object.getPrototypeOf(value) !== Object.prototype
      || Reflect.ownKeys(value).length !== fermionFockParameterKeys.length) return null;
    const p: FermionFockParameters = {};
    for (const key of fermionFockParameterKeys) {
      const descriptor = Object.getOwnPropertyDescriptor(value, key);
      if (!descriptor || !Object.hasOwn(descriptor, 'value') || !Number.isSafeInteger(descriptor.value) || Math.abs(descriptor.value) > 1000000) return null;
      p[key] = descriptor.value + 0;
    }
    if (p.modes < 1 || p.modes > 4 || p.sector < 0 || p.sector > 2 || p.hDenominator < 1
      || p.leftMode < 0 || p.leftMode >= p.modes || p.rightMode < 0 || p.rightMode >= p.modes
      || ![0, 1].includes(p.leftCreation) || ![0, 1].includes(p.rightCreation)) return null;
    for (let i = 0; i < 4; i++) {
      if (i >= p.modes && p[`onsite${i}`] !== 0) return null;
      for (let j = i + 1; j < 4; j++) if (j >= p.modes && (
        p[`density${i}${j}`] !== 0 || ['hopping', 'pairing'].some(prefix =>
          p[`${prefix}Real${i}${j}`] !== 0 || p[`${prefix}Imag${i}${j}`] !== 0))) return null;
    }
    let nonzero = false;
    for (let i = 0; i < 16; i++) {
      const active = p[`stateReal${i}`] !== 0 || p[`stateImag${i}`] !== 0;
      if (!active) continue;
      if (i >= 2 ** p.modes) return null;
      if (p.sector !== 0 && fermionOccupationCount(i) % 2 !== p.sector - 1) return null;
      nonzero = true;
    }
    return nonzero ? p : null;
  } catch { return null; }
}

export function isFermionFockParameters(value: unknown): value is FermionFockParameters { return read(value) !== null; }
export function snapshotFermionFockParameters(value: FermionFockParameters): Readonly<FermionFockParameters> {
  const p = read(value);
  if (!p) throw new RangeError('74 owned bounded integers; modes1..4,sector0=formal/1=even/2=odd,positive denominator,valid probes and nonzero sector-compatible state required');
  return Object.freeze(p);
}
export function createFermionFockParameters(modes = 2, sector = 1): Readonly<FermionFockParameters> {
  const p = { ...Object.fromEntries(fermionFockParameterKeys.map(key => [key, 0])), modes, sector, hDenominator: 1, leftMode: 0, rightMode: Math.min(1, modes - 1), leftCreation: 1, rightCreation: 1 } as FermionFockParameters;
  p[`stateReal${sector === 2 ? 1 : 0}`] = 1;
  return snapshotFermionFockParameters(p);
}
