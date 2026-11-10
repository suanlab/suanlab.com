/** Owned bounded integer preparation for a general small Hermitian Hamiltonian. */
export const rayleighKeys = [
  'dimension', 'hDenominator',
  ...Array.from({length: 4}, (_, i) => Array.from({length: 4 - i}, (_, k) => {
    const j = i + k;
    return i === j ? [`hReal${i}${j}`] : [`hReal${i}${j}`, `hImag${i}${j}`];
  }).flat()).flat(),
  ...['state', 'direction'].flatMap(prefix => Array.from({length: 4}, (_, i) => [`${prefix}Real${i}`, `${prefix}Imag${i}`]).flat()),
  'pathNumerator', 'pathDenominator',
] as readonly string[];
export type RayleighParameters = Record<string, number>;
export function isRayleighParameters(value: unknown): value is RayleighParameters {
  if (!value || typeof value !== 'object' || Object.getPrototypeOf(value) !== Object.prototype || Reflect.ownKeys(value).length !== rayleighKeys.length) return false;
  const descriptors = Object.getOwnPropertyDescriptors(value);
  if (!rayleighKeys.every(key => Object.hasOwn(descriptors, key) && Object.hasOwn(descriptors[key], 'value') && Number.isSafeInteger(descriptors[key].value) && Math.abs(descriptors[key].value) <= 1000000)) return false;
  const p = value as RayleighParameters, d = p.dimension;
  if (d < 1 || d > 4 || p.hDenominator < 1 || p.pathDenominator < 1) return false;
  for (let i = 0; i < 4; i++) {
    if (i >= d && ['state', 'direction'].some(prefix => p[`${prefix}Real${i}`] !== 0 || p[`${prefix}Imag${i}`] !== 0)) return false;
    for (let j = i; j < 4; j++) if ((i >= d || j >= d) && (p[`hReal${i}${j}`] !== 0 || i !== j && p[`hImag${i}${j}`] !== 0)) return false;
  }
  return Array.from({length: d}, (_, i) => p[`stateReal${i}`] !== 0 || p[`stateImag${i}`] !== 0).some(Boolean);
}
export function snapshotRayleighParameters(value: RayleighParameters): Readonly<RayleighParameters> {
  if (!isRayleighParameters(value)) throw new RangeError('36 owned bounded integers; dimension1..4,positive denominators,nonzero state and zero inactive coordinates required');
  return Object.freeze(Object.fromEntries(rayleighKeys.map(key => [key, value[key] + 0])));
}
