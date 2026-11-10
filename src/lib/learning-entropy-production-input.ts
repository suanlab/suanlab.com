/** Reader-safe owned preparation contract. No numerical or browser imports. */
export interface ThreeStateEntropyParameters {
  a: number; b: number; c: number; d: number; e: number; f: number; rateDenominator: number;
  timeNumerator: number; timeDenominator: number; sampleSteps: number;
  initialState: number; pathCode: number; jumpCount: number;
  tiltNumerator: number; tiltDenominator: number;
}
export const entropyParameterKeys = ['a', 'b', 'c', 'd', 'e', 'f', 'rateDenominator', 'timeNumerator', 'timeDenominator', 'sampleSteps', 'initialState', 'pathCode', 'jumpCount', 'tiltNumerator', 'tiltDenominator'] as const;
const integer = (x: unknown, lo = 1, hi = 1000000000): x is number => typeof x === 'number' && Number.isSafeInteger(x) && x >= lo && x <= hi;
export function isThreeStateEntropyParameters(value: unknown): value is ThreeStateEntropyParameters {
  if (!value || typeof value !== 'object' || Object.getPrototypeOf(value) !== Object.prototype || Reflect.ownKeys(value).length !== entropyParameterKeys.length) return false;
  const p = value as Record<string, unknown>;
  if (!entropyParameterKeys.every(k => Object.prototype.hasOwnProperty.call(p, k) && integer(p[k], k === 'tiltNumerator' ? -1000000000 : ['timeNumerator', 'initialState', 'pathCode', 'jumpCount'].includes(k) ? 0 : 1, k === 'sampleSteps' ? 64 : k === 'initialState' ? 2 : k === 'jumpCount' ? 16 : 1000000000))) return false;
  const v = p as unknown as ThreeStateEntropyParameters;
  return v.timeNumerator <= 4 * v.timeDenominator && v.tiltNumerator >= -v.tiltDenominator && v.tiltNumerator <= 2 * v.tiltDenominator
    && v.pathCode < 2 ** v.jumpCount && (v.timeNumerator > 0 || v.jumpCount === 0);
}
export function snapshotEntropyParameters(value: ThreeStateEntropyParameters): Readonly<ThreeStateEntropyParameters> {
  if (!isThreeStateEntropyParameters(value)) throw new RangeError('Invalid entropy production preparation');
  return Object.freeze(Object.fromEntries(entropyParameterKeys.map(k => [k, value[k] + 0]))) as unknown as Readonly<ThreeStateEntropyParameters>;
}
