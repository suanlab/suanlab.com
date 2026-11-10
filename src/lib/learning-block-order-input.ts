/** Reader-safe finite input contract. No enumeration, rational model or data imports. */
export interface BlockOrderInput {
  shape: number[];
  blockShape: number[];
  boundary: 'open' | 'periodic';
  disagreementActivity: string;
  blockUpActivities: string[];
  selectedUpCounts: number[];
  witnessPair: number[];
}
export function blockOrderData(value: unknown, keys: readonly string[]): Record<string, unknown> | null {
  try {
    if (!value || typeof value !== 'object' || Object.getPrototypeOf(value) !== Object.prototype || Reflect.ownKeys(value).length !== keys.length) return null;
    const own: Record<string, unknown> = {};
    for (const key of keys) {
      const d = Object.getOwnPropertyDescriptor(value, key);
      if (!d || !Object.hasOwn(d, 'value')) return null;
      Object.defineProperty(own, key, { value: d.value, enumerable: true });
    }
    return own;
  } catch { return null; }
}
function array(value: unknown, minimum: number, maximum = minimum): unknown[] | null {
  if (!Array.isArray(value) || Object.getPrototypeOf(value) !== Array.prototype) return null;
  const d = Object.getOwnPropertyDescriptor(value, 'length');
  if (!d || !Object.hasOwn(d, 'value') || d.value < minimum || d.value > maximum || Reflect.ownKeys(value).length !== d.value + 1) return null;
  const entries = [];
  for (let i = 0; i < d.value; i++) {
    const e = Object.getOwnPropertyDescriptor(value, String(i));
    if (!e || !Object.hasOwn(e, 'value')) return null;
    entries.push(e.value);
  }
  return entries;
}
function activity(value: unknown): string | null {
  if (typeof value !== 'string' || value.length > 13 || !/^[1-9]\d{0,5}(\/[1-9]\d{0,5})?$/.test(value)) return null;
  const [ns, ds = '1'] = value.split('/');
  let n = BigInt(ns), d = BigInt(ds), a = n, b = d;
  while (b) [a, b] = [b, a % b];
  n /= a; d /= a;
  return d === BigInt(1) ? String(n) : `${n}/${d}`;
}
const fields = ['shape', 'blockShape', 'boundary', 'disagreementActivity', 'blockUpActivities', 'selectedUpCounts', 'witnessPair'];
/** Descriptor-only ownership: caller getters, iterators and toJSON are never invoked. */
export function readBlockOrderInput(value: unknown): BlockOrderInput | null {
  try {
    const own = blockOrderData(value, fields);
    if (!own) return null;
    const shape = array(own.shape, 1, 2), bs = array(own.blockShape, shape?.length ?? 0);
    const integers = (xs: unknown[] | null, lo: number, hi: number): xs is number[] =>
      !!xs && xs.every(x => typeof x === 'number' && Number.isInteger(x) && x >= lo && x <= hi);
    if (!integers(shape, 2, 16) || !integers(bs, 1, 16) || shape.some((s, a) => s % bs[a] !== 0)) return null;
    const n = shape.reduce((a, b) => a * b, 1), v = bs.reduce((a, b) => a * b, 1), blocks = n / v;
    if (n > 16 || blocks > 4 || !['open', 'periodic'].includes(own.boundary as string)) return null;
    const x = activity(own.disagreementActivity), ys = array(own.blockUpActivities, blocks)?.map(activity);
    const selected = array(own.selectedUpCounts, blocks), pair = array(own.witnessPair, blocks === 1 ? 0 : 2);
    if (!x || !ys || ys.some(y => y === null) || !integers(selected, 0, v) || !integers(pair, 0, blocks - 1)) return null;
    if (blocks > 1 && pair[0] >= pair[1]) return null;
    return Object.freeze({ shape: Object.freeze(shape) as number[], blockShape: Object.freeze(bs) as number[], boundary: own.boundary as BlockOrderInput['boundary'],
      disagreementActivity: x, blockUpActivities: Object.freeze(ys) as string[], selectedUpCounts: Object.freeze(selected) as number[], witnessPair: Object.freeze(pair) as number[] });
  } catch { return null; }
}
export const blockOrderDefault = readBlockOrderInput({ shape: [4], blockShape: [2], boundary: 'open', disagreementActivity: '1/2',
  blockUpActivities: ['1','1'], selectedUpCounts: [1,1], witnessPair: [0,1] })!;
function example(id: string, patch: Partial<BlockOrderInput>) {
  const input = readBlockOrderInput({ ...blockOrderDefault, ...patch });
  if (!input) throw Error(`Invalid block-order example ${id}`);
  return Object.freeze({ id, input });
}
export const blockOrderExamples = Object.freeze([
  example('unequal-hidden-weights', {}),
  example('nonneighbor-closure', { shape: [6], blockUpActivities: ['1','1','1'], selectedUpCounts: [1,1,1], witnessPair: [0,2] }),
  example('boundary-versus-mean', { blockShape: [4], blockUpActivities: ['1'], selectedUpCounts: [3], witnessPair: [] }),
  example('hidden-boundary-response', { shape: [6], blockUpActivities: ['1','1','1'], selectedUpCounts: [2,1,0], witnessPair: [0,2] }),
  example('periodic-neighbors', { shape: [6], boundary: 'periodic', blockUpActivities: ['1','1','1'], selectedUpCounts: [1,1,1], witnessPair: [0,2] }),
  example('uncoupled-counting', { disagreementActivity: '1' }),
  example('two-dimensional', { shape: [4,4], blockShape: [2,2], boundary: 'periodic', blockUpActivities: ['1','1','1','1'], selectedUpCounts: [2,2,2,2], witnessPair: [0,3] }),
  example('antiferromagnetic', { disagreementActivity: '2' }),
  example('biased-profiles', { blockUpActivities: ['3','1/2'] }),
]);
