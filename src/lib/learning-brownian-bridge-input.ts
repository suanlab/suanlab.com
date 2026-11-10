/** Reader-safe time-slicing inputs. Exact arithmetic stays in the calculation Worker. */
export interface BrownianBridgeInput {
  intervals: number;
  time: string;
  diffusion: string;
  start: string;
  end: string;
  probe: number;
  source: string;
}
const keys = ['intervals', 'time', 'diffusion', 'start', 'end', 'probe', 'source'] as const;
function fraction(value: unknown, positive = false): value is string {
  if (typeof value !== 'string' || !/^-?(?:0|[1-9]\d{0,11})(?:\/[1-9]\d{0,11})?$/.test(value) || /^-0(?:\/|$)/.test(value)) return false;
  const [n, d = '1'] = value.split('/');
  // Twelve-digit integers and sixteen times a denominator stay below 2^53.
  return Math.abs(Number(n)) <= 16 * Number(d) && (!positive || Number(n) > 0);
}
export function readBrownianBridgeInput(value: unknown): Readonly<BrownianBridgeInput> | null {
  try {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
    const descriptors = Object.getOwnPropertyDescriptors(value);
    if (Reflect.ownKeys(descriptors).length !== keys.length || keys.some(k => !descriptors[k] || !Object.hasOwn(descriptors[k], 'value'))) return null;
    const own = Object.fromEntries(keys.map(k => [k, descriptors[k].value]));
    if (!Number.isInteger(own.intervals) || own.intervals < 2 || own.intervals > 32 || !Number.isInteger(own.probe) || Object.is(own.probe, -0) || own.probe < 0 || own.probe > own.intervals) return null;
    if (!fraction(own.time, true) || !fraction(own.diffusion, true) || !fraction(own.start) || !fraction(own.end) || !fraction(own.source)) return null;
    return Object.freeze({ intervals: own.intervals, time: own.time, diffusion: own.diffusion, start: own.start, end: own.end, probe: own.probe, source: own.source });
  } catch { return null; }
}
export const brownianBridgeDefault: Readonly<BrownianBridgeInput> = Object.freeze({ intervals: 4, time: '1', diffusion: '1', start: '0', end: '1', probe: 2, source: '0' });
const example = (id: string, patch: Partial<BrownianBridgeInput>) => {
  const input = readBrownianBridgeInput({ ...brownianBridgeDefault, ...patch });
  if (!input) throw Error('Invalid bridge example: ' + id);
  return Object.freeze({ id, input });
};
export const brownianBridgeExamples = Object.freeze([
  example('four-interval-hand-check', {}),
  example('one-interior-coordinate', { intervals: 2, probe: 1 }),
  example('refined-eight', { intervals: 8, probe: 4 }),
  example('maximum-thirty-two', { intervals: 32, probe: 16 }),
  example('double-diffusion', { diffusion: '2' }),
  example('double-time', { time: '2' }),
  example('translated-endpoints', { start: '3', end: '4' }),
  example('changed-displacement', { end: '2' }),
  example('middle-source', { source: '2' }),
  example('negative-source', { source: '-2' }),
  example('fixed-start-source', { probe: 0, start: '-2', source: '3' }),
  example('fixed-end-source', { probe: 4, source: '16' }),
  example('fractional-grid', { intervals: 7, time: '3/2', diffusion: '2/3', start: '-1/2', end: '5/4', probe: 3, source: '1/3' }),
  example('equivalent-spelling', { time: '2/2', diffusion: '3/3', start: '0/5', end: '4/4', source: '0/7' }),
  example('minimum-positive-factors', { intervals: 32, time: '1/999999999989', diffusion: '1/999999999983', start: '-16', end: '16', probe: 1, source: '16' }),
  example('maximum-coprime', { intervals: 32, time: '999999999983/62499999999', diffusion: '999999999979/62499999999', start: '-999999999961/62499999999', end: '999999999959/62499999999', probe: 17, source: '999999999953/62499999999' }),
]);
