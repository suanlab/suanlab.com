/** Reader-safe owned parameters. No exact arithmetic engine belongs in this module. */
export interface EuclideanKernelInput {
  intervals: number;
  time: string;
  mass: string;
  hbar: string;
  omegaSquared: string;
  start: string;
  end: string;
  probe: number;
}
const keys = ['intervals', 'time', 'mass', 'hbar', 'omegaSquared', 'start', 'end', 'probe'] as const;
function fraction(value: unknown, sign: 'positive' | 'nonnegative' | 'signed'): value is string {
  if (typeof value !== 'string' || !/^-?(?:0|[1-9]\d{0,11})(?:\/[1-9]\d{0,11})?$/.test(value) || /^-0(?:\/|$)/.test(value)) return false;
  const [numerator, denominator = '1'] = value.split('/'), n = Number(numerator), d = Number(denominator);
  return Math.abs(n) <= 16 * d && (sign === 'signed' || (sign === 'positive' ? n > 0 : n >= 0));
}
export function readEuclideanKernelInput(value: unknown): Readonly<EuclideanKernelInput> | null {
  try {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
    const descriptors = Object.getOwnPropertyDescriptors(value);
    if (Reflect.ownKeys(descriptors).length !== keys.length || keys.some(key => !descriptors[key] || !Object.hasOwn(descriptors[key], 'value'))) return null;
    const own = Object.fromEntries(keys.map(key => [key, descriptors[key].value]));
    if (!Number.isInteger(own.intervals) || own.intervals < 2 || own.intervals > 32 || !Number.isInteger(own.probe) || Object.is(own.probe, -0) || own.probe < 0 || own.probe > own.intervals) return null;
    if (!fraction(own.time, 'positive') || !fraction(own.mass, 'positive') || !fraction(own.hbar, 'positive') || !fraction(own.omegaSquared, 'nonnegative') || !fraction(own.start, 'signed') || !fraction(own.end, 'signed')) return null;
    return Object.freeze({ intervals: own.intervals, time: own.time, mass: own.mass, hbar: own.hbar, omegaSquared: own.omegaSquared, start: own.start, end: own.end, probe: own.probe });
  } catch { return null; }
}
export const euclideanKernelDefault: Readonly<EuclideanKernelInput> = Object.freeze({ intervals: 2, time: '1', mass: '1', hbar: '1', omegaSquared: '1', start: '0', end: '1', probe: 1 });
function example(id: string, patch: Partial<EuclideanKernelInput>) {
  const input = readEuclideanKernelInput({ ...euclideanKernelDefault, ...patch });
  if (!input) throw Error('Invalid Euclidean kernel example: ' + id);
  return Object.freeze({ id, input });
}
export const euclideanKernelExamples = Object.freeze([
  example('one-variable-hand-check', {}),
  example('four-intervals', { intervals: 4, probe: 2 }),
  example('refined-eight', { intervals: 8, probe: 4 }),
  example('maximum-thirty-two', { intervals: 32, probe: 16 }),
  example('free-particle', { intervals: 4, omegaSquared: '0', probe: 2 }),
  example('stronger-confinement', { intervals: 4, omegaSquared: '4', probe: 2 }),
  example('reverse-endpoints', { start: '1', end: '0' }),
  example('equal-endpoints', { start: '1', end: '1' }),
  example('opposite-endpoints', { start: '-1', end: '1' }),
  example('zero-endpoints', { end: '0' }),
  example('double-time', { time: '2' }),
  example('double-mass', { mass: '2' }),
  example('double-hbar', { hbar: '2' }),
  example('off-diagonal-error-comparison', { intervals: 4, start: '1/3', end: '4/3', probe: 2 }),
  example('fractional-model', { intervals: 7, time: '3/2', mass: '2/3', hbar: '5/4', omegaSquared: '7/5', start: '-1/2', end: '5/4', probe: 3 }),
  example('equivalent-spelling', { time: '2/2', mass: '3/3', hbar: '4/4', omegaSquared: '5/5', start: '0/7', end: '8/8' }),
  example('fixed-endpoint-probe', { intervals: 4, probe: 0 }),
  example('minimum-positive-factors', { intervals: 32, time: '1/999999999989', mass: '1/999999999983', hbar: '1/999999999979', omegaSquared: '1/999999999961', start: '-16', end: '16', probe: 1 }),
  example('maximum-coprime', { intervals: 32, time: '999999999983/62499999999', mass: '999999999979/62499999999', hbar: '999999999961/62499999999', omegaSquared: '999999999959/62499999999', start: '-999999999953/62499999999', end: '999999999929/62499999999', probe: 17 }),
  example('mixed-coprime-denominators', { intervals: 32, time: '999999999983/999999999989', mass: '999999999979/999999999961', hbar: '999999999953/999999999929', omegaSquared: '999999999893/999999999883', start: '-999999999797/999999999761', end: '999999999757/999999999749', probe: 17 }),
]);
