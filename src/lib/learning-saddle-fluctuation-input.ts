/** Reader-safe owned inputs. Exact arithmetic stays in Worker-side modules. */
export interface SaddleChainInput {
  mode: 'chain'; intervals: number; time: string; mass: string; hbar: string;
  omegaSquared: string; quartic: string; start: string; end: string; probe: number;
}
export interface SaddleWellInput { mode: 'well'; hbar: string; kappa: string; }
export type SaddleFluctuationInput = Readonly<SaddleChainInput | SaddleWellInput>;
const chainKeys = ['mode', 'intervals', 'time', 'mass', 'hbar', 'omegaSquared', 'quartic', 'start', 'end', 'probe'] as const;
const wellKeys = ['mode', 'hbar', 'kappa'] as const;
function fraction(value: unknown, sign: 'positive' | 'nonnegative' | 'signed'): value is string {
  if (typeof value !== 'string' || !/^-?(?:0|[1-9]\d{0,11})(?:\/[1-9]\d{0,11})?$/.test(value) || /^-0(?:\/|$)/.test(value)) return false;
  const [numerator, denominator = '1'] = value.split('/'), n = Number(numerator), d = Number(denominator);
  return Math.abs(n) <= 16 * d && (sign === 'signed' || (sign === 'positive' ? n > 0 : n >= 0));
}
export function readSaddleFluctuationInput(value: unknown): SaddleFluctuationInput | null {
  try {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
    const descriptors = Object.getOwnPropertyDescriptors(value), mode = descriptors.mode;
    if (!mode || !Object.hasOwn(mode, 'value') || !['chain', 'well'].includes(mode.value)) return null;
    const keys = mode.value === 'chain' ? chainKeys : wellKeys;
    if (Reflect.ownKeys(descriptors).length !== keys.length || keys.some(key => !descriptors[key] || !Object.hasOwn(descriptors[key], 'value'))) return null;
    const own = Object.fromEntries(keys.map(key => [key, descriptors[key].value]));
    if (!fraction(own.hbar, 'positive')) return null;
    if (mode.value === 'well') return fraction(own.kappa, 'signed') ? Object.freeze({ mode: 'well', hbar: own.hbar, kappa: own.kappa }) : null;
    if (!Number.isInteger(own.intervals) || own.intervals < 2 || own.intervals > 32 || Object.is(own.intervals, -0)
      || !Number.isInteger(own.probe) || Object.is(own.probe, -0) || own.probe < 0 || own.probe > own.intervals) return null;
    if (!fraction(own.time, 'positive') || !fraction(own.mass, 'positive') || !fraction(own.omegaSquared, 'nonnegative')
      || !fraction(own.quartic, 'nonnegative') || !fraction(own.start, 'signed') || !fraction(own.end, 'signed')) return null;
    return Object.freeze({ mode: 'chain', intervals: own.intervals, time: own.time, mass: own.mass, hbar: own.hbar,
      omegaSquared: own.omegaSquared, quartic: own.quartic, start: own.start, end: own.end, probe: own.probe });
  } catch { return null; }
}
export const saddleFluctuationDefault: Readonly<SaddleChainInput> = Object.freeze({ mode: 'chain', intervals: 2, time: '1', mass: '1', hbar: '1', omegaSquared: '1', quartic: '1', start: '0', end: '37/32', probe: 1 });
function chain(id: string, patch: Partial<SaddleChainInput>) {
  const input = readSaddleFluctuationInput({ ...saddleFluctuationDefault, ...patch });
  if (!input || input.mode !== 'chain') throw Error('Invalid saddle chain example: ' + id);
  return Object.freeze({ id, input });
}
function well(id: string, kappa: string, hbar = '1') {
  const input = readSaddleFluctuationInput({ mode: 'well', kappa, hbar });
  if (!input || input.mode !== 'well') throw Error('Invalid saddle well example: ' + id);
  return Object.freeze({ id, input });
}
export const saddleFluctuationExamples = Object.freeze([
  chain('one-variable-nonzero-hand', {}),
  chain('zero-endpoints-certified', { end: '0' }),
  chain('positive-first-correction', { end: '13/2' }),
  chain('coupled-known-saddle', { intervals: 3, time: '3', omegaSquared: '0', start: '5/8', end: '5/8' }),
  chain('four-coupled-known', { intervals: 4, time: '4', omegaSquared: '0', start: '3289/4096', end: '3289/4096', probe: 2 }),
  chain('six-coupled-known', { intervals: 6, time: '6', omegaSquared: '0', start: '107284647753/68719476736', end: '107284647753/68719476736', probe: 3 }),
  chain('eight-intervals', { intervals: 8, probe: 4 }),
  chain('maximum-thirty-two', { intervals: 32, probe: 16 }),
  chain('quadratic-reference', { intervals: 4, quartic: '0', end: '1', probe: 2 }),
  chain('free-linear-reference', { intervals: 32, time: '16', quartic: '0', omegaSquared: '0', start: '-16', end: '16', probe: 16 }),
  chain('stiff-positive-boundary-layer', { intervals: 32, time: '16', mass: '1/999999999989', omegaSquared: '0', quartic: '16', start: '16', end: '16', probe: 16 }),
  chain('stiff-sign-changing-layer', { intervals: 32, time: '16', mass: '1/999999999989', omegaSquared: '0', quartic: '16', start: '-16', end: '16', probe: 16 }),
  chain('minimum-positive-factors', { intervals: 32, time: '1/999999999989', mass: '1/999999999983', hbar: '1/999999999979', omegaSquared: '1/999999999961', quartic: '1/999999999959', start: '-16', end: '16', probe: 1 }),
  chain('maximum-coprime', { intervals: 32, time: '999999999983/62499999999', mass: '999999999979/62499999999', hbar: '999999999961/62499999999', omegaSquared: '999999999959/62499999999', quartic: '999999999953/62499999999', start: '-999999999929/62499999999', end: '999999999893/62499999999', probe: 17 }),
  chain('mixed-coprime-denominators', { intervals: 32, time: '999999999983/999999999989', mass: '999999999979/999999999961', hbar: '999999999953/999999999929', omegaSquared: '999999999893/999999999883', quartic: '999999999797/999999999761', start: '-999999999757/999999999749', end: '999999999727/999999999719', probe: 17 }),
  chain('fractional-model', { intervals: 7, time: '3/2', mass: '2/3', hbar: '5/4', omegaSquared: '7/5', quartic: '11/13', start: '-1/2', end: '5/4', probe: 3 }),
  chain('equivalent-spelling', { time: '2/2', mass: '3/3', hbar: '4/4', omegaSquared: '5/5', quartic: '6/6', start: '0/7' }),
  chain('reverse-endpoints', { start: '37/32', end: '0' }),
  chain('opposite-endpoints', { start: '1', end: '-1' }),
  chain('fixed-endpoint-probe', { intervals: 4, probe: 0 }),
  chain('double-hbar', { hbar: '2' }),
  chain('double-time', { time: '2' }),
  chain('double-mass', { mass: '2' }),
  chain('weak-quartic', { quartic: '1/999999999989' }),
  well('single-well', '-1'), well('degenerate-well', '0'), well('double-well', '1'),
  well('narrow-double-well', '16', '1/999999999989'), well('weak-well-curvature', '1/999999999989', '16'),
  well('narrow-single-well', '-16', '1/999999999989'), well('fractional-well', '-7/5', '5/4'), well('zero-curvature-spelling', '0/7'),
]);
