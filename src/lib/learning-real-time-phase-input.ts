/** Reader-safe owned inputs; exact algebra and transcendental work stay in Workers. */
export interface RealTimeChainInput {
  mode: 'chain'; intervals: number; time: string; mass: string; hbar: string;
  omegaSquared: string; start: string; end: string; damping: string; probe: number;
}
export interface RealTimePacketInput {
  mode: 'packet'; intervals: number; time: string; mass: string; hbar: string;
  omegaSquared: string; position: string; momentum: string; chirp: string;
  width: string; widthMode: 'fixed-variance' | 'hbar-scaled-variance';
}
export interface RealTimeFoldInput { mode: 'fold'; control: string; hbar: string; }
export type RealTimePhaseInput = Readonly<RealTimeChainInput | RealTimePacketInput | RealTimeFoldInput>;
const chainKeys = ['mode', 'intervals', 'time', 'mass', 'hbar', 'omegaSquared', 'start', 'end', 'damping', 'probe'] as const;
const packetKeys = ['mode', 'intervals', 'time', 'mass', 'hbar', 'omegaSquared', 'position', 'momentum', 'chirp', 'width', 'widthMode'] as const;
const foldKeys = ['mode', 'control', 'hbar'] as const;
function fraction(value: unknown, sign: 'positive' | 'nonnegative' | 'signed'): value is string {
  // All products are <=16*10^12, so these comparisons are exact safe integers.
  if (typeof value !== 'string' || !/^-?(?:0|[1-9]\d{0,11})(?:\/[1-9]\d{0,11})?$/.test(value) || /^-0(?:\/|$)/.test(value)) return false;
  const [numerator, denominator = '1'] = value.split('/');
  const n = Number(numerator), d = Number(denominator);
  return Math.abs(n) <= 16*d && (sign === 'signed' || (sign === 'positive' ? n > 0 : n >= 0));
}
export function readRealTimePhaseInput(value: unknown): RealTimePhaseInput | null {
  try {
    if (!value || typeof value !== 'object' || Object.getPrototypeOf(value) !== Object.prototype) return null;
    const descriptors = Object.getOwnPropertyDescriptors(value), mode = descriptors.mode;
    if (!mode || !Object.hasOwn(mode, 'value')) return null;
    const keys = mode.value === 'chain' ? chainKeys : mode.value === 'packet' ? packetKeys : mode.value === 'fold' ? foldKeys : null;
    if (!keys || Reflect.ownKeys(descriptors).length !== keys.length || keys.some(key => !descriptors[key] || !Object.hasOwn(descriptors[key], 'value'))) return null;
    const own = Object.fromEntries(keys.map(key => [key, descriptors[key].value]));
    if (!fraction(own.hbar, 'positive')) return null;
    if (own.mode === 'fold') return fraction(own.control, 'signed') ? Object.freeze({mode: 'fold', control: own.control, hbar: own.hbar}) : null;
    if (!Number.isInteger(own.intervals) || Object.is(own.intervals, -0) || own.intervals < 2 || own.intervals > 32
      || !fraction(own.time, 'positive') || !fraction(own.mass, 'positive') || !fraction(own.omegaSquared, 'signed')) return null;
    if (own.mode === 'chain') {
      if (!Number.isInteger(own.probe) || Object.is(own.probe, -0) || own.probe < 0 || own.probe > own.intervals
        || !fraction(own.start, 'signed') || !fraction(own.end, 'signed') || !fraction(own.damping, 'nonnegative')) return null;
      return Object.freeze({mode: 'chain', intervals: own.intervals, time: own.time, mass: own.mass, hbar: own.hbar,
        omegaSquared: own.omegaSquared, start: own.start, end: own.end, damping: own.damping, probe: own.probe});
    }
    if (!fraction(own.position, 'signed') || !fraction(own.momentum, 'signed') || !fraction(own.chirp, 'signed')
      || !fraction(own.width, 'positive') || !['fixed-variance', 'hbar-scaled-variance'].includes(own.widthMode)) return null;
    return Object.freeze({mode: 'packet', intervals: own.intervals, time: own.time, mass: own.mass, hbar: own.hbar,
      omegaSquared: own.omegaSquared, position: own.position, momentum: own.momentum, chirp: own.chirp,
      width: own.width, widthMode: own.widthMode});
  } catch { return null; }
}
export const realTimeChainDefault: Readonly<RealTimeChainInput> = Object.freeze({mode: 'chain', intervals: 4, time: '1', mass: '1', hbar: '1/2', omegaSquared: '1', start: '0', end: '1', damping: '0', probe: 2});
export const realTimePacketDefault: Readonly<RealTimePacketInput> = Object.freeze({mode: 'packet', intervals: 4, time: '1', mass: '1', hbar: '1/2', omegaSquared: '0', position: '0', momentum: '1', chirp: '0', width: '2/3', widthMode: 'fixed-variance'});
export const realTimeFoldDefault: Readonly<RealTimeFoldInput> = Object.freeze({mode: 'fold', control: '1', hbar: '1/2'});
function example<T extends RealTimePhaseInput>(id: string, value: T) {
  const input = readRealTimePhaseInput(value);
  if (!input) throw Error('Invalid real-time example: '+id);
  return Object.freeze({id, input});
}
const chain = (id: string, patch: Partial<RealTimeChainInput>) => example(id, {...realTimeChainDefault, ...patch});
const packet = (id: string, patch: Partial<RealTimePacketInput>) => example(id, {...realTimePacketDefault, ...patch});
const fold = (id: string, patch: Partial<RealTimeFoldInput>) => example(id, {...realTimeFoldDefault, ...patch});
export const realTimePhaseExamples = Object.freeze([
  chain('harmonic-four', {}),
  chain('free-thirty-two', {intervals:32,time:'16',omegaSquared:'0',start:'-16',end:'16',probe:16}),
  chain('negative-mode-hand', {intervals:2,time:'2',omegaSquared:'3',probe:1}),
  chain('zero-leading-pivot-invertible', {intervals:3,time:'3',omegaSquared:'2',start:'1',end:'0',probe:1}),
  chain('caustic-compatible', {intervals:2,time:'2',omegaSquared:'2',start:'1',end:'-1',probe:1}),
  chain('caustic-incompatible', {intervals:2,time:'2',omegaSquared:'2',start:'1',end:'0',probe:1}),
  chain('caustic-thirty-two', {intervals:32,time:'16',omegaSquared:'8',start:'1/3',end:'1/3',probe:16}),
  chain('near-caustic-left', {intervals:32,time:'16',mass:'1/999999999989',hbar:'1/999999999983',omegaSquared:'799999999908/99999999989',start:'-16',end:'16',probe:16}),
  chain('near-caustic-right', {intervals:32,time:'16',mass:'1/999999999989',hbar:'1/999999999983',omegaSquared:'799999999916/99999999989',start:'-16',end:'16',probe:16}),
  chain('physical-inverted', {intervals:8,time:'4',omegaSquared:'-4',probe:4}),
  chain('coarse-positive-hyperbolic', {intervals:2,time:'16',omegaSquared:'16',probe:1}),
  chain('regulated-free-branch', {time:'4',omegaSquared:'0',start:'0',end:'0',damping:'1/999999999989'}),
  chain('regulated-singular-thirty-two', {intervals:32,time:'16',omegaSquared:'8',damping:'1/999999999989',probe:16}),
  chain('minimum-positive-factors', {intervals:32,time:'1/999999999989',mass:'1/999999999983',hbar:'1/999999999979',omegaSquared:'-16',start:'-16',end:'16',damping:'1/999999999961',probe:31}),
  chain('maximum-mixed-coprime', {intervals:32,time:'999999999983/999999999989',mass:'999999999979/999999999961',hbar:'999999999953/999999999929',omegaSquared:'-999999999893/999999999883',start:'-999999999757/999999999749',end:'999999999727/999999999719',damping:'999999999797/999999999761',probe:17}),
  chain('equivalent-spelling', {time:'2/2',mass:'3/3',hbar:'4/8',omegaSquared:'5/5',start:'0/7'}),
  chain('endpoint-probe', {probe:0}),
  packet('fixed-width-free', {}),
  packet('scaled-width-free', {widthMode:'hbar-scaled-variance'}),
  packet('chirp-focus-ensemble', {chirp:'-1',hbar:'1/999999999989'}),
  packet('chirp-focus-localized', {chirp:'-1',hbar:'1/999999999989',widthMode:'hbar-scaled-variance'}),
  packet('harmonic-parity', {intervals:2,time:'2',omegaSquared:'2'}),
  packet('harmonic-full-cycle', {intervals:3,time:'3',omegaSquared:'3'}),
  packet('inverted-packet', {intervals:32,time:'16',omegaSquared:'-16',chirp:'16',position:'-16',momentum:'16'}),
  packet('minimum-width', {intervals:32,time:'1/999999999989',mass:'1/999999999983',hbar:'1/999999999979',width:'1/999999999961',widthMode:'hbar-scaled-variance',chirp:'-16'}),
  fold('two-real-stationary-points', {}),
  fold('merging-stationary-points', {control:'0'}),
  fold('no-real-stationary-points', {control:'-1'}),
  fold('maximum-positive-physical-control', {control:'16',hbar:'1/999999999989'}),
  fold('maximum-negative-physical-control', {control:'-16',hbar:'1/999999999989'}),
  fold('tiny-positive-control', {control:'1/999999999989',hbar:'16'}),
  fold('tiny-negative-control', {control:'-1/999999999989',hbar:'16'}),
]);
