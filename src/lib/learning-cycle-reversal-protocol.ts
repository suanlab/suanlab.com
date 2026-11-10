/** Strict transport validation only; numerical evaluation stays in the worker. */
import type { ContinuousCycleReversalParameters, continuousCycleReversalModel } from './learning-continuous-cycle-reversal';
import { isMasterObservation } from './learning-master-equation-protocol';
import { parseNumberJsonDraft } from './learning-json-number-input';
export const reversalKinds = ['continuous-cycle-path-reversal'] as const;
export type ReversalKind = typeof reversalKinds[number];
export type ReversalInput = { kind: ReversalKind; parameters: ContinuousCycleReversalParameters };
export type ReversalSuccess = { version: 1; requestId: number; ok: true; kind: ReversalKind; input: ReversalInput; inputKey: string; result: ReturnType<typeof continuousCycleReversalModel> };
export type ReversalResponse = ReversalSuccess | { version: 1; requestId: number | null; ok: false; error: 'invalid-request' | 'invalid-input' | 'calculation-failed' };
export const reversalRecord = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && Object.getPrototypeOf(v) === Object.prototype;
type Check = (v: unknown) => boolean;
const shape = (v: unknown, s: Record<string, Check>): v is Record<string, unknown> => reversalRecord(v)
  && Reflect.ownKeys(v).length === Object.keys(s).length && Object.entries(s).every(([k, c]) => Object.prototype.hasOwnProperty.call(v, k) && c(v[k]));
const integer = (v: unknown, min = 1, max = 1000000): v is number => typeof v === 'number' && Number.isSafeInteger(v) && v >= min && v <= max;
const lit = (value: unknown): Check => v => v === value;
const array = (n: number, check: Check): Check => v => Array.isArray(v) && Object.getPrototypeOf(v) === Array.prototype
  && v.length === n && Reflect.ownKeys(v).length === n + 1 && Array.from({ length: n }, (_, i) => Object.prototype.hasOwnProperty.call(v, i) && check(v[i])).every(Boolean);
const fields = (keys: string, check: Check) => Object.fromEntries(keys.split(' ').map(k => [k, check]));
const same = (a: unknown, b: unknown) => { try { return JSON.stringify(a) === JSON.stringify(b); } catch { return false; } };
export const reversalScalarKeys = ['clockwise01','clockwise12','clockwise20','counter10','counter21','counter02','rateDenominator','timeDenominator'] as const;
export function isReversalInput(v: unknown): v is ReversalInput {
  if (!shape(v, { kind: lit(reversalKinds[0]), parameters: reversalRecord })) return false;
  const p = v.parameters as Record<string, unknown>, states = p.states;
  if (!Array.isArray(states) || states.length < 1 || states.length > 65) return false;
  return shape(p, { ...Object.fromEntries(reversalScalarKeys.map(k => [k, integer])),
    states: array(states.length, x => integer(x, 0, 2)), dwellNumerators: array(states.length, x => integer(x, 0)) })
    && states.every((x, i) => i === 0 || x !== states[i-1])
    && (p.dwellNumerators as number[]).every((x, i) => i === states.length - 1 || x > 0);
}
export function snapshotReversalInput(input: ReversalInput): ReversalInput {
  if (!isReversalInput(input)) throw new RangeError('Invalid continuous cycle preparation');
  const p = input.parameters;
  return Object.freeze({ kind: input.kind, parameters: Object.freeze({
    ...Object.fromEntries(reversalScalarKeys.map(k => [k, p[k] + 0])),
    states: Object.freeze(p.states.map(n => n + 0)), dwellNumerators: Object.freeze(p.dwellNumerators.map(n => n + 0)),
  }) }) as unknown as ReversalInput;
}
export const reversalInputKey = (input: ReversalInput) => JSON.stringify(snapshotReversalInput(input));
export const isReversalRequestEnvelope = (v: unknown): v is { version: 1; requestId: number; input: unknown } =>
  shape(v, { version: lit(1), requestId: x => integer(x, 1, Number.MAX_SAFE_INTEGER), input: () => true });
export function parseReversalStoredInput(raw: string, kind: ReversalKind): ReversalInput | null {
  try {
    if (typeof raw !== 'string' || raw.length > 32768) return null;
    const v = parseNumberJsonDraft(raw, ['version','parameters']);
    return shape(v, { version: lit(1), value: isReversalInput }) && (v.value as ReversalInput).kind === kind ? snapshotReversalInput(v.value as ReversalInput) : null;
  } catch { return null; }
}
/** This guard checks shape, identity and availability, not scientific truth. */
export function isReversalSuccessForInput(v: unknown, input: ReversalInput, id: number): v is ReversalSuccess {
  if (!integer(id, 1, Number.MAX_SAFE_INTEGER) || !isReversalInput(input) || !shape(v, {
    version: lit(1), requestId: lit(id), ok: lit(true), kind: lit(input.kind), input: isReversalInput,
    inputKey: lit(reversalInputKey(input)), result: reversalRecord,
  }) || reversalInputKey(v.input as ReversalInput) !== reversalInputKey(input)) return false;
  const p = input.parameters, n = p.states.length, cache = new WeakMap<object, boolean>();
  const q: Check = x => { if (!reversalRecord(x)) return false; if (cache.has(x)) return cache.get(x)!; const good = isMasterObservation(x); cache.set(x, good); return good; };
  const exact: Check = x => isMasterObservation(x, true), vec = (k: number, c: Check = exact) => array(k, c), matrix = array(3, vec(3));
  const row: Check = x => shape(x, { index: x => integer(x, 0, n-1), state: x => integer(x, 0, 2),
    ...fields('dwell elapsed integratedEscape endpointRatio jumpRateRatio pathRatio', exact),
    ...fields('endpointLogRatio jumpLogRatio totalLogRatio logForwardDensity logReverseDensity', q), pathRatioIsExactlyOne: x => typeof x === 'boolean' });
  const r = v.result as Record<string, unknown>, forward = BigInt(p.clockwise01)*BigInt(p.clockwise12)*BigInt(p.clockwise20), backward = BigInt(p.counter10)*BigInt(p.counter21)*BigInt(p.counter02);
  const good = shape(r, { parameters: x => same(x, snapshotReversalInput(input).parameters), precisionBits: lit(256),
    generator: matrix, stationary: vec(3), treeWeights: vec(3), treeNormalization: exact, escapeRates: vec(3), reversedGenerator: matrix,
    orientedEdges: array(3, x => shape(x, { from: x => integer(x,0,2), to: x => integer(x,0,2),
      ...fields('forwardFlow backwardFlow netCurrent stationaryFlowRatio', exact), ...fields('logStationaryFlowRatio relativeEntropyContribution', q) })),
    ...fields('stationaryResidual generatorRowResidual reversedRowResidual jumpEpochStationary', vec(3)),
    ...fields('forwardCycleRateProduct backwardCycleRateProduct cycleRatio cycleCurrent stationaryActivity totalTime', exact),
    ...fields('cycleAffinity pathRelativeEntropyRate', q),
    chosenPathOccupation: p.dwellNumerators.every(x => x === 0) ? lit(null) : vec(3), path: array(n, row), current: row,
    classification: x => shape(x, { irreducible: lit(true), allStatesEvenUnderTimeReversal: lit(true), detailedBalance: lit(forward === backward),
      stationaryIsUniform: x => typeof x === 'boolean', clockwiseCurrent: lit(forward > backward), counterclockwiseCurrent: lit(forward < backward),
      forwardPathFavored: x => typeof x === 'boolean', pathRatioIsExactlyOne: x => typeof x === 'boolean', chosenPathIsSampled: lit(false), thermalInterpretationSpecified: lit(false) }),
    approximationContract: x => typeof x === 'string' && x.length > 0 && x.length < 12000 });
  if (!good || !same(r.current, (r.path as unknown[]).at(-1))) return false;
  return (r.path as {index: number;state: number}[]).every((row, i) => row.index === i && row.state === p.states[i])
    && (r.orientedEdges as {from: number;to: number}[]).every((edge, i) => edge.from === i && edge.to === (i+1)%3);
}
