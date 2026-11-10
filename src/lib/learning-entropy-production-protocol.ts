/** Transport shape and preparation identity, not a proof of sender mathematics.
 * Numerical model import is type-only so readers do not load the engine. */
import type { threeStateEntropyProductionModel } from './learning-three-state-entropy-production';
import { isThreeStateEntropyParameters, snapshotEntropyParameters, type ThreeStateEntropyParameters } from './learning-entropy-production-input';
import { isMasterObservation } from './learning-master-equation-protocol';
import { parseNumberJsonDraft } from './learning-json-number-input';
export const entropyKind = 'stationary-jump-path-entropy-production' as const;
export type EntropyKind = typeof entropyKind;
export type EntropyInput = { kind: typeof entropyKind; parameters: ThreeStateEntropyParameters };
export type EntropySuccess = { version: 1; requestId: number; ok: true; kind: typeof entropyKind; input: EntropyInput; inputKey: string; result: ReturnType<typeof threeStateEntropyProductionModel> };
export type EntropyResponse = EntropySuccess | { version: 1; requestId: number | null; ok: false; error: 'invalid-request' | 'invalid-input' | 'calculation-failed' };
type Check = (x: unknown) => boolean;
export const entropyRecord = (x: unknown): x is Record<string, unknown> => !!x && typeof x === 'object' && Object.getPrototypeOf(x) === Object.prototype;
const shape = (x: unknown, fields: Record<string, Check>): x is Record<string, unknown> => entropyRecord(x) && Reflect.ownKeys(x).length === Object.keys(fields).length && Object.entries(fields).every(([k, c]) => Object.prototype.hasOwnProperty.call(x, k) && c(x[k]));
const literal = (x: unknown): Check => y => y === x;
const choice = (...xs: unknown[]): Check => y => xs.includes(y);
const integer = (x: unknown, lo = 0, hi = Number.MAX_SAFE_INTEGER): x is number => typeof x === 'number' && Number.isSafeInteger(x) && x >= lo && x <= hi;
const array = (n: number, c: Check): Check => x => Array.isArray(x) && Object.getPrototypeOf(x) === Array.prototype && x.length === n && Reflect.ownKeys(x).length === n + 1 && Array.from({ length: n }, (_, i) => Object.prototype.hasOwnProperty.call(x, i) && c(x[i])).every(Boolean);
const fields = (names: string, c: Check) => Object.fromEntries(names.split(' ').map(k => [k, c]));
const text: Check = x => typeof x === 'string' && x.length > 0 && x.length < 12000;
const same = (x: unknown, y: unknown) => { try { return JSON.stringify(x) === JSON.stringify(y); } catch { return false; } };
export const isEntropyInput = (x: unknown): x is EntropyInput => shape(x, { kind: literal(entropyKind), parameters: isThreeStateEntropyParameters });
export function snapshotEntropyInput(x: EntropyInput): EntropyInput {
  if (!isEntropyInput(x)) throw new RangeError('Invalid entropy preparation');
  return Object.freeze({ kind: entropyKind, parameters: snapshotEntropyParameters(x.parameters) });
}
export const entropyInputKey = (x: EntropyInput) => JSON.stringify(snapshotEntropyInput(x));
export const isEntropyRequestEnvelope = (x: unknown): x is { version: 1; requestId: number; input: unknown } => shape(x, { version: literal(1), requestId: x => integer(x, 1), input: () => true });
export function parseEntropyStoredInput(raw: string): EntropyInput | null {
  try {
    if (typeof raw !== 'string' || raw.length > 32768) return null;
    const value = parseNumberJsonDraft(raw, ['version', 'parameters']);
    return shape(value, { version: literal(1), value: isEntropyInput }) ? snapshotEntropyInput(value.value as EntropyInput) : null;
  } catch { return null; }
}
export function isEntropySuccessForInput(value: unknown, input: EntropyInput, id: number): value is EntropySuccess {
  if (!integer(id, 1) || !isEntropyInput(input) || !shape(value, { version: literal(1), requestId: literal(id), ok: literal(true), kind: literal(entropyKind), input: isEntropyInput, inputKey: literal(entropyInputKey(input)), result: entropyRecord }) || entropyInputKey(value.input as EntropyInput) !== entropyInputKey(input)) return false;
  const p = input.parameters, r = value.result as Record<string, unknown>, cache = new WeakMap<object, boolean>();
  const q: Check = x => { if (!entropyRecord(x)) return false; if (cache.has(x)) return cache.get(x)!; const good = isMasterObservation(x); cache.set(x, good); return good; };
  const exact: Check = x => isMasterObservation(x, true), vec = (c: Check = q) => array(3, c), matrix = (c: Check = q) => array(3, vec(c)), nullable = (c: Check): Check => x => x === null || c(x);
  const equilibrium = BigInt(p.a) * BigInt(p.b) * BigInt(p.c) === BigInt(p.d) * BigInt(p.e) * BigInt(p.f);
  const homogeneous = p.a === p.b && p.b === p.c && p.d === p.e && p.e === p.f;
  const states = [p.initialState];
  for (let k = 0; k < p.jumpCount; k++) states.push((states[k] + ((p.pathCode >> k) & 1 ? 2 : 1)) % 3);
  const row: Check = x => {
    if (!entropyRecord(x) || !integer(x.step, 0, p.sampleSteps)) return false;
    const zero = x.step === 0 || p.timeNumerator === 0, identity = zero || equilibrium || p.tiltNumerator === 0 || p.tiltNumerator === p.tiltDenominator;
    const reason = zero ? 'zero-time' : equilibrium ? 'detailed-balance-path-identity' : p.tiltNumerator === 0 ? 'zero-tilt-normalization' : p.tiltNumerator === p.tiltDenominator ? 'adjoint-normalization' : null;
    const status = identity ? literal('analytic-exact-one') : choice('directed-positive-exponential-enclosure', 'unresolved-positive-exponential-upper-bound');
    const valid = shape(x, { step: y => integer(y, 0, p.sampleSteps), time: exact, ...fields('stationaryShannonEntropy meanEntropy logMoment mirrorLogMoment logSymmetryDifference', q), ...fields('moment mirrorMoment', nullable(q)), ...fields('momentStatus mirrorMomentStatus', status), symmetryEnclosuresConsistent: literal(true), expectedLogMomentSign: literal(identity ? 0 : p.tiltNumerator > 0 && p.tiltNumerator < p.tiltDenominator ? -1 : 1),
      momentProof: y => shape(y, { method: literal(identity ? 'normalization-identity' : 'shifted-positive-series-log-normalization'), identityReason: literal(reason), squarings: y => integer(y, 0, 1024), retainedTerms: literal(identity ? 0 : 65), ...fields('baseTailUpperBound logScale', q), linearPrecisionBits: choice(256, 512, 1024, 2048) }),
      ...fields('homogeneousClosedLogMoment homogeneousEntropyVariance homogeneousClockwiseCountMean homogeneousReverseCountMean', homogeneous ? q : literal(null)), homogeneousCountReferenceStatus: literal(homogeneous ? 'homogeneous-marked-poisson-reference' : 'not-applicable-state-dependent-rates') });
    if (!valid) return false;
    return ['moment', 'mirrorMoment'].every(k => x[k] === null ? x[k + 'Status'] === 'unresolved-positive-exponential-upper-bound' : x[k + 'Status'] !== 'unresolved-positive-exponential-upper-bound' && entropyRecord(x[k]) && x[k].resolution !== 'structural-zero' && (x[k].value === null || (typeof x[k].value === 'number' && x[k].value > 0)));
  };
  const valid = shape(r, { parameters: x => same(x, snapshotEntropyParameters(p)), precisionBits: literal(256), ...fields('generator stationaryFlows adjointGenerator', matrix(exact)), ...fields('tiltedGenerator mirrorTiltedGenerator', matrix()), stationaryLaw: vec(exact), ...fields('stationaryCurrent cycleRateRatio stationaryShannonEntropyRate stationaryMassResidual', exact), ...fields('cycleAffinity entropyRate pairedEntropyRate stationaryShannonEntropy', q), adjointMassResidual: vec(exact),
    edges: array(3, x => shape(x, { from: y => integer(y, 0, 2), to: y => integer(y, 0, 2), ...fields('forwardFlow reverseFlow current', exact), ...fields('logFluxRatio entropyRate', q) })),
    path: x => shape(x, { states: y => same(y, states), reversedStates: y => same(y, [...states].reverse()), jumpTimes: array(p.jumpCount, exact), equalDwellTime: exact, jumps: array(p.jumpCount, y => shape(y, { index: y => integer(y, 0, p.jumpCount - 1), ...fields('from to', y => integer(y, 0, 2)), clockwise: choice(true, false), ...fields('rateRatio fluxRatio', exact), ...fields('rateOnlyIncrement entropyIncrement', q) })), ...fields('initialWeight finalStationaryWeight survivalAction rateRatio boundaryRatio totalRatio', exact), ...fields('rateOnlyEntropy systemBoundaryEntropy totalEntropy totalViaBoundarySum reversedTotalEntropy logForwardDensityInSeconds logReversedDensityInSeconds logDensityDifference', q), totalEntropyExactSign: choice(-1, 0, 1), ...fields('densityReference construction', text) }),
    history: array(p.sampleSteps + 1, row), current: row, approximationContract: text,
    classification: x => shape(x, { detailedBalance: literal(equilibrium), entropyRateExactSign: literal(equilibrium ? 0 : 1), homogeneousRates: literal(homogeneous), ...fields('evenTimeReversalStates autonomousRates stationaryInitialization allReverseEdgesPositive entropyMeasuredInNats momentNormalizationIdentityAtZeroAndOne finiteTimeSymmetryAppliesToSpecifiedExperiment productionUIImplemented', literal(true)), ...fields('thermalBathSpecified sampledPaths pathLawOrTiltRenormalized', literal(false)) }) });
  if (!valid) return false;
  const history = r.history as Record<string, unknown>[], path = r.path as Record<string, unknown>;
  return history.every((x, i) => x.step === i) && same(r.current, history[history.length - 1])
    && (r.edges as { from: number; to: number }[]).every((x, i) => x.from === i && x.to === (i + 1) % 3)
    && (path.jumps as { index: number; from: number; to: number; clockwise: boolean }[]).every((x, i) => x.index === i && x.from === states[i] && x.to === states[i + 1] && x.clockwise === (states[i + 1] === (states[i] + 1) % 3));
}
