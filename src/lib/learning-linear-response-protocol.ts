/** Reader-safe validation and identity; numerical models are imported as types only. */
import type { FiniteStateResponseExperimentParameters, finiteStateResponseExperiment } from './learning-finite-state-response-experiment';
import { isMasterObservation } from './learning-master-equation-protocol';
import { parseNumberJsonDraft } from './learning-json-number-input';
export const lrKinds = ['markov-perturbation-linear-response'] as const;
export type LrKind = typeof lrKinds[number];
export type LrInput = { kind: LrKind; parameters: FiniteStateResponseExperimentParameters };
export type LrSuccess = { version: 1; requestId: number; ok: true; kind: LrKind; input: LrInput; inputKey: string; result: ReturnType<typeof finiteStateResponseExperiment> };
export type LrResponse = LrSuccess | { version: 1; requestId: number | null; ok: false; error: 'invalid-request' | 'invalid-input' | 'calculation-failed' };
export const lrParameterKeys = ['a', 'b', 'c', 'd', 'e', 'f', 'rateDenominator', 'kineticNumerator', 'kineticDenominator', 'frequencyNumerator', 'frequencyDenominator', 'timeNumerator', 'timeDenominator', 'sampleSteps', 'amplitudeNumerator', 'amplitudeDenominator'] as const;
const baseKeys = lrParameterKeys.slice(0, 11);
export const lrRecord = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && Object.getPrototypeOf(v) === Object.prototype;
type Check = (v: unknown) => boolean;
const shape = (v: unknown, fields: Record<string, Check>): v is Record<string, unknown> => lrRecord(v) && Reflect.ownKeys(v).length === Object.keys(fields).length && Object.entries(fields).every(([k, c]) => Object.prototype.hasOwnProperty.call(v, k) && c(v[k]));
const integer = (x: unknown, lo = 1, hi = 1000000): x is number => typeof x === 'number' && Number.isSafeInteger(x) && x >= lo && x <= hi;
const lit = (x: unknown): Check => v => v === x;
const array = (n: number, check: Check): Check => v => Array.isArray(v) && Object.getPrototypeOf(v) === Array.prototype && v.length === n && Reflect.ownKeys(v).length === n + 1 && Array.from({ length: n }, (_, i) => Object.prototype.hasOwnProperty.call(v, i) && check(v[i])).every(Boolean);
const fields = (names: string, check: Check) => Object.fromEntries(names.split(' ').map(k => [k, check]));
const same = (a: unknown, b: unknown) => { try { return JSON.stringify(a) === JSON.stringify(b); } catch { return false; } };
const text: Check = x => typeof x === 'string' && x.length > 0 && x.length < 12000;
const bool: Check = x => typeof x === 'boolean';

export function isLrInput(value: unknown): value is LrInput {
  if (!shape(value, { kind: lit(lrKinds[0]), parameters: lrRecord })) return false;
  const p = value.parameters as Record<string, number>;
  return shape(p, Object.fromEntries(lrParameterKeys.map(k => [k, (x: unknown) => integer(x, ['kineticNumerator', 'amplitudeNumerator'].includes(k) ? -1000000 : ['frequencyNumerator', 'timeNumerator'].includes(k) ? 0 : 1, k === 'sampleSteps' ? 64 : 1000000)]))) && p.timeNumerator <= 4 * p.timeDenominator && Math.abs(p.amplitudeNumerator) <= p.amplitudeDenominator;
}
export function snapshotLrInput(value: LrInput): LrInput {
  if (!isLrInput(value)) throw new RangeError('Invalid linear response preparation');
  return Object.freeze({ kind: value.kind, parameters: Object.freeze(Object.fromEntries(lrParameterKeys.map(k => [k, value.parameters[k] + 0]))) }) as unknown as LrInput;
}
export const lrInputKey = (value: LrInput) => JSON.stringify(snapshotLrInput(value));
export const isLrRequestEnvelope = (v: unknown): v is { version: 1; requestId: number; input: unknown } => shape(v, { version: lit(1), requestId: x => integer(x, 1, Number.MAX_SAFE_INTEGER), input: () => true });
export function parseLrStoredInput(raw: string, kind: LrKind): LrInput | null {
  try {
    if (typeof raw !== 'string' || raw.length > 32768) return null;
    const v = parseNumberJsonDraft(raw, ['version', 'parameters']);
    return shape(v, { version: lit(1), value: isLrInput }) && (v.value as LrInput).kind === kind ? snapshotLrInput(v.value as LrInput) : null;
  } catch { return null; }
}
export function isLrSuccessForInput(value: unknown, input: LrInput, id: number): value is LrSuccess {
  if (!integer(id, 1, Number.MAX_SAFE_INTEGER) || !isLrInput(input) || !shape(value, { version: lit(1), requestId: lit(id), ok: lit(true), kind: lit(input.kind), input: isLrInput, inputKey: lit(lrInputKey(input)), result: lrRecord }) || lrInputKey(value.input as LrInput) !== lrInputKey(input)) return false;
  const p = input.parameters, r = value.result as Record<string, unknown>, cache = new WeakMap<object, boolean>();
  const q: Check = x => { if (!lrRecord(x)) return false; if (cache.has(x)) return cache.get(x)!; const good = isMasterObservation(x); cache.set(x, good); return good; };
  const exact: Check = x => isMasterObservation(x, true), vec = (c: Check = q) => array(3, c), matrix = (c: Check = q) => array(3, vec(c));
  const reference: Check = x => shape(x, {
    parameters: x => same(x, Object.fromEntries(baseKeys.map(k => [k, p[k]]))), stage: lit('static-and-frequency-preparation'),
    ...fields('generator generatorDerivative', matrix(exact)), ...fields('stationaryLaw signedSource conjugateObservable sourceProtocolContribution', vec(exact)),
    ...fields('stationaryCurrent sourceMassResidual stationaryDerivativeMassResidual', exact),
    observables: array(3, x => shape(x, { state: x => integer(x, 0, 2), ...fields('stationaryOccupancy initialKernel stationaryDerivative staticSusceptibility staticIdentityResidual', exact), poissonSolution: vec(exact), frequencySusceptibility: x => shape(x, { ...fields('real imaginary magnitudeSquared', exact), phaseDefined: bool }) })),
    classification: x => shape(x, { strictlyPositiveRates: lit(true), detailedBalance: bool, stationaryInitialization: lit(true), frequencyIsAngular: lit(true), frequencyUsesStaticLimit: lit(p.frequencyNumerator === 0), sampledPaths: lit(false), thermalBathSpecified: lit(false), timeDomainImplemented: lit(false), finiteAmplitudeImplemented: lit(false) }), approximationContract: text,
  });
  const proof = (zero: boolean): Check => x => shape(x, { squarings: y => integer(y, 0, 1024), retainedPoissonTerms: lit(zero ? 1 : 65), ...fields('basePoissonMean baseTailUpperBound', exact), method: lit(zero ? 'exact-zero-time' : 'positive-uniformization-scaling-squaring') });
  const row: Check = x => {
    if (!lrRecord(x) || !integer(x.step, 0, p.sampleSteps)) return false;
    const zero = x.step === 0 || p.timeNumerator === 0, unchanged = zero || p.amplitudeNumerator === 0;
    if (!shape(x, { step: y => integer(y, 0, p.sampleSteps), time: exact, ...fields('kernel stepSusceptibility predictedOccupancy', vec()), actualOccupancy: y => y === null || vec()(y), actualMinusLinearPrediction: y => y === null || vec()(y), finiteInputStatus: y => ['exact-unchanged-law', 'certified-enclosure', 'unresolved-positive-rate-reciprocal'].includes(y as string), baselineProof: proof(zero), finiteInputProof: y => y === null || proof(false)(y) })) return false;
    if (unchanged) return x.finiteInputStatus === 'exact-unchanged-law' && x.actualOccupancy !== null && x.actualMinusLinearPrediction !== null && x.finiteInputProof === null;
    return x.finiteInputStatus === 'certified-enclosure' ? x.actualOccupancy !== null && x.actualMinusLinearPrediction !== null && x.finiteInputProof !== null : x.finiteInputStatus === 'unresolved-positive-rate-reciprocal' && x.actualOccupancy === null && x.actualMinusLinearPrediction === null && x.finiteInputProof === null;
  };
  if (!shape(r, { parameters: x => same(x, snapshotLrInput(input).parameters), precisionBits: lit(256), reference, amplitude: exact, history: array(p.sampleSteps + 1, row), current: row, perturbedRates: array(6, x => x === null || q(x)), classification: x => shape(x, { timeDomainImplemented: lit(true), finiteAmplitudeImplemented: lit(true), stationaryInitialization: lit(true), allMathematicalFiniteInputRatesPositive: lit(true), finiteInputRatesEnclosed: bool, sourceExactlyZero: bool, identicallyZeroResponseStates: array(3, bool), normalizationOfTruncation: lit(false), sampledPaths: lit(false), eulerTimeStepping: lit(false) }), approximationContract: text })) return false;
  const result = r as unknown as LrSuccess['result'];
  const rationalIs = (x: LrSuccess['result']['amplitude'], n: bigint, d: bigint) => {
    const b = x.bounds;
    return !!b && same(b.lower, b.upper) && BigInt(b.lower.numerator.coefficient) * d === n * BigInt(b.lower.denominator.coefficient);
  };
  const zero = (x: LrSuccess['result']['amplitude']) => x.resolution === 'structural-zero';
  if (!rationalIs(result.amplitude, BigInt(p.amplitudeNumerator), BigInt(p.amplitudeDenominator)) || result.classification.sourceExactlyZero !== result.reference.signedSource.every(zero)) return false;
  if (result.classification.sourceExactlyZero && !result.classification.identicallyZeroResponseStates.every(Boolean)) return false;
  if (!result.history.every(x => {
    const zeroTime = x.step === 0 || p.timeNumerator === 0, unchanged = zeroTime || p.amplitudeNumerator === 0;
    if (!rationalIs(x.time, BigInt(p.timeNumerator) * BigInt(x.step), BigInt(p.timeDenominator) * BigInt(p.sampleSteps))) return false;
    if (zeroTime && (!same(x.kernel, result.reference.signedSource) || !x.stepSusceptibility.every(zero))) return false;
    if (unchanged && (!same(x.actualOccupancy, result.reference.stationaryLaw) || !same(x.predictedOccupancy, result.reference.stationaryLaw) || !x.actualMinusLinearPrediction?.every(zero))) return false;
    return result.classification.identicallyZeroResponseStates.every((isZero, j) => !isZero || zero(x.kernel[j]) && zero(x.stepSusceptibility[j]) && same(x.predictedOccupancy[j], result.reference.stationaryLaw[j]));
  })) return false;
  return result.history.every((x, i) => x.step === i) && same(result.current, result.history.at(-1)) && result.reference.observables.every((x, i) => x.state === i && x.frequencySusceptibility.phaseDefined === (x.frequencySusceptibility.real.sign !== 0 || x.frequencySusceptibility.imaginary.sign !== 0))
    && result.classification.finiteInputRatesEnclosed === result.perturbedRates.every(x => x !== null)
    && result.history.every(x => x.finiteInputStatus === 'exact-unchanged-law' || (x.finiteInputStatus === 'certified-enclosure') === result.classification.finiteInputRatesEnclosed);
}
