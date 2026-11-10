/** Reader-safe transport validation, not a proof of a sender's mathematics. */
import { drivenThermalParameterKeys, reservoirThermalParameterKeys, isDrivenThermalParameters, isReservoirThermalParameters, type DrivenThermalParameters, type ReservoirThermalParameters } from './learning-thermal-input';
import type { drivenTwoStateThermodynamics } from './learning-driven-two-state-thermodynamics';
import type { reservoirTwoStateThermodynamics } from './learning-reservoir-two-state-thermodynamics';
import { isMasterObservation } from './learning-master-equation-protocol';
import { parseNumberJsonDraft } from './learning-json-number-input';
export const thermalKinds = ['driven-two-state-heat-work', 'reservoir-resolved-two-state-heat-flow'] as const;
export type ThermalInput = { kind: typeof thermalKinds[0]; parameters: DrivenThermalParameters } | { kind: typeof thermalKinds[1]; parameters: ReservoirThermalParameters };
export type ThermalSuccess = { version: 1; requestId: number; ok: true; inputKey: string } & (
  { kind: typeof thermalKinds[0]; input: Extract<ThermalInput, { kind: typeof thermalKinds[0] }>; result: ReturnType<typeof drivenTwoStateThermodynamics> } |
  { kind: typeof thermalKinds[1]; input: Extract<ThermalInput, { kind: typeof thermalKinds[1] }>; result: ReturnType<typeof reservoirTwoStateThermodynamics> });
export type ThermalResponse = ThermalSuccess | { version: 1; requestId: number | null; ok: false; error: 'invalid-request' | 'invalid-input' | 'calculation-failed' };
type Check = (x: unknown) => boolean;
export const thermalRecord = (x: unknown): x is Record<string, unknown> => !!x && typeof x === 'object' && Object.getPrototypeOf(x) === Object.prototype;
const shape = (x: unknown, spec: Record<string, Check>): x is Record<string, unknown> => thermalRecord(x) && Reflect.ownKeys(x).length === Object.keys(spec).length && Object.entries(spec).every(([k, c]) => Object.hasOwn(x, k) && c(x[k]));
const literal = (value: unknown): Check => x => x === value;
const choice = (...values: unknown[]): Check => x => values.includes(x);
const bool: Check = x => typeof x === 'boolean';
const integer = (x: unknown, lo = 0, hi = Number.MAX_SAFE_INTEGER): x is number => typeof x === 'number' && Number.isSafeInteger(x) && x >= lo && x <= hi;
const array = (length: number, check: Check): Check => x => Array.isArray(x) && Object.getPrototypeOf(x) === Array.prototype && x.length === length && Reflect.ownKeys(x).length === length + 1 && Array.from({ length }, (_, i) => Object.hasOwn(x, i) && check(x[i])).every(Boolean);
const fields = (names: string, check: Check) => Object.fromEntries(names.split(' ').map(k => [k, check]));
const nullable = (check: Check): Check => x => x === null || check(x);
const text: Check = x => typeof x === 'string' && x.length > 0 && x.length < 12000;
const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);
export function isThermalInput(x: unknown): x is ThermalInput {
  return shape(x, { kind: choice(...thermalKinds), parameters: thermalRecord }) && (x.kind === thermalKinds[0] ? isDrivenThermalParameters(x.parameters) : isReservoirThermalParameters(x.parameters));
}
export function snapshotThermalInput(input: ThermalInput): ThermalInput {
  if (!isThermalInput(input)) throw new RangeError('Invalid thermal preparation');
  const keys = input.kind === thermalKinds[0] ? drivenThermalParameterKeys : reservoirThermalParameterKeys;
  const p = input.parameters as unknown as Record<string, number>;
  return Object.freeze({ kind: input.kind, parameters: Object.freeze(Object.fromEntries(keys.map(k => [k, p[k] + 0]))) }) as unknown as ThermalInput;
}
export const thermalInputKey = (input: ThermalInput) => JSON.stringify(snapshotThermalInput(input));
export const isThermalRequestEnvelope = (x: unknown): x is { version: 1; requestId: number; input: unknown } => shape(x, { version: literal(1), requestId: x => integer(x, 1), input: () => true });
export function parseThermalInputDraft(raw: string): ThermalInput | null {
  try {
    if (typeof raw !== 'string' || raw.length > 32768) return null;
    const x = parseNumberJsonDraft(raw, ['parameters']);
    return isThermalInput(x) ? snapshotThermalInput(x) : null;
  } catch { return null; }
}
export function parseThermalStoredInput(raw: string): ThermalInput | null {
  try {
    if (typeof raw !== 'string' || raw.length > 32768) return null;
    const x = parseNumberJsonDraft(raw, ['version', 'parameters']);
    return shape(x, { version: literal(1), value: isThermalInput }) ? snapshotThermalInput(x.value as ThermalInput) : null;
  } catch { return null; }
}
export function isThermalSuccessForInput(value: unknown, input: ThermalInput, id: number): value is ThermalSuccess {
  if (!integer(id, 1) || !isThermalInput(input) || !shape(value, { version: literal(1), requestId: literal(id), ok: literal(true), kind: literal(input.kind), input: isThermalInput, inputKey: literal(thermalInputKey(input)), result: thermalRecord }) || thermalInputKey(value.input as ThermalInput) !== thermalInputKey(input)) return false;
  const r = value.result as Record<string, unknown>, cache = new WeakMap<object, boolean>();
  const q: Check = x => { if (!thermalRecord(x)) return false; if (cache.has(x)) return cache.get(x)!; const valid = isMasterObservation(x); cache.set(x, valid); return valid; };
  const exact: Check = x => isMasterObservation(x, true), vec = (c: Check = q) => array(2, c), matrix = (c: Check = q) => array(2, vec(c));
  const common = { parameters: x => (input.kind === thermalKinds[0] ? isDrivenThermalParameters(x) : isReservoirThermalParameters(x)) && same(x, snapshotThermalInput(input).parameters), precisionBits: literal(256), approximationContract: text } satisfies Record<string, Check>;
  if (input.kind === thermalKinds[0]) {
    const p = input.parameters;
    const stage: Check = x => shape(x, { label: choice('initial', 'first-quench', 'first-hold', 'second-quench', 'second-hold'), time: exact, ...fields('stateOneProbability meanEnergy meanWork heatIntoSystem bathEntropy systemEntropy totalEntropyProduction nonequilibriumFreeEnergy canonicalFreeEnergy relativeEntropyToCanonical independentFirstLawResidual meanEnergyJ meanWorkJ heatIntoSystemJ', q), firstLawResidual: exact });
    const hold: Check = x => shape(x, { stationaryOneProbability: exact, decay: q, transition: matrix(), generator: matrix(), ...fields('heatIntoSystem totalEntropyProduction', q), exactlyZeroProduction: bool });
    const work: Check = x => {
      if (!thermalRecord(x) || !integer(x.initialState, 0, 1) || !integer(x.secondQuenchState, 0, 1)) return false;
      const reachable = p.time1Numerator !== 0 || x.initialState === x.secondQuenchState;
      return shape(x, { initialState: y => integer(y, 0, 1), secondQuenchState: y => integer(y, 0, 1), reachable: literal(reachable), probability: q, work: reachable ? q : literal(null), inverseWorkRatio: exact, reverseCanonicalProbability: q, forwardToReverseMassRatio: reachable ? exact : literal(null), canonicalCrooksPrediction: reachable ? exact : literal(null), initialDensityCorrection: exact });
    };
    const endpoint: Check = x => {
      if (!thermalRecord(x) || !['x', 'y', 'z'].every(k => integer(x[k], 0, 1))) return false;
      const reachable = (p.time1Numerator !== 0 || x.x === x.y) && (p.time2Numerator !== 0 || x.y === x.z);
      return shape(x, { ...fields('x y z', y => integer(y, 0, 1)), reachable: literal(reachable), ...fields('probability inverseEntropyContribution', q), ...fields('work heatIntoSystem energyChange bathEntropy systemEntropyChange totalEntropy', reachable ? q : literal(null)), firstLawResidual: reachable ? exact : literal(null) });
    };
    if (!shape(r, { ...common, ...fields('temperatureK thermalEnergyJ boltzmannConstantJPerK', exact), gapInThermalUnits: array(3, q), equilibriumOneProbabilities: array(3, exact), stages: array(5, stage), holds: array(2, hold), workAtoms: array(4, work), endpointAtoms: array(8, endpoint), ...fields('meanWork heatIntoSystem meanWorkVariance totalEntropyProduction nonequilibriumDissipatedWork canonicalFreeEnergyChange canonicalDissipatedWork initialCanonicalRelativeEntropy finalCanonicalRelativeEntropy workExponentialMoment computedReverseNormalization', q), ...fields('initialDistributionIsCanonical jarzynskiApplicable', bool), jarzynskiIdentity: nullable(exact), ...fields('entropyIFTIdentity firstLawResidual freeEnergyBalanceResidual', exact), classification: x => shape(x, { ...fields('fullSupportInitialLaw specifiedConstantTemperatureBath workOnSystem heatIntoSystem markovianEvenStates endpointBinsIntegrateAllJumpCounts', literal(true)), ...fields('sampledPaths normalizedOrTruncatedPathSum registeredReader', literal(false)), ...fields('exactlyZeroWork zeroMeanWorkIdentityApplies', bool) }) })) return false;
    const stages = r.stages as Record<string, unknown>[], works = r.workAtoms as Record<string, unknown>[], atoms = r.endpointAtoms as Record<string, unknown>[];
    const canonical = BigInt(p.initialOneNumerator) * (BigInt(p.r0Numerator) + BigInt(p.r0Denominator)) === BigInt(p.r0Denominator) * BigInt(p.initialDenominator);
    return r.initialDistributionIsCanonical === canonical && stages.every((s, i) => s.label === ['initial', 'first-quench', 'first-hold', 'second-quench', 'second-hold'][i]) && works.every((w, i) => w.initialState === Math.floor(i / 2) && w.secondQuenchState === i % 2) && atoms.every((a, i) => a.x === Math.floor(i / 4) && a.y === Math.floor(i / 2) % 2 && a.z === i % 2) && r.jarzynskiApplicable === r.initialDistributionIsCanonical && (r.jarzynskiIdentity !== null) === r.jarzynskiApplicable;
  }
  const p = input.parameters, states = Array.from({ length: p.jumpCount + 1 }, (_, i) => (p.initialState + i) % 2), baths = Array.from({ length: p.jumpCount }, (_, i) => (p.bathCode >> i) & 1 ? 'B' : 'A');
  const history: Check = x => shape(x, { step: y => integer(y, 0, p.sampleSteps), time: exact, ...fields('meanHeatIntoSystemA meanHeatIntoSystemB meanHeatIntoSystemAJ meanHeatIntoSystemBJ meanBathEntropyA meanBathEntropyB meanTotalEntropyProduction stationaryShannonEntropy', q), ...fields('meanSystemEntropyChange meanEnergyChange meanWork aggregateObservedEntropy firstLawResidual', exact) });
  const channel: Check = x => shape(x, { name: choice('A', 'B'), beta: y => integer(y, 1, 8), ...fields('temperatureK clock equilibriumOneProbabilityAlone upRate downRate forwardTraffic reverseTraffic current reversedFilmUpRate reversedFilmDownRate', exact), ...fields('heatIntoSystemRate heatIntoSystemRateJPerSecond bathEntropyRate fullEntropyContributionRate', q) });
  const jump: Check = x => shape(x, { index: y => integer(y, 0, 15), ...fields('from to', y => integer(y, 0, 1)), bath: choice('A', 'B'), ...fields('rate reverseRate aggregateObservedEntropyIncrement', exact), ...fields('heatIntoSystem bathEntropy totalEntropyIncrement cumulativeHeatIntoSystemA cumulativeHeatIntoSystemB cumulativeTotalEntropy', q) });
  const detailed = p.oddsNumerator === p.oddsDenominator || p.betaA === p.betaB;
  if (!shape(r, { ...common, ...fields('referenceTemperatureK boltzmannConstantJPerK referenceThermalEnergyJ', exact), reservoirTemperaturesK: vec(exact), ...fields('gapInReferenceThermalUnits gapJ stationaryShannonEntropy stationaryEnergyInReferenceThermalUnits physicalEntropyRate pairedPhysicalEntropyRate conditionalJeffreysDivergence conditionalLabelEntropyRate', q), ...fields('aggregateStationaryCurrent aggregateObservedEntropyRate commonAggregateTraffic energyRateResidual stationaryShannonEntropyRate entropyIFTIdentity', exact), generator: matrix(exact), reversedFilmAggregateGenerator: matrix(exact), stationaryLaw: vec(exact), channels: array(2, channel), conditionalBathChoicesUp: vec(exact), conditionalBathChoicesDown: vec(exact), entropyIFTMethod: literal('analytic-normalization-of-stationary-marked-time-reversal'), history: array(p.sampleSteps + 1, history), current: history,
    path: x => shape(x, { states: y => same(y, states), reversedStates: y => same(y, [...states].reverse()), baths: y => same(y, baths), reversedBaths: y => same(y, [...baths].reverse()), jumps: array(p.jumpCount, jump), jumpTimes: array(p.jumpCount, exact), ...fields('equalDwellTime survivalAction work firstLawResidual totalRatio aggregateObservedEntropy conditionalBathWordProbability reversedConditionalBathWordProbability conditionalBathWordRatio', exact), ...fields('systemBoundaryEntropy bathEntropyA bathEntropyB bathEntropy heatIntoSystemA heatIntoSystemB heatIntoSystemAJ heatIntoSystemBJ energyChange totalEntropy reversedTotalEntropy logForwardDensityInSeconds logReversedDensityInSeconds independentLogDensityDifference', q), densityReference: text }),
    classification: x => shape(x, { ...fields('stationaryInitialization evenTimeReversalStates bothChannelsPositive aggregateDetailedBalance', literal(true)), channelDetailedBalance: literal(detailed), physicalEntropyRateExactSign: literal(detailed ? 0 : 1), zeroEnergyGap: literal(p.oddsNumerator === p.oddsDenominator), hotterChannel: literal(p.betaA === p.betaB ? 'equal-temperatures' : p.betaA < p.betaB ? 'A' : 'B'), heatFlowDirection: literal(detailed ? 'none' : p.betaA < p.betaB ? 'A-to-B' : 'B-to-A'), ...fields('sampledPaths registeredReader', literal(false)) }) })) return false;
  const h = r.history as Record<string, unknown>[], c = r.channels as Record<string, unknown>[], path = r.path as Record<string, unknown>;
  return h.every((x, i) => x.step === i) && same(r.current, h.at(-1)) && c.every((x, i) => x.name === ['A', 'B'][i] && x.beta === [p.betaA, p.betaB][i]) && (path.jumps as Record<string, unknown>[]).every((x, i) => x.index === i && x.from === states[i] && x.to === states[i + 1] && x.bath === baths[i]);
}
