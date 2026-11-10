import { parseNumberDraft } from './learning-number-input';
/** Lightweight transport/saved-input checks. Numerical imports are erased types.
 * This validates transport shape and input identity, not the mathematical
 * correctness of an arbitrary sender's claimed numerical solution.
 */
import { boltzmannGibbsKinds as kinds, isBoltzmannGibbsKind, type BoltzmannGibbsInput as ModelInput, type BoltzmannGibbsKind } from './learning-boltzmann-gibbs';
import type { weightedBoltzmannModel } from './learning-weighted-boltzmann';
import type { finiteGibbsModel } from './learning-finite-gibbs';
import type { quadraticGibbsModel } from './learning-quadratic-gibbs';
import type { FiniteGibbsPathOptions, FiniteGibbsPathResult } from './learning-finite-gibbs-path';
export type { BoltzmannGibbsKind };
export type BoltzmannGibbsInput = Exclude<ModelInput, { kind: 'finite-gibbs-conditionals-scan-support' }>
  | (Extract<ModelInput, { kind: 'finite-gibbs-conditionals-scan-support' }> & { path?: FiniteGibbsPathOptions });
type ResultMap = {
  'reference-measure-boltzmann-softmax-identifiability': ReturnType<typeof weightedBoltzmannModel>;
  'finite-gibbs-conditionals-scan-support': ReturnType<typeof finiteGibbsModel>;
  'quadratic-gaussian-gibbs-normalizability': ReturnType<typeof quadraticGibbsModel>;
};
export type BoltzmannGibbsSuccess = { [K in BoltzmannGibbsKind]: {
  version: 1; requestId: number; ok: true; kind: K; input: Extract<BoltzmannGibbsInput, { kind: K }>; inputKey: string; result: ResultMap[K];
} & (K extends 'finite-gibbs-conditionals-scan-support' ? { path: FiniteGibbsPathResult | null } : unknown) }[BoltzmannGibbsKind];
export interface BoltzmannGibbsRequest { version: 1; requestId: number; input: BoltzmannGibbsInput }
export interface BoltzmannGibbsStarted { version: 1; requestId: number; phase: 'started'; kind: BoltzmannGibbsKind }
export type BoltzmannGibbsResponse = BoltzmannGibbsSuccess | { version: 1; requestId: number | null; ok: false; error: 'invalid-request' | 'invalid-input' | 'calculation-failed' };
export const boltzmannRecord = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);
const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const array = (v: unknown, size: number, check: (x: unknown) => boolean): v is unknown[] => Array.isArray(v) && v.length === size && Array.from(v).every(check);
const pair = (v: unknown, check = finite) => array(v, 2, check);
const numberFields = (v: Record<string, unknown>, keys: string[]) => keys.every(k => finite(v[k]));
export function isBoltzmannGibbsInput(v: unknown): v is BoltzmannGibbsInput {
  if (!boltzmannRecord(v) || typeof v.kind !== 'string' || !isBoltzmannGibbsKind(v.kind) || !boltzmannRecord(v.parameters)) return false;
  const p = v.parameters;
  if (!finite(p.temperature) || !finite(p.energyOffset) || v.kind !== kinds[1] && v.path !== undefined) return false;
  if (v.kind === kinds[0]) {
    return Array.isArray(p.energyLevels) && p.energyLevels.length >= 1 && p.energyLevels.length <= 32 && Array.from(p.energyLevels).every(finite)
      && array(p.referenceWeights, p.energyLevels.length, x => finite(x) && x >= 0) && (p.referenceWeights as number[]).some(x => x > 0);
  }
  if (v.kind === kinds[1]) {
    const count = p.support === 'full-product' ? 4 : 2;
    return (v.path === undefined || isFiniteGibbsPathOptions(v.path)) && p.temperature !== 0 && numberFields(p, ['couplingEnergy', 'firstFieldEnergy', 'secondFieldEnergy']) && ['full-product', 'fixed-zero-sum'].includes(p.support as string)
      && (p.initialStateIndex === undefined || finite(p.initialStateIndex) && Number.isInteger(p.initialStateIndex) && p.initialStateIndex >= 0 && p.initialStateIndex < count)
      && (p.observationSteps === undefined || Array.isArray(p.observationSteps) && p.observationSteps.length >= 1 && p.observationSteps.length <= 8
        && Array.from(p.observationSteps).every((s, i, a) => finite(s) && Number.isInteger(s) && s >= 1 && s <= 64 && (i === 0 || s > a[i - 1])));
  }
  return p.temperature > 0 && numberFields(p, ['firstStiffness', 'crossStiffness', 'secondStiffness', 'firstSourceEnergy', 'secondSourceEnergy'])
    && (p.coordinateScale === undefined || pair(p.coordinateScale, (x: unknown): x is number => finite(x) && x !== 0))
    && (p.coordinateShift === undefined || pair(p.coordinateShift));
}
export function snapshotBoltzmannGibbsInput(value: BoltzmannGibbsInput): BoltzmannGibbsInput {
  if (!isBoltzmannGibbsInput(value)) throw new RangeError('Invalid Boltzmann/Gibbs input');
  if (value.kind === kinds[0]) {
    const p = value.parameters;
    return Object.freeze({ kind: value.kind, parameters: Object.freeze({ energyLevels: Object.freeze([...p.energyLevels]), referenceWeights: Object.freeze([...p.referenceWeights]), energyOffset: p.energyOffset, temperature: p.temperature }) });
  }
  if (value.kind === kinds[1]) {
    const p = value.parameters;
    return Object.freeze({ kind: value.kind, parameters: Object.freeze({ couplingEnergy: p.couplingEnergy, firstFieldEnergy: p.firstFieldEnergy, secondFieldEnergy: p.secondFieldEnergy,
      energyOffset: p.energyOffset, temperature: p.temperature, support: p.support, initialStateIndex: p.initialStateIndex ?? 0, observationSteps: Object.freeze([...(p.observationSteps ?? [1, 2, 4])]) }),
      ...(value.path === undefined ? {} : { path: Object.freeze({ seed: value.path.seed, bits: value.path.bits, steps: value.path.steps, mode: value.path.mode }) }) });
  }
  const p = value.parameters;
  return Object.freeze({ kind: value.kind, parameters: Object.freeze({ firstStiffness: p.firstStiffness, crossStiffness: p.crossStiffness, secondStiffness: p.secondStiffness,
    firstSourceEnergy: p.firstSourceEnergy, secondSourceEnergy: p.secondSourceEnergy, energyOffset: p.energyOffset, temperature: p.temperature,
    coordinateScale: Object.freeze([...(p.coordinateScale ?? [1, 1])]) as readonly [number, number], coordinateShift: Object.freeze([...(p.coordinateShift ?? [0, 0])]) as readonly [number, number] }) });
}
export const boltzmannGibbsInputKey = (v: BoltzmannGibbsInput) => JSON.stringify(snapshotBoltzmannGibbsInput(v));
export const validBoltzmannGibbsSavedInput = (kind: BoltzmannGibbsKind, v: unknown): v is BoltzmannGibbsInput => isBoltzmannGibbsInput(v) && v.kind === kind;
export type ParsedBoltzmannNumber = { ok: true; value: number } | { ok: false; error: 'invalid-number' | 'outside-input-range' };
export function parseBoltzmannNumber(text: string, integer = false): ParsedBoltzmannNumber {
  const s = text.trim();
  if (s.length > 256 || !/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(s)) return { ok: false, error: 'invalid-number' };
  const value = Number(s);
  if (!Number.isFinite(value) || value === 0 && /[1-9]/.test(s.split(/[eE]/)[0]) || integer && parseNumberDraft(s, -Number.MAX_VALUE, Number.MAX_VALUE, true) === null) return { ok: false, error: 'outside-input-range' };
  return { ok: true, value: value === 0 ? 0 : value };
}
export function parseBoltzmannNumberList(text: string, maximum = 32, integer = false): { ok: true; values: number[] } | { ok: false; error: 'invalid-number' | 'outside-input-range' | 'invalid-list' } {
  if (!Number.isInteger(maximum) || maximum < 1 || maximum > 32 || text.length > 8192) return { ok: false, error: 'invalid-list' };
  const tokens = text.includes(',') ? text.split(',').map(s => s.trim()) : text.trim().split(/\s+/);
  if (tokens.length < 1 || tokens.length > maximum || tokens.some(s => s === '')) return { ok: false, error: 'invalid-list' };
  const values: number[] = [];
  for (const token of tokens) { const p = parseBoltzmannNumber(token, integer); if (!p.ok) return p; values.push(p.value); }
  return { ok: true, values };
}
// A 64-sweep distance can multiply 127 dyadic factors of 8192-bit precision:
// <=1,040,384 coefficient bits (~313,187 decimal digits), before normalization.
export const boltzmannTransportLimits = { coefficientCharacters: 350000, absoluteExponent: 1100000 } as const;
function binary(v: unknown, positive = false): boolean {
  return boltzmannRecord(v) && typeof v.coefficient === 'string' && v.coefficient.length <= boltzmannTransportLimits.coefficientCharacters
    && v.coefficient !== '-0' && (positive ? /^[1-9]\d*$/ : /^-?(?:0|[1-9]\d*)$/).test(v.coefficient)
    && finite(v.exponent) && Number.isInteger(v.exponent) && Math.abs(v.exponent) <= boltzmannTransportLimits.absoluteExponent;
}
const rational = (v: unknown) => boltzmannRecord(v) && binary(v.numerator) && binary(v.denominator, true);
export function isBoltzmannGibbsObservation(v: unknown): boolean {
  if (!boltzmannRecord(v) || !(v.value === null || finite(v.value)) || !(v.sign === null || [-1, 0, 1].includes(v.sign as number))
    || !(v.logMagnitude === null || finite(v.logMagnitude)) || !['exact-rational', 'structural-zero', 'interval-resolved', 'unresolved-at-precision-ceiling', 'mathematical-infinity'].includes(v.resolution as string)
    || !['finite', 'outside-number-range', 'not-applicable', 'unresolved'].includes(v.logResolution as string)) return false;
  if (v.logResolution === 'finite' ? !finite(v.logMagnitude) : v.logMagnitude !== null) return false;
  if (v.resolution === 'structural-zero') {
    if (v.value !== 0 || v.sign !== 0 || v.logMagnitude !== null || v.logResolution !== 'not-applicable') return false;
  } else if (v.sign === 0 || v.value === 0) return false;
  if (v.resolution === 'mathematical-infinity') return v.value === null && [-1, 1].includes(v.sign as number) && v.logResolution === 'not-applicable' && v.bounds === null;
  if (v.resolution === 'unresolved-at-precision-ceiling' && v.value !== null) return false;
  if (['interval-resolved', 'exact-rational'].includes(v.resolution as string) && ![-1, 1].includes(v.sign as number)) return false;
  if (finite(v.value) && v.value !== 0 && Math.sign(v.value) !== v.sign) return false;
  const hasBounds = boltzmannRecord(v.bounds) && rational(v.bounds.lower) && rational(v.bounds.upper);
  if (!hasBounds) return v.bounds === null && v.resolution === 'unresolved-at-precision-ceiling';
  if (v.resolution === 'structural-zero') {
    const b = v.bounds as Record<string, Record<string, { coefficient: string }>>;
    return b.lower.numerator.coefficient === '0' && b.upper.numerator.coefficient === '0';
  }
  return true;
}
const obs = isBoltzmannGibbsObservation;
const fields = (v: Record<string, unknown>, keys: string[]) => keys.every(k => obs(v[k]));
const vector = (v: unknown, n: number) => array(v, n, obs);
const matrix = (v: unknown, n: number) => array(v, n, row => vector(row, n));
const sign = (v: unknown, expected: number) => boltzmannRecord(v) && v.sign === expected;
const mass = (v: unknown, positive: boolean) => obs(v) && sign(v, positive ? 1 : 0) && boltzmannRecord(v) && (v.value === null || finite(v.value) && v.value >= 0 && v.value <= 1);
const same = (a: unknown, b: unknown) => { try { return JSON.stringify(a) === JSON.stringify(b); } catch { return false; } };
const precisionValues = [128, 256, 512, 1024, 2048, 4096, 8192];
function precision(v: Record<string, unknown>) {
  const i = precisionValues.indexOf(v.precisionBits as number);
  return i >= 0 && same(v.attempts, precisionValues.slice(0, i + 1));
}
const modes = ['first', 'second', 'random', 'systematic', 'blocked'] as const;
function resolvedFlag(value: Record<string, unknown>, parts: unknown[]): boolean {
  const resolved = (v: unknown): boolean => Array.isArray(v) ? Array.from(v).every(resolved)
    : boltzmannRecord(v) && typeof v.resolution === 'string' && v.resolution !== 'unresolved-at-precision-ceiling';
  return value.fullyResolved === parts.every(resolved);
}
export function boltzmannGibbsResultMatchesInput(input: BoltzmannGibbsInput, value: unknown): boolean {
  if (!isBoltzmannGibbsInput(input) || !boltzmannRecord(value) || typeof value.fullyResolved !== 'boolean' || value.boltzmannConstant !== 1.380649e-23) return false;
  if (input.kind === kinds[0]) {
    const p = input.parameters, allowed = p.referenceWeights.filter(m => m > 0).length;
    const mode = p.temperature === 0 ? 'positive-side-zero-temperature-limit' : p.temperature < 0 ? 'bounded-negative-temperature' : 'positive-temperature';
    if (value.scope !== 'finite-reference-measure-boltzmann' || value.mode !== mode || !precision(value)
      || value.allowedStateCount !== allowed || !finite(value.anchorStateCount) || !Number.isInteger(value.anchorStateCount) || value.anchorStateCount < 1 || value.anchorStateCount > allowed
      || !fields(value, ['referenceMeasureTotal', 'meanEnergy', 'energyVariance', 'labelEntropyNats', 'meanLogReferenceWeight', 'relativeEntropyInKB', 'heatCapacityInKB', 'freeEnergy', 'logPartition'])) return false;
    return array(value.states, p.energyLevels.length, s => boltzmannRecord(s) && fields(s, ['energy', 'referenceWeight', 'probability', 'logProbability']))
      && (value.states as Record<string, unknown>[]).every((s, i) => s.index === i && s.allowed === (p.referenceWeights[i] > 0)
        && boltzmannRecord(s.referenceWeight) && s.referenceWeight.value === p.referenceWeights[i]
        && boltzmannRecord(s.probability) && mass(s.probability, s.probability.sign === 1)
        && (s.probability.sign === 0 ? p.temperature === 0 || p.referenceWeights[i] === 0 : p.referenceWeights[i] > 0)
        && boltzmannRecord(s.logProbability) && (s.probability.sign === 0 ? s.logProbability.resolution === 'mathematical-infinity' && s.logProbability.sign === -1 : s.logProbability.sign === -1 || s.logProbability.sign === 0))
      && resolvedFlag(value, [...['referenceMeasureTotal', 'meanEnergy', 'energyVariance', 'labelEntropyNats', 'meanLogReferenceWeight', 'relativeEntropyInKB', 'heatCapacityInKB', 'freeEnergy', 'logPartition'].map(k => value[k]),
        ...(value.states as Record<string, unknown>[]).flatMap(s => [s.probability, s.logProbability])]);
  }
  const returned = { kind: input.kind, parameters: value.parameters };
  if (!isBoltzmannGibbsInput(returned) || boltzmannGibbsInputKey(returned) !== boltzmannGibbsInputKey({ ...input, path: undefined } as BoltzmannGibbsInput)) return false;
  if (input.kind === kinds[1]) {
    const p = snapshotBoltzmannGibbsInput(input).parameters as typeof input.parameters;
    const states = [[-1, -1], [-1, 1], [1, -1], [1, 1]].filter(s => p.support === 'full-product' || s[0] + s[1] === 0), n = states.length, steps = p.observationSteps!;
    if (value.scope !== 'finite-two-spin-conditional-gibbs-kernels' || !precision(value) || value.initialStateIndex !== p.initialStateIndex || !same(value.observationSteps, steps)
      || !array(value.states, n, s => boltzmannRecord(s) && obs(s.energy)) || !(value.states as Record<string, unknown>[]).every((s, i) => s.index === i && same(s.spins, states[i]))
      || !vector(value.target, n) || !(value.target as unknown[]).every(q => mass(q, true)) || !vector(value.logTarget, n) || !(value.logTarget as unknown[]).every(q => sign(q, -1))
      || !obs(value.systematicContraction) || !boltzmannRecord(value.systematicDetailedBalanceWitness)) return false;
    const records = ['kernels', 'flows', 'distributions', 'totalVariationToTarget', 'diagnostics', 'numericalChecks'];
    if (!records.every(k => boltzmannRecord(value[k]))) return false;
    const first = states.map(s => states.map(t => s[1] === t[1])), second = states.map(s => states.map(t => s[0] === t[0]));
    const graphs = { first, second, random: first.map((row, i) => row.map((v, j) => v || second[i][j])),
      systematic: first.map(row => states.map((_, j) => row.some((v, k) => v && second[k][j]))), blocked: states.map(() => states.map(() => true)) };
    for (const name of modes) {
      const K = (value.kernels as Record<string, unknown>)[name], F = (value.flows as Record<string, unknown>)[name], D = (value.distributions as Record<string, unknown>)[name];
      const distances = (value.totalVariationToTarget as Record<string, unknown>)[name], diagnostic = (value.diagnostics as Record<string, unknown>)[name], checks = (value.numericalChecks as Record<string, unknown>)[name];
      if (!matrix(K, n) || !matrix(F, n) || !array(D, steps.length, row => vector(row, n)) || !vector(distances, steps.length) || !boltzmannRecord(diagnostic) || !boltzmannRecord(checks)) return false;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (!mass((K as unknown[][])[i][j], graphs[name][i][j]) || !mass((F as unknown[][])[i][j], graphs[name][i][j])) return false;
      let reachable = states.map((_, i) => i === p.initialStateIndex);
      for (let t = 1; t <= steps[steps.length - 1]; t++) {
        reachable = states.map((_, j) => reachable.some((v, i) => v && graphs[name][i][j]));
        const index = steps.indexOf(t);
        if (index >= 0 && !(D as unknown[][])[index].every((q, j) => mass(q, reachable[j]))) return false;
      }
      const irreducible = name === 'blocked' || p.support === 'full-product' && (name === 'random' || name === 'systematic');
      const reversible = name !== 'systematic' || p.couplingEnergy === 0 || p.support === 'fixed-zero-sum';
      const classes = irreducible ? [states.map((_, i) => i)] : p.support === 'fixed-zero-sum' ? [[0], [1]] : name === 'first' ? [[0, 2], [1, 3]] : [[0, 1], [2, 3]];
      if (diagnostic.irreducible !== irreducible || diagnostic.detailedBalance !== reversible || !same(diagnostic.stronglyConnectedComponents, classes)
        || diagnostic.uniqueStationaryDistribution !== irreducible || diagnostic.convergesToTargetFromEveryInitialState !== irreducible
        || diagnostic.rowStochastic !== true || diagnostic.targetStationary !== true || diagnostic.positiveSelfLoops !== true || diagnostic.aperiodicWithinEachClass !== true
        || diagnostic.coordinatesResampledPerStep !== (name === 'systematic' || name === 'blocked' ? 2 : 1)
        || checks.rowSumsContainOne !== true || checks.stationarityResidualsContainZero !== true || !vector(checks.rowSums, n) || !vector(checks.stationarityResiduals, n)) return false;
    }
    const expected = p.support === 'fixed-zero-sum' || p.couplingEnergy === 0 ? 0 : Math.sign(p.couplingEnergy) * Math.sign(p.temperature), w = value.systematicDetailedBalanceWitness;
    const flow = (value.flows as Record<string, unknown[][]>).systematic;
    return w.sourceIndex === 0 && w.destinationIndex === 1 && w.provenSign === expected && obs(w.difference) && sign(w.difference, expected)
      && same(w.forwardFlow, flow[0][1]) && same(w.reverseFlow, flow[1][0])
      && resolvedFlag(value, [value.target, value.logTarget, value.systematicContraction, w.difference, ...modes.flatMap(name => ['kernels', 'flows', 'distributions', 'totalVariationToTarget'].map(k => (value[k] as Record<string, unknown>)[name]))]);
  }
  if (value.scope !== 'continuous-quadratic-gibbs-lebesgue-r2' || typeof value.normalizable !== 'boolean' || !obs(value.determinant)
    || ![-1, 0, 1].includes(value.exactDeterminantSign as number) || !sign(value.determinant, value.exactDeterminantSign as number)) return false;
  if (value.normalizable !== (input.parameters.firstStiffness > 0 && value.exactDeterminantSign === 1)) return false;
  if (!value.normalizable) return value.classification === 'not-positive-definite' && value.precisionBits === 0 && same(value.attempts, []) && value.fullyResolved === true
    && ['flat-or-linear-unconfined-direction', 'negative-curvature-direction'].includes(value.reason as string)
    && obs(value.logPartition) && boltzmannRecord(value.logPartition) && value.logPartition.resolution === 'mathematical-infinity' && value.logPartition.sign === 1
    && !['mean', 'covariance', 'precisionMatrix', 'densityAtMean', 'transformedCoordinates'].some(k => k in value);
  if (value.classification !== 'positive-definite' || !precision(value) || !fields(value, ['thermalEnergy', 'minimumEnergy', 'meanEnergy', 'heatCapacityInKB', 'logPartition', 'freeEnergy', 'differentialEntropyNats', 'densityAtMean', 'correlation'])
    || !vector(value.mean, 2) || !matrix(value.covariance, 2) || !matrix(value.precisionMatrix, 2) || !boltzmannRecord(value.transformedCoordinates)) return false;
  const z = value.transformedCoordinates, c = value.covariance as unknown[][], inverse = value.precisionMatrix as unknown[][];
  const p = input.parameters, crossSign = -Math.sign(p.crossStiffness), scales = p.coordinateScale ?? [1, 1];
  if (!fields(z, ['logAbsoluteJacobian', 'logPartition', 'differentialEntropyNats', 'densityAtMean']) || !vector(z.mean, 2) || !matrix(z.covariance, 2)) return false;
  const cz = z.covariance as unknown[][];
  return sign(value.thermalEnergy, 1) && sign(value.densityAtMean, 1) && sign(z.densityAtMean, 1)
    && boltzmannRecord(value.heatCapacityInKB) && value.heatCapacityInKB.value === 1
    && sign(value.correlation, crossSign) && sign(c[0][0], 1) && sign(c[1][1], 1) && sign(c[0][1], crossSign) && same(c[0][1], c[1][0])
    && sign(inverse[0][0], 1) && sign(inverse[1][1], 1) && sign(inverse[0][1], Math.sign(p.crossStiffness)) && same(inverse[0][1], inverse[1][0])
    && sign(cz[0][0], 1) && sign(cz[1][1], 1) && sign(cz[0][1], crossSign * Math.sign(scales[0]) * Math.sign(scales[1])) && same(cz[0][1], cz[1][0])
    && resolvedFlag(value, [...['logPartition', 'freeEnergy', 'differentialEntropyNats', 'densityAtMean', 'correlation'].map(k => value[k]), ...['logAbsoluteJacobian', 'logPartition', 'differentialEntropyNats', 'densityAtMean'].map(k => z[k])]);
}
export function isBoltzmannGibbsSuccessForInput(v: unknown, input: BoltzmannGibbsInput, requestId: number): v is BoltzmannGibbsSuccess {
  return boltzmannRecord(v) && v.version === 1 && v.ok === true && v.requestId === requestId && v.kind === input.kind && isBoltzmannGibbsInput(v.input)
    && boltzmannGibbsInputKey(v.input) === boltzmannGibbsInputKey(input) && v.inputKey === boltzmannGibbsInputKey(input) && boltzmannGibbsResultMatchesInput(input, v.result)
    && (input.kind !== kinds[1] || input.path === undefined ? input.kind !== kinds[1] || v.path === null : isFiniteGibbsPathForInput(v.path, input));
}

export function isFiniteGibbsPathOptions(v: unknown): v is FiniteGibbsPathOptions {
  return boltzmannRecord(v) && finite(v.seed) && Number.isInteger(v.seed) && v.seed >= 0 && v.seed <= 0xffffffff
    && finite(v.bits) && Number.isInteger(v.bits) && v.bits >= 1 && v.bits <= 32
    && finite(v.steps) && Number.isInteger(v.steps) && v.steps >= 1 && v.steps <= 2048
    && modes.includes(v.mode as typeof modes[number]);
}
function isFiniteGibbsPathForInput(value: unknown, input: Extract<BoltzmannGibbsInput, { kind: 'finite-gibbs-conditionals-scan-support' }>): boolean {
  if (!input.path || !boltzmannRecord(value) || !isFiniteGibbsPathOptions(value.options) || !same(value.options, { seed: input.path.seed, bits: input.path.bits, steps: input.path.steps, mode: input.path.mode })) return false;
  if (value.status === 'unavailable') return value.reason === 'uncertain-grid-boundary';
  if (value.status !== 'ready' || value.prng !== 'LCG32-high-bits' || typeof value.contract !== 'string' || value.contract.length > 2048) return false;
  const p = input.path, n = input.parameters.support === 'full-product' ? 4 : 2, initial = input.parameters.initialStateIndex ?? 0;
  if (value.initialStateIndex !== initial || !array(value.states, p.steps + 1, x => finite(x) && Number.isInteger(x) && x >= 0 && x < n)
    || value.states[0] !== initial || !array(value.observedCounts, n, x => finite(x) && Number.isInteger(x) && x >= 0 && x <= p.steps + 1)
    || value.coordinatesResampled !== p.steps * (p.mode === 'systematic' || p.mode === 'blocked' ? 2 : 1)
    || value.randomDrawCount !== p.steps * (p.mode === 'systematic' || p.mode === 'random' ? 2 : 1)
    || !finite(value.finalPrngState) || !Number.isInteger(value.finalPrngState) || value.finalPrngState < 0 || value.finalPrngState > 0xffffffff
    || typeof value.quantizedIrreducible !== 'boolean' || !boltzmannRecord(value.quantizedKernel) || !boltzmannRecord(value.oneStepTVBound)) return false;
  const counts = Array.from({ length: n }, () => 0); (value.states as number[]).forEach(i => counts[i]++);
  if (!same(counts, value.observedCounts)) return false;
  const power = p.mode === 'systematic' ? 2 * p.bits : p.mode === 'random' ? p.bits + 1 : p.bits;
  const numerator = p.mode === 'systematic' ? 2 : p.mode === 'blocked' ? n - 1 : 1;
  if (value.quantizedKernel.denominatorPower !== power || value.oneStepTVBound.numerator !== numerator || value.oneStepTVBound.denominatorPower !== p.bits
    || !array(value.quantizedKernel.numerators, n, row => array(row, n, x => typeof x === 'string' && x.length <= 21 && /^(?:0|[1-9]\d*)$/.test(x)))) return false;
  // Bounded integer counters only (<=65 bits), never a scientific recalculation.
  const M = BigInt(1) << BigInt(power), rows = value.quantizedKernel.numerators as string[][];
  if (!rows.every(row => row.reduce((sum, v) => sum + BigInt(v), BigInt(0)) === M)) return false;
  const reach = rows.map(row => row.map(v => v !== '0'));
  for (let k = 0; k < n; k++) for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) reach[i][j] ||= reach[i][k] && reach[k][j];
  if (value.quantizedIrreducible !== reach.every(row => row.every(Boolean))) return false;
  const states = value.states as number[];
  for (let i = 1; i < states.length; i++) if (BigInt(rows[states[i - 1]][states[i]]) === BigInt(0)) return false;
  return true;
}
