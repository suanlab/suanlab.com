/** Shared transport and saved-input contract; calculator imports are type-only. */
import type { partitionCumulantModel, PartitionCumulantParameters } from './learning-partition-cumulants';
import type { coupledPartitionModel, CoupledPartitionParameters } from './learning-coupled-partition';
import type { energyLadderModel, EnergyLadderParameters } from './learning-energy-ladder';
export type PartitionInput =
  | { readonly kind: 'finite-spectrum-partition-cumulant-generators'; readonly parameters: Readonly<PartitionCumulantParameters> }
  | { readonly kind: 'coupled-binary-sites-partition-factorization'; readonly parameters: Readonly<CoupledPartitionParameters> }
  | { readonly kind: 'infinite-energy-ladder-normalization-cutoff'; readonly parameters: Readonly<EnergyLadderParameters> };
export type PartitionSuccess =
  | { version: 1; requestId: number; ok: true; kind: 'finite-spectrum-partition-cumulant-generators'; result: ReturnType<typeof partitionCumulantModel> }
  | { version: 1; requestId: number; ok: true; kind: 'coupled-binary-sites-partition-factorization'; result: ReturnType<typeof coupledPartitionModel> }
  | { version: 1; requestId: number; ok: true; kind: 'infinite-energy-ladder-normalization-cutoff'; result: ReturnType<typeof energyLadderModel> };
export interface PartitionRequest { version: 1; requestId: number; input: PartitionInput }
export interface PartitionStarted { version: 1; requestId: number; phase: 'started'; kind: PartitionInput['kind'] }
export type PartitionResponse = PartitionSuccess | { version: 1; requestId: number | null; ok: false; error: 'invalid-request' | 'invalid-input' | 'calculation-failed' };
export const partitionRecord = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);
const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
export function isPartitionInput(v: unknown): v is PartitionInput {
  if (!partitionRecord(v) || !partitionRecord(v.parameters)) return false;
  const p = v.parameters;
  if (!finite(p.temperature)) return false;
  if (v.kind === 'finite-spectrum-partition-cumulant-generators') return Array.isArray(p.energyLevels) && p.energyLevels.length >= 1 && p.energyLevels.length <= 32 && Array.from(p.energyLevels).every(finite) && finite(p.energyOffset);
  if (v.kind === 'coupled-binary-sites-partition-factorization') return finite(p.couplingEnergy) && finite(p.fieldEnergy) && ['full-product', 'fixed-zero-sum'].includes(p.support as string);
  return v.kind === 'infinite-energy-ladder-normalization-cutoff' && finite(p.gapEnergy) && p.gapEnergy >= 0 && finite(p.cutoff) && Number.isInteger(p.cutoff) && p.cutoff >= 0 && p.cutoff <= 4096;
}
export function snapshotPartitionInput(input: PartitionInput): PartitionInput {
  if (!isPartitionInput(input)) throw new RangeError('Invalid partition input');
  if (input.kind === 'finite-spectrum-partition-cumulant-generators') return Object.freeze({ kind: input.kind, parameters: Object.freeze({ energyLevels: Object.freeze([...input.parameters.energyLevels]), energyOffset: input.parameters.energyOffset, temperature: input.parameters.temperature }) });
  if (input.kind === 'coupled-binary-sites-partition-factorization') return Object.freeze({ kind: input.kind, parameters: Object.freeze({ couplingEnergy: input.parameters.couplingEnergy, fieldEnergy: input.parameters.fieldEnergy, temperature: input.parameters.temperature, support: input.parameters.support }) });
  return Object.freeze({ kind: input.kind, parameters: Object.freeze({ gapEnergy: input.parameters.gapEnergy, temperature: input.parameters.temperature, cutoff: input.parameters.cutoff }) });
}
export const partitionInputKey = (input: PartitionInput) => JSON.stringify(snapshotPartitionInput(input));
const resolutions = ['exact-rational', 'structural-zero', 'interval-resolved', 'derived-exponential', 'log-tail-approximation', 'unresolved-at-precision-ceiling', 'mathematical-infinity'];
function binary(v: unknown, positive = false): boolean {
  return partitionRecord(v) && typeof v.coefficient === 'string' && v.coefficient.length <= 100000
    && (positive ? /^[1-9]\d*$/ : /^-?(?:0|[1-9]\d*)$/).test(v.coefficient)
    && finite(v.exponent) && Number.isInteger(v.exponent) && Math.abs(v.exponent) <= 1000000;
}
function rational(v: unknown): boolean { return partitionRecord(v) && binary(v.numerator) && binary(v.denominator, true); }
export function isPartitionObservation(v: unknown): boolean {
  if (!partitionRecord(v) || !(v.value === null || finite(v.value)) || !(v.sign === null || [-1, 0, 1].includes(v.sign as number))
    || !(v.logMagnitude === null || finite(v.logMagnitude)) || !resolutions.includes(v.resolution as string)
    || !['finite', 'outside-number-range', 'not-applicable', 'unresolved'].includes(v.logResolution as string)) return false;
  if (v.resolution === 'structural-zero' && (v.value !== 0 || v.sign !== 0 || v.logMagnitude !== null)) return false;
  if (v.resolution === 'mathematical-infinity' && (v.value !== null || ![-1, 1].includes(v.sign as number))) return false;
  if (v.resolution === 'unresolved-at-precision-ceiling' && v.value !== null) return false;
  if (v.value === 0 && (v.sign !== 0 || v.resolution !== 'structural-zero')) return false;
  if (finite(v.value) && v.value !== 0 && Math.sign(v.value) !== v.sign) return false;
  return v.bounds === null || partitionRecord(v.bounds) && rational(v.bounds.lower) && rational(v.bounds.upper);
}
const observations = (v: Record<string, unknown>, keys: string[]) => keys.every(key => isPartitionObservation(v[key]));
const coreKeys = ['meanEnergy', 'energyVariance', 'thirdEnergyCumulant', 'heatCapacity', 'heatCapacityInKB', 'entropyInKB', 'entropy', 'centeredLogPartition', 'logPartition', 'partitionFunction', 'freeEnergy'];
const ladderKeys = ['meanIndex', 'indexVariance', 'meanEnergy', 'energyVariance', 'heatCapacityInKB', 'heatCapacity', 'entropyInKB', 'entropy', 'freeEnergy', 'logPartition', 'partitionFunction'];
const comparisonKeys = ['missingMass', 'retainedMass', 'logRetainedMass', 'totalVariation', 'forwardKL', 'reverseKL', 'meanIndexBias', 'indexVarianceBias', 'meanEnergyBias'];
const scopes = {
  'finite-spectrum-partition-cumulant-generators': 'finite-explicit-microstates-fixed-hamiltonian-with-common-offset',
  'coupled-binary-sites-partition-factorization': 'two-labeled-binary-sites-canonical-fixed-support',
  'infinite-energy-ladder-normalization-cutoff': 'specified-equally-spaced-spectrum-with-finite-conditional-cutoff',
};
export function partitionResultMatchesInput(input: PartitionInput, result: unknown): boolean {
  if (!partitionRecord(result) || result.scope !== scopes[input.kind] || typeof result.fullyResolved !== 'boolean') return false;
  const returned = { kind: input.kind, parameters: result.parameters };
  if (!isPartitionInput(returned) || partitionInputKey(returned) !== partitionInputKey(input)) return false;
  if (input.kind === 'finite-spectrum-partition-cumulant-generators') {
    return observations(result, [...coreKeys, 'energyAnchor', 'thermalEnergy', 'inverseTemperature'])
      && partitionRecord(result.logZBetaDerivatives) && observations(result.logZBetaDerivatives, ['first', 'second', 'third'])
      && Array.isArray(result.states) && result.states.length === input.parameters.energyLevels.length
      && Array.from(result.states).every((s, i) => partitionRecord(s) && s.index === i && s.relativeEnergy === input.parameters.energyLevels[i] && observations(s, ['energy', 'probability', 'logProbability']));
  }
  if (input.kind === 'coupled-binary-sites-partition-factorization') {
    if (!partitionRecord(result.joint) || !observations(result.joint, coreKeys) || !observations(result, ['meanAlignment', 'meanMagnetization', 'spinCovariance', 'mutualInformation', 'logFactorizationDefect', 'freeEnergyDefect']) || typeof result.factorizationIdentity !== 'boolean') return false;
    if (!Array.isArray(result.meanSpins) || result.meanSpins.length !== 2 || !Array.from(result.meanSpins).every(isPartitionObservation)
      || !Array.isArray(result.sourceHessian) || result.sourceHessian.length !== 2 || !Array.from(result.sourceHessian).every(row => Array.isArray(row) && row.length === 2 && Array.from(row).every(isPartitionObservation))) return false;
    if (!partitionRecord(result.responses) || !['alignmentToCoupling', 'magnetizationToField', 'alignmentToField'].every(key => {
      const response = (result.responses as Record<string, unknown>)[key];
      return partitionRecord(response) && (response.status === 'undefined-at-zero-temperature' ? response.observation === null : ['finite-temperature-derivative', 'locally-constant-zero'].includes(response.status as string) && isPartitionObservation(response.observation));
    })) return false;
    const support = input.parameters.support;
    return Array.isArray(result.states) && result.states.length === 4 && Array.from(result.states).every((s, i) => partitionRecord(s) && s.index === i
      && s.allowed === (support === 'full-product' || i === 1 || i === 2) && Array.isArray(s.spins) && s.spins.length === 2
      && s.spins[0] === (i < 2 ? -1 : 1) && s.spins[1] === (i % 2 ? 1 : -1)
      && (s.allowed ? isPartitionObservation(s.energy) : s.energy === null) && observations(s, ['probability', 'logProbability']));
  }
  if (!partitionRecord(result.finite) || !observations(result.finite, ladderKeys) || !partitionRecord(result.infinite)
    || !['normalizable', 'not-normalizable', 'positive-side-zero-temperature-limit'].includes(result.infinite.status as string)
    || !observations(result, ['q', 'anchoredDecayRatio']) || !(result.x === null || isPartitionObservation(result.x))
    || result.finiteStateCount !== input.parameters.cutoff + 1 || typeof result.sampledAllFiniteStates !== 'boolean') return false;
  if (result.infinite.quantities !== null && (!partitionRecord(result.infinite.quantities) || !observations(result.infinite.quantities, ladderKeys))) return false;
  if (result.infinite.status === 'not-normalizable' && (result.infinite.quantities !== null || result.comparison !== null)) return false;
  if (result.comparison !== null && (!partitionRecord(result.comparison) || !observations(result.comparison, comparisonKeys))) return false;
  return Array.isArray(result.samples) && result.samples.length >= 1 && result.samples.length <= 64
    && Array.from(result.samples).every((s, i, rows) => partitionRecord(s) && finite(s.n) && Number.isInteger(s.n) && s.n >= 0 && s.n <= input.parameters.cutoff + 1
      && (i === 0 || partitionRecord(rows[i - 1]) && s.n > (rows[i - 1] as Record<string, number>).n)
      && observations(s, ['energy', 'finiteProbability']) && (s.infiniteProbability === null || isPartitionObservation(s.infiniteProbability)));
}
