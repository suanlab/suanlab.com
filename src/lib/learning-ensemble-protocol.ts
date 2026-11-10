/** Lightweight client/worker contract. Calculators are imported only as types. */
import type { finiteReservoirModel } from './learning-finite-reservoir';
import type { canonicalBinaryModel, CanonicalBinaryParameters } from './learning-canonical-binary';
import type { grandCanonicalOccupancyModel, GrandOccupancyParameters } from './learning-grand-canonical-occupancy';
export type EnsembleInput =
  | { readonly kind: 'finite-binary-reservoir-canonical-comparison'; readonly parameters: { readonly subsystemSites: number; readonly reservoirSites: number; readonly totalExcitations: number } }
  | { readonly kind: 'canonical-binary-energy-fluctuations'; readonly parameters: Readonly<CanonicalBinaryParameters> }
  | { readonly kind: 'grand-canonical-finite-occupancy-fluctuations'; readonly parameters: Readonly<GrandOccupancyParameters> };
export type EnsembleSuccess =
  | { version: 1; requestId: number; ok: true; kind: 'finite-binary-reservoir-canonical-comparison'; result: ReturnType<typeof finiteReservoirModel> }
  | { version: 1; requestId: number; ok: true; kind: 'canonical-binary-energy-fluctuations'; result: ReturnType<typeof canonicalBinaryModel> }
  | { version: 1; requestId: number; ok: true; kind: 'grand-canonical-finite-occupancy-fluctuations'; result: ReturnType<typeof grandCanonicalOccupancyModel> };
export interface EnsembleRequest { version: 1; requestId: number; input: EnsembleInput }
export interface EnsembleStarted { version: 1; requestId: number; phase: 'started'; kind: EnsembleInput['kind'] }
export type EnsembleResponse = EnsembleSuccess | { version: 1; requestId: number | null; ok: false; error: 'invalid-request' | 'invalid-input' | 'calculation-failed' };
export const ensembleRecord = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);
const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
const count = (v: unknown): v is number => finite(v) && Number.isInteger(v) && v >= 0 && v <= 4096;
export function isEnsembleInput(value: unknown): value is EnsembleInput {
  if (!ensembleRecord(value) || !ensembleRecord(value.parameters)) return false;
  const p = value.parameters;
  if (value.kind === 'finite-binary-reservoir-canonical-comparison') {
    if (!count(p.subsystemSites) || !count(p.reservoirSites) || !count(p.totalExcitations)) return false;
    const N = p.subsystemSites + p.reservoirSites;
    return N >= 1 && N <= 4096 && p.totalExcitations <= N;
  }
  if (value.kind !== 'canonical-binary-energy-fluctuations' && value.kind !== 'grand-canonical-finite-occupancy-fluctuations') return false;
  return count(p.sites) && finite(p.energyGap) && p.energyGap >= 0 && finite(p.temperature)
    && (value.kind !== 'grand-canonical-finite-occupancy-fluctuations' || finite(p.chemicalPotential));
}
/** Canonical scalar snapshots strip unrelated properties and caller mutation. */
export function snapshotEnsembleInput(input: EnsembleInput): EnsembleInput {
  if (!isEnsembleInput(input)) throw new RangeError('Invalid ensemble input');
  if (input.kind === 'finite-binary-reservoir-canonical-comparison') return Object.freeze({ kind: input.kind,
    parameters: Object.freeze({ subsystemSites: input.parameters.subsystemSites, reservoirSites: input.parameters.reservoirSites, totalExcitations: input.parameters.totalExcitations }) });
  if (input.kind === 'canonical-binary-energy-fluctuations') return Object.freeze({ kind: input.kind,
    parameters: Object.freeze({ sites: input.parameters.sites, energyGap: input.parameters.energyGap, temperature: input.parameters.temperature }) });
  return Object.freeze({ kind: input.kind, parameters: Object.freeze({ sites: input.parameters.sites, energyGap: input.parameters.energyGap,
    chemicalPotential: input.parameters.chemicalPotential, temperature: input.parameters.temperature }) });
}
export const ensembleInputKey = (input: EnsembleInput) => JSON.stringify(snapshotEnsembleInput(input));
const scopes: Record<EnsembleInput['kind'], string> = {
  'finite-binary-reservoir-canonical-comparison': 'finite-uniform-binary-shell-versus-matching-canonical',
  'canonical-binary-energy-fluctuations': 'canonical-distinguishable-independent-binary-sites',
  'grand-canonical-finite-occupancy-fluctuations': 'grand-canonical-classical-labeled-exclusion-sites',
};
function observation(v: unknown): boolean {
  if (!ensembleRecord(v) || ![-1, 0, 1].includes(v.sign as number) || !(v.value === null || finite(v.value)) || !(v.logMagnitude === null || finite(v.logMagnitude))) return false;
  return v.sign === 0 ? v.value === 0 && v.logMagnitude === null : v.value === null || (v.value as number) * (v.sign as number) > 0;
}
const nullableObservation = (v: unknown) => v === null || observation(v);
const spectrumObservation = (v: unknown) => observation(v) && ensembleRecord(v)
  && (v.logNegativeLogMagnitude === null || finite(v.logNegativeLogMagnitude))
  && ['finite', 'not-applicable', 'outside-number-range'].includes(v.logMagnitudeStatus as string)
  && ['finite', 'not-applicable', 'outside-number-range'].includes(v.logNegativeLogMagnitudeStatus as string);
const nullableSpectrum = (v: unknown) => v === null || spectrumObservation(v);
/** Identity and bounded response shape checks; this is not a numerical proof.
 * No calculator is loaded or evaluated by the receiving client. */
export function ensembleResultMatchesInput(input: EnsembleInput, result: unknown): boolean {
  if (!ensembleRecord(result) || result.scope !== scopes[input.kind] || !ensembleRecord(result.parameters) || !Array.isArray(result.distribution)) return false;
  const p = result.parameters;
  if (!Object.entries(input.parameters).every(([key, value]) => p[key] === value)) return false;
  if (input.kind === 'finite-binary-reservoir-canonical-comparison') {
    if (p.totalSites !== input.parameters.subsystemSites + input.parameters.reservoirSites || result.distribution.length !== input.parameters.subsystemSites + 1) return false;
    if (!['meanExcitations', 'varianceExcitations', 'canonicalVariance', 'totalVariation', 'entropyNats', 'microToCanonicalKlNats', 'canonicalToMicroKlNats'].every(key => observation(result[key]))) return false;
    if (typeof result.wholeShellCount !== 'string' || typeof result.canonicalCommonDenominator !== 'string') return false;
    return Array.from(result.distribution).every((row, i) => ensembleRecord(row) && row.excitations === i && typeof row.compatibleMicrostates === 'string'
      && typeof row.canonicalNumerator === 'string' && observation(row.microcanonical) && observation(row.canonical));
  }
  if (result.distribution.length !== input.parameters.sites + 1) return false;
  if (input.kind === 'canonical-binary-energy-fluctuations') {
    if (!['oneProbability', 'zeroProbability', 'meanExcitations', 'excitationVariance', 'meanEnergy', 'energyVariance', 'entropy', 'entropyInKB', 'heatCapacity', 'heatCapacityInKB', 'logPartition'].every(key => spectrumObservation(result[key]))) return false;
    if (!nullableSpectrum(result.relativeCountFluctuation) || !nullableSpectrum(result.relativeEnergyFluctuation) || !nullableObservation(result.scaledEnergyGap)) return false;
    return Array.from(result.distribution).every((row, i) => ensembleRecord(row) && row.excitations === i && typeof row.multiplicity === 'string' && spectrumObservation(row.probability));
  }
  if (!['oneProbability', 'zeroProbability', 'meanParticles', 'meanHoles', 'numberVariance', 'meanEnergy', 'energyVariance', 'energyNumberCovariance', 'entropy', 'entropyInKB', 'grandPotential'].every(key => spectrumObservation(result[key]))) return false;
  if (!['numberSusceptibility', 'relativeNumberFluctuation', 'logGrandPartition'].every(key => nullableSpectrum(result[key]))) return false;
  if (!observation(result.occupationCostMinusChemicalPotential) || !nullableObservation(result.scaledChemicalBias)) return false;
  if (!['finite-temperature', 'infinite-zero-temperature-limit', 'finite-zero-temperature-limit'].includes(result.logGrandPartitionStatus as string)) return false;
  if (!['finite-temperature', 'empty-system', 'undefined-at-zero-temperature-threshold', 'zero-temperature-away-from-threshold'].includes(result.susceptibilityStatus as string)) return false;
  return Array.from(result.distribution).every((row, i) => ensembleRecord(row) && row.particles === i && typeof row.multiplicity === 'string'
    && spectrumObservation(row.probability) && nullableSpectrum(row.conditionalPatternProbability));
}
