/** Lightweight shared protocol. Model imports are type-only: the UI must not
 * accidentally bundle or execute expensive calculators on the main thread.
 */
import type { binaryMacrostateModel } from './learning-binary-macrostates';
import type { macroPartitionModel } from './learning-macro-partition';
import type { noisyMacroObservationModel } from './learning-noisy-macro-observation';
export type MicroMacroInput =
  | { readonly kind: 'binary-shell-multiplicity-block-counts'; readonly parameters: { readonly sites: number; readonly excitations: number; readonly blockASites: number; readonly energyGap: number } }
  | { readonly kind: 'finite-macro-partition-conditional-entropy'; readonly parameters: { readonly trialWeights: readonly number[]; readonly targetWeights: readonly number[]; readonly mapping: readonly number[]; readonly groupCount: number } }
  | { readonly kind: 'finite-noisy-macro-observation-channel'; readonly parameters: { readonly preparationWeights: readonly number[]; readonly channelWeights: readonly (readonly number[])[] } };
export type MicroMacroSuccess =
  | { version: 1; requestId: number; ok: true; kind: 'binary-shell-multiplicity-block-counts'; result: ReturnType<typeof binaryMacrostateModel> }
  | { version: 1; requestId: number; ok: true; kind: 'finite-macro-partition-conditional-entropy'; result: ReturnType<typeof macroPartitionModel> }
  | { version: 1; requestId: number; ok: true; kind: 'finite-noisy-macro-observation-channel'; result: ReturnType<typeof noisyMacroObservationModel> };
export interface MicroMacroRequest { version: 1; requestId: number; input: MicroMacroInput }
export interface MicroMacroStarted { version: 1; requestId: number; phase: 'started'; kind: MicroMacroInput['kind'] }
export type MicroMacroResponse = MicroMacroSuccess | { version: 1; requestId: number | null; ok: false; error: 'invalid-request' | 'invalid-input' | 'calculation-failed' };
export const microMacroRecord = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);
const integer = (v: unknown, max: number) => typeof v === 'number' && Number.isInteger(v) && v >= 0 && v <= max;
const weightVector = (v: unknown, max: number): v is number[] => Array.isArray(v) && v.length > 0 && v.length <= max
  && Array.from(v).every(x => typeof x === 'number' && Number.isFinite(x) && x >= 0) && v.some(x => x > 0);
export function isMicroMacroInput(v: unknown): v is MicroMacroInput {
  if (!microMacroRecord(v) || !microMacroRecord(v.parameters)) return false;
  const p = v.parameters;
  switch (v.kind) {
    case 'binary-shell-multiplicity-block-counts':
      return integer(p.sites, 4096) && integer(p.excitations, p.sites as number) && integer(p.blockASites, p.sites as number)
        && typeof p.energyGap === 'number' && Number.isFinite(p.energyGap) && p.energyGap > 0;
    case 'finite-macro-partition-conditional-entropy':
      return weightVector(p.trialWeights, 32) && weightVector(p.targetWeights, 32) && p.trialWeights.length === p.targetWeights.length
        && integer(p.groupCount, 32) && (p.groupCount as number) > 0 && Array.isArray(p.mapping) && p.mapping.length === p.trialWeights.length
        && Array.from(p.mapping).every(x => integer(x, (p.groupCount as number) - 1));
    case 'finite-noisy-macro-observation-channel': {
      if (!weightVector(p.preparationWeights, 16) || !Array.isArray(p.channelWeights) || p.channelWeights.length !== p.preparationWeights.length) return false;
      const rows = Array.from(p.channelWeights);
      return rows.every(row => weightVector(row, 16)) && rows.every(row => (row as number[]).length === (rows[0] as number[]).length);
    }
    default: return false;
  }
}
/** Immutable parameter snapshots keep pending/result labels tied to the actual
 * submitted preparation, even when a caller later mutates its source arrays.
 */
export function snapshotMicroMacroInput(input: MicroMacroInput): MicroMacroInput {
  if (input.kind === 'binary-shell-multiplicity-block-counts')
    return Object.freeze({ kind: input.kind, parameters: Object.freeze({ sites: input.parameters.sites, excitations: input.parameters.excitations,
      blockASites: input.parameters.blockASites, energyGap: input.parameters.energyGap }) });
  if (input.kind === 'finite-macro-partition-conditional-entropy') return Object.freeze({ kind: input.kind, parameters: Object.freeze({
    trialWeights: Object.freeze([...input.parameters.trialWeights]), targetWeights: Object.freeze([...input.parameters.targetWeights]),
    mapping: Object.freeze([...input.parameters.mapping]), groupCount: input.parameters.groupCount }) });
  return Object.freeze({ kind: input.kind, parameters: Object.freeze({ preparationWeights: Object.freeze([...input.parameters.preparationWeights]),
    channelWeights: Object.freeze(input.parameters.channelWeights.map(row => Object.freeze([...row]))) }) });
}
export const microMacroInputKey = (input: MicroMacroInput) => JSON.stringify(snapshotMicroMacroInput(input));
