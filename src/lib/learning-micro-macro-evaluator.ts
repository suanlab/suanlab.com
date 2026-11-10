/** Worker-side dispatch only. Do not import this module from UI components. */
import { binaryMacrostateModel } from './learning-binary-macrostates';
import { macroPartitionModel } from './learning-macro-partition';
import { noisyMacroObservationModel } from './learning-noisy-macro-observation';
import { isMicroMacroInput, microMacroRecord, type MicroMacroResponse } from './learning-micro-macro-protocol';
export function evaluateMicroMacroRequest(value: unknown): MicroMacroResponse {
  const requestId = microMacroRecord(value) && Number.isSafeInteger(value.requestId) && (value.requestId as number) > 0 ? value.requestId as number : null;
  if (!microMacroRecord(value) || value.version !== 1 || requestId === null) return { version: 1, requestId, ok: false, error: 'invalid-request' };
  if (!isMicroMacroInput(value.input)) return { version: 1, requestId, ok: false, error: 'invalid-input' };
  const input = value.input;
  try {
    if (input.kind === 'binary-shell-multiplicity-block-counts') {
      const p = input.parameters;
      return { version: 1, requestId, ok: true, kind: input.kind, result: binaryMacrostateModel(p.sites, p.excitations, p.blockASites, p.energyGap) };
    }
    if (input.kind === 'finite-macro-partition-conditional-entropy') {
      const p = input.parameters;
      return { version: 1, requestId, ok: true, kind: input.kind, result: macroPartitionModel(p.trialWeights, p.targetWeights, p.mapping, p.groupCount) };
    }
    const p = input.parameters;
    return { version: 1, requestId, ok: true, kind: input.kind, result: noisyMacroObservationModel(p.preparationWeights, p.channelWeights) };
  } catch {
    return { version: 1, requestId, ok: false, error: 'calculation-failed' };
  }
}
