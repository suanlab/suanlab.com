/** Worker-only numerical dispatch. UI/controller modules import no models. */
import { finiteBoundaryExtensivityModel } from './learning-boundary-extensivity';
import { spatialDensitySelfAveragingModel } from './learning-spatial-self-averaging';
import { twoPhaseRoundingModel } from './learning-two-phase-rounding';
import { finiteSizeRecord, finiteSizeKinds, isFiniteSizeInput, isFiniteSizeRequestEnvelope, snapshotFiniteSizeInput, finiteSizeInputKey, type FiniteSizeResponse } from './learning-finite-size-protocol';
export function evaluateFiniteSizeRequest(value: unknown): FiniteSizeResponse {
 const requestId = finiteSizeRecord(value) && Number.isSafeInteger(value.requestId) && (value.requestId as number) > 0 ? value.requestId as number : null;
 if (!isFiniteSizeRequestEnvelope(value)) return { version: 1, requestId, ok: false, error: 'invalid-request' };
 if (!isFiniteSizeInput(value.input)) return { version: 1, requestId, ok: false, error: 'invalid-input' };
 try {
  const input = snapshotFiniteSizeInput(value.input), base = { version: 1 as const, requestId: value.requestId, ok: true as const, inputKey: finiteSizeInputKey(input) };
  if (input.kind === finiteSizeKinds[0]) return { ...base, kind: input.kind, input, result: finiteBoundaryExtensivityModel(input.parameters) };
  if (input.kind === finiteSizeKinds[1]) return { ...base, kind: input.kind, input, result: spatialDensitySelfAveragingModel(input.parameters) };
  return { ...base, kind: input.kind, input, result: twoPhaseRoundingModel(input.parameters) };
 } catch { return { version: 1, requestId, ok: false, error: 'calculation-failed' }; }
}
