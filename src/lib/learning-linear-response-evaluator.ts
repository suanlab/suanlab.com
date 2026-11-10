import { finiteStateResponseExperiment } from './learning-finite-state-response-experiment';
import { lrRecord, isLrInput, isLrRequestEnvelope, snapshotLrInput, lrInputKey, type LrResponse } from './learning-linear-response-protocol';
export function evaluateLrRequest(value: unknown): LrResponse {
  const requestId = lrRecord(value) && Number.isSafeInteger(value.requestId) && (value.requestId as number) > 0 ? value.requestId as number : null;
  if (!isLrRequestEnvelope(value)) return { version: 1, requestId, ok: false, error: 'invalid-request' };
  if (!isLrInput(value.input)) return { version: 1, requestId, ok: false, error: 'invalid-input' };
  try { const input = snapshotLrInput(value.input); return { version: 1, requestId: value.requestId, ok: true, kind: input.kind, input, inputKey: lrInputKey(input), result: finiteStateResponseExperiment(input.parameters) }; }
  catch { return { version: 1, requestId, ok: false, error: 'calculation-failed' }; }
}
