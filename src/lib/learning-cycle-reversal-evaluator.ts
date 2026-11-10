import { continuousCycleReversalModel } from './learning-continuous-cycle-reversal';
import { reversalRecord, isReversalInput, isReversalRequestEnvelope, snapshotReversalInput, reversalInputKey, type ReversalResponse } from './learning-cycle-reversal-protocol';
export function evaluateReversalRequest(v: unknown): ReversalResponse {
  const requestId = reversalRecord(v) && Number.isSafeInteger(v.requestId) && (v.requestId as number) > 0 ? v.requestId as number : null;
  if (!isReversalRequestEnvelope(v)) return { version: 1, requestId, ok: false, error: 'invalid-request' };
  if (!isReversalInput(v.input)) return { version: 1, requestId, ok: false, error: 'invalid-input' };
  try {
    const input = snapshotReversalInput(v.input);
    return { version: 1, requestId: v.requestId, ok: true, kind: input.kind, input, inputKey: reversalInputKey(input), result: continuousCycleReversalModel(input.parameters) };
  } catch { return { version: 1, requestId, ok: false, error: 'calculation-failed' }; }
}
