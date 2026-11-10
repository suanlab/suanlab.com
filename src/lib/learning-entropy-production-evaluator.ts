import { threeStateEntropyProductionModel } from './learning-three-state-entropy-production';
import { entropyRecord, isEntropyInput, isEntropyRequestEnvelope, snapshotEntropyInput, entropyInputKey, type EntropyResponse } from './learning-entropy-production-protocol';
export function evaluateEntropyRequest(value: unknown): EntropyResponse {
  const requestId = entropyRecord(value) && Number.isSafeInteger(value.requestId) && (value.requestId as number) > 0 ? value.requestId as number : null;
  if (!isEntropyRequestEnvelope(value)) return { version: 1, requestId, ok: false, error: 'invalid-request' };
  if (!isEntropyInput(value.input)) return { version: 1, requestId, ok: false, error: 'invalid-input' };
  try {
    const input = snapshotEntropyInput(value.input);
    return { version: 1, requestId: value.requestId, ok: true, kind: input.kind, input, inputKey: entropyInputKey(input), result: threeStateEntropyProductionModel(input.parameters) };
  } catch { return { version: 1, requestId, ok: false, error: 'calculation-failed' }; }
}
