import { evaluateSynchronizationRequest } from './learning-synchronization-evaluator';
import { isSynchronizationInput, isSynchronizationRequestEnvelope } from './learning-synchronization-protocol';
const scope = self as unknown as { onmessage: ((event: MessageEvent<unknown>) => void) | null; postMessage(value: unknown): void };
scope.onmessage = event => {
  const value = event.data;
  if (isSynchronizationRequestEnvelope(value) && isSynchronizationInput(value.input)) scope.postMessage({ version: 1, requestId: value.requestId, phase: 'started', kind: value.input.kind });
  scope.postMessage(evaluateSynchronizationRequest(value));
};
