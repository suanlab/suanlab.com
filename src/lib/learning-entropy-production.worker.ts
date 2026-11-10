import { evaluateEntropyRequest } from './learning-entropy-production-evaluator';
import { isEntropyInput, isEntropyRequestEnvelope } from './learning-entropy-production-protocol';
const scope = self as unknown as { onmessage: ((event: MessageEvent<unknown>) => void) | null; postMessage(value: unknown): void };
scope.onmessage = event => {
  const value = event.data;
  if (isEntropyRequestEnvelope(value) && isEntropyInput(value.input)) scope.postMessage({ version: 1, requestId: value.requestId, phase: 'started', kind: value.input.kind });
  scope.postMessage(evaluateEntropyRequest(value));
};
