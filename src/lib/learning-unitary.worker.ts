import { evaluateUnitaryRequest } from './learning-unitary-evaluator';
import { isUnitaryInput, isUnitaryRequestEnvelope } from './learning-unitary-protocol';
const scope = self as unknown as { onmessage: ((event: MessageEvent<unknown>) => void) | null; postMessage(value: unknown): void };
scope.onmessage = event => {
  const value = event.data;
  if (isUnitaryRequestEnvelope(value) && isUnitaryInput(value.input)) scope.postMessage({ version: 1, requestId: value.requestId, phase: 'started', kind: value.input.kind });
  scope.postMessage(evaluateUnitaryRequest(value));
};
