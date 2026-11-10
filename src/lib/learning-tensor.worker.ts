import { evaluateTensorRequest } from './learning-tensor-evaluator';
import { isTensorInput, isTensorRequestEnvelope } from './learning-tensor-protocol';
const scope = self as unknown as { onmessage: ((event: MessageEvent<unknown>) => void) | null; postMessage(value: unknown): void };
scope.onmessage = event => {
  const value = event.data;
  if (isTensorRequestEnvelope(value) && isTensorInput(value.input)) scope.postMessage({ version: 1, requestId: value.requestId, phase: 'started', kind: value.input.kind });
  scope.postMessage(evaluateTensorRequest(value));
};
