import { evaluateUncertaintyRequest } from './learning-uncertainty-evaluator';
import { isUncertaintyInput, isUncertaintyRequestEnvelope } from './learning-uncertainty-protocol';
const scope = self as unknown as { onmessage: ((event: MessageEvent<unknown>) => void) | null; postMessage(value: unknown): void };
scope.onmessage = event => {
  const value = event.data;
  if (isUncertaintyRequestEnvelope(value) && isUncertaintyInput(value.input)) scope.postMessage({ version: 1, requestId: value.requestId, phase: 'started', kind: value.input.kind });
  scope.postMessage(evaluateUncertaintyRequest(value));
};
