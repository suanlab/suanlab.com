import { evaluateObservableRequest } from './learning-observable-evaluator';
import { isObservableInput, isObservableRequestEnvelope } from './learning-observable-protocol';
const scope = self as unknown as { onmessage: ((event: MessageEvent<unknown>) => void) | null; postMessage(value: unknown): void };
scope.onmessage = event => {
  const value = event.data;
  if (isObservableRequestEnvelope(value) && isObservableInput(value.input)) scope.postMessage({ version: 1, requestId: value.requestId, phase: 'started', kind: value.input.kind });
  scope.postMessage(evaluateObservableRequest(value));
};
