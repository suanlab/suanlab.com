import { evaluateLrRequest } from './learning-linear-response-evaluator';
import { isLrInput, isLrRequestEnvelope } from './learning-linear-response-protocol';
const scope = self as unknown as { onmessage: ((event: MessageEvent<unknown>) => void) | null; postMessage(value: unknown): void };
scope.onmessage = event => {
  const value = event.data;
  if (isLrRequestEnvelope(value) && isLrInput(value.input)) scope.postMessage({ version: 1, requestId: value.requestId, phase: 'started', kind: value.input.kind });
  scope.postMessage(evaluateLrRequest(value));
};
