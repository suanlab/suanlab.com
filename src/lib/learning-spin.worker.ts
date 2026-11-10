import { evaluateSpinRequest } from './learning-spin-evaluator';
import { isSpinInput, isSpinRequestEnvelope } from './learning-spin-protocol';
const scope = self as unknown as { onmessage: ((event: MessageEvent<unknown>) => void) | null; postMessage(value: unknown): void };
scope.onmessage = event => {
  const value = event.data;
  if (isSpinRequestEnvelope(value) && isSpinInput(value.input)) scope.postMessage({ version: 1, requestId: value.requestId, phase: 'started', kind: value.input.kind });
  scope.postMessage(evaluateSpinRequest(value));
};
