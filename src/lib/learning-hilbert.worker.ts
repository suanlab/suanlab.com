import { evaluateHilbertRequest } from './learning-hilbert-evaluator';
import { isHilbertInput, isHilbertRequestEnvelope } from './learning-hilbert-protocol';
const scope = self as unknown as { onmessage: ((event: MessageEvent<unknown>) => void) | null; postMessage(value: unknown): void };
scope.onmessage = event => {
  const value = event.data;
  if (isHilbertRequestEnvelope(value) && isHilbertInput(value.input)) scope.postMessage({ version: 1, requestId: value.requestId, phase: 'started', kind: value.input.kind });
  scope.postMessage(evaluateHilbertRequest(value));
};
