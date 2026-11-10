import { evaluateCoarseRequest } from './learning-coarse-evaluator';
import { isCoarseInput, isCoarseRequestEnvelope } from './learning-coarse-protocol';
const scope = self as unknown as { onmessage: ((event: MessageEvent<unknown>) => void) | null; postMessage(value: unknown): void };
scope.onmessage = event => {
  const value = event.data;
  if (isCoarseRequestEnvelope(value) && isCoarseInput(value.input)) scope.postMessage({ version: 1, requestId: value.requestId, phase: 'started', kind: value.input.kind });
  scope.postMessage(evaluateCoarseRequest(value));
};
