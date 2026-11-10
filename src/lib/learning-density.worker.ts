import { evaluateDensityRequest } from './learning-density-evaluator';
import { isDensityInput, isDensityRequestEnvelope } from './learning-density-protocol';
const scope = self as unknown as { onmessage: ((event: MessageEvent<unknown>) => void) | null; postMessage(value: unknown): void };
scope.onmessage = event => {
  const value = event.data;
  if (isDensityRequestEnvelope(value) && isDensityInput(value.input)) scope.postMessage({ version: 1, requestId: value.requestId, phase: 'started', kind: value.input.kind });
  scope.postMessage(evaluateDensityRequest(value));
};
