import { evaluateThermalRequest } from './learning-thermal-evaluator';
import { isThermalInput, isThermalRequestEnvelope } from './learning-thermal-protocol';
const scope = self as unknown as { onmessage: ((event: MessageEvent<unknown>) => void) | null; postMessage(value: unknown): void };
scope.onmessage = event => {
  const value = event.data;
  if (isThermalRequestEnvelope(value) && isThermalInput(value.input)) scope.postMessage({ version: 1, requestId: value.requestId, phase: 'started', kind: value.input.kind });
  scope.postMessage(evaluateThermalRequest(value));
};
