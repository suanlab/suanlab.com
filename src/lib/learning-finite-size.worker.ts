import { evaluateFiniteSizeRequest } from './learning-finite-size-evaluator';
import { isFiniteSizeInput, isFiniteSizeRequestEnvelope } from './learning-finite-size-protocol';
const scope = self as unknown as { onmessage: ((event: MessageEvent<unknown>) => void) | null; postMessage(value: unknown): void };
scope.onmessage = event => {
 const value = event.data;
 if (isFiniteSizeRequestEnvelope(value) && isFiniteSizeInput(value.input)) scope.postMessage({ version: 1, requestId: value.requestId, phase: 'started', kind: value.input.kind });
 scope.postMessage(evaluateFiniteSizeRequest(value));
};
