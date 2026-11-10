import { evaluateCurrentRequest } from './learning-probability-current-evaluator';
import { isCurrentInput, isCurrentRequestEnvelope } from './learning-probability-current-protocol';
const scope = self as unknown as { onmessage: ((event: MessageEvent<unknown>) => void) | null; postMessage(value: unknown): void };
scope.onmessage = event => {
 const value = event.data;
 if (isCurrentRequestEnvelope(value) && isCurrentInput(value.input)) scope.postMessage({ version: 1, requestId: value.requestId, phase: 'started', kind: value.input.kind });
 scope.postMessage(evaluateCurrentRequest(value));
};
