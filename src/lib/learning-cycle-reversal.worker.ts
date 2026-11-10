import { evaluateReversalRequest } from './learning-cycle-reversal-evaluator';
import { isReversalInput, isReversalRequestEnvelope } from './learning-cycle-reversal-protocol';
const scope = self as unknown as { onmessage: ((event: MessageEvent<unknown>) => void) | null; postMessage(value: unknown): void };
scope.onmessage = event => {
 const value = event.data;
 if (isReversalRequestEnvelope(value) && isReversalInput(value.input)) scope.postMessage({ version: 1, requestId: value.requestId, phase: 'started', kind: value.input.kind });
 scope.postMessage(evaluateReversalRequest(value));
};
