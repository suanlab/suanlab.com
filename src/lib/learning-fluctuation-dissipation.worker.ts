import { evaluateFdtRequest } from './learning-fluctuation-dissipation-evaluator';
import { isFdtInput, isFdtRequestEnvelope } from './learning-fluctuation-dissipation-protocol';
const scope = self as unknown as { onmessage: ((event: MessageEvent<unknown>) => void) | null; postMessage(value: unknown): void };
scope.onmessage = event => {
 const value = event.data;
 if (isFdtRequestEnvelope(value) && isFdtInput(value.input)) scope.postMessage({ version: 1, requestId: value.requestId, phase: 'started', kind: value.input.kind });
 scope.postMessage(evaluateFdtRequest(value));
};
