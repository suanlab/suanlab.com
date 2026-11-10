import { evaluateMasterRequest } from './learning-master-equation-evaluator';
import { isMasterInput, isMasterRequestEnvelope } from './learning-master-equation-protocol';
const scope = self as unknown as { onmessage: ((event: MessageEvent<unknown>) => void) | null; postMessage(value: unknown): void };
scope.onmessage = event => {
 const value = event.data;
 if (isMasterRequestEnvelope(value) && isMasterInput(value.input)) scope.postMessage({ version: 1, requestId: value.requestId, phase: 'started', kind: value.input.kind });
 scope.postMessage(evaluateMasterRequest(value));
};
