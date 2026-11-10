import { evaluateFokkerRequest } from './learning-fokker-planck-evaluator';
import { isFokkerInput, isFokkerRequestEnvelope } from './learning-fokker-planck-protocol';
const scope = self as unknown as { onmessage: ((event: MessageEvent<unknown>) => void) | null; postMessage(value: unknown): void };
scope.onmessage = event => {
 const value = event.data;
 if (isFokkerRequestEnvelope(value) && isFokkerInput(value.input)) scope.postMessage({ version: 1, requestId: value.requestId, phase: 'started', kind: value.input.kind });
 scope.postMessage(evaluateFokkerRequest(value));
};
