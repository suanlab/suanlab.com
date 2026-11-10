import { evaluateQuantumVariationalRequest } from './learning-quantum-variational-evaluator';
import { isQuantumVariationalInput, isQuantumVariationalRequestEnvelope } from './learning-quantum-variational-protocol';
const scope = self as unknown as { onmessage: ((event: MessageEvent<unknown>) => void) | null; postMessage(value: unknown): void };
scope.onmessage = event => {
  const value = event.data;
  if (isQuantumVariationalRequestEnvelope(value) && isQuantumVariationalInput(value.input)) scope.postMessage({ version: 1, requestId: value.requestId, phase: 'started', kind: value.input.kind });
  scope.postMessage(evaluateQuantumVariationalRequest(value));
};
