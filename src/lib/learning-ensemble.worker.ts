import { evaluateEnsembleRequest } from './learning-ensemble-evaluator';
import { ensembleRecord, isEnsembleInput, type EnsembleStarted } from './learning-ensemble-protocol';
const scope = self as unknown as { onmessage: ((event: MessageEvent<unknown>) => void) | null; postMessage(value: unknown): void };
scope.onmessage = event => {
  const value: unknown = event.data;
  if (ensembleRecord(value) && value.version === 1 && Number.isSafeInteger(value.requestId) && (value.requestId as number) > 0 && isEnsembleInput(value.input)) {
    const started: EnsembleStarted = { version: 1, requestId: value.requestId as number, phase: 'started', kind: value.input.kind };
    scope.postMessage(started);
  }
  scope.postMessage(evaluateEnsembleRequest(value));
};
