import { evaluateBoltzmannGibbsRequest } from './learning-boltzmann-gibbs-evaluator';
import { boltzmannRecord, isBoltzmannGibbsInput, type BoltzmannGibbsStarted } from './learning-boltzmann-gibbs-protocol';
const scope = self as unknown as { onmessage: ((event: MessageEvent<unknown>) => void) | null; postMessage(value: unknown): void };
scope.onmessage = event => {
  const value: unknown = event.data;
  if (boltzmannRecord(value) && value.version === 1 && Number.isSafeInteger(value.requestId) && (value.requestId as number) > 0 && isBoltzmannGibbsInput(value.input)) {
    const started: BoltzmannGibbsStarted = { version: 1, requestId: value.requestId as number, phase: 'started', kind: value.input.kind };
    scope.postMessage(started);
  }
  scope.postMessage(evaluateBoltzmannGibbsRequest(value));
};
