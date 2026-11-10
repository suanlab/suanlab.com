import { evaluateMicroMacroRequest } from './learning-micro-macro-evaluator';
import { isMicroMacroInput, microMacroRecord, type MicroMacroStarted } from './learning-micro-macro-protocol';
// Use a narrow worker surface rather than adding conflicting DOM/worker libs
// to the entire static-site TypeScript project.
const scope = self as unknown as { onmessage: ((event: MessageEvent<unknown>) => void) | null; postMessage(value: unknown): void };
scope.onmessage = event => {
  const value: unknown = event.data;
  if (microMacroRecord(value) && value.version === 1 && Number.isSafeInteger(value.requestId) && (value.requestId as number) > 0 && isMicroMacroInput(value.input)) {
    const started: MicroMacroStarted = { version: 1, requestId: value.requestId as number, phase: 'started', kind: value.input.kind };
    scope.postMessage(started);
  }
  scope.postMessage(evaluateMicroMacroRequest(value));
};
