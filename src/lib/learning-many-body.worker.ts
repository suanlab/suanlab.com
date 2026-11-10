import { evaluateManyBodyRequest } from './learning-many-body-evaluator';
const scope = self as unknown as { onmessage: ((event: MessageEvent<unknown>) => void) | null; postMessage(value: unknown): void };
scope.onmessage = event => {
  scope.postMessage(evaluateManyBodyRequest(event.data, message => scope.postMessage(message)));
};
