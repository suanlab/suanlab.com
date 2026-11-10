import { evaluateBlockOrderRequest } from './learning-block-order-evaluator';
const scope = self as unknown as { onmessage: ((event: MessageEvent<unknown>) => void) | null; postMessage(value: unknown): void };
scope.onmessage = event => scope.postMessage(evaluateBlockOrderRequest(event.data, message => scope.postMessage(message)));
