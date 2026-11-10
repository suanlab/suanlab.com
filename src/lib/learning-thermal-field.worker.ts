import { evaluateThermalFieldRequest } from './learning-thermal-field-evaluator';
const scope = self as unknown as { onmessage: ((event: MessageEvent<unknown>) => void) | null; postMessage(value: unknown): void };
scope.onmessage = event => scope.postMessage(evaluateThermalFieldRequest(event.data, message => scope.postMessage(message)));
