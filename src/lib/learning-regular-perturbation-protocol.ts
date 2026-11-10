/** Reader-safe transport metadata. Mathematical checking runs in a Worker. */
import { readRegularPerturbationInput, type RegularPerturbationInput } from './learning-regular-perturbation-input';

export type RegularPerturbationResult = ReturnType<typeof import('./learning-regular-perturbation').calculateRegularPerturbation>;
export const regularPerturbationInputKey = (input: RegularPerturbationInput) => JSON.stringify(input);
export interface RegularPerturbationSuccess {
  version: 1; requestId: number; ok: true; kind: 'regular-perturbation';
  input: RegularPerturbationInput; inputKey: string; result: RegularPerturbationResult;
}
export interface RegularPerturbationFailure {
  version: 1; requestId: number | null; ok: false;
  error: 'invalid-request' | 'invalid-input' | 'resource-limit' | 'calculation-failed';
}
export interface RegularPerturbationStarted { version: 1; requestId: number; phase: 'started'; kind: 'regular-perturbation'; }
export type RegularPerturbationReceipt =
  | { version: 1; requestId: number; phase: 'validated'; kind: 'regular-perturbation'; inputKey: string; response: RegularPerturbationSuccess }
  | { version: 1; requestId: number | null; phase: 'rejected'; error: 'invalid-request' | 'invalid-input' | 'invalid-response' | 'resource-limit' };

export function regularPerturbationData(value: unknown, keys: readonly string[]): Record<string, unknown> | null {
  if (!value || typeof value !== 'object' || Object.getPrototypeOf(value) !== Object.prototype || Reflect.ownKeys(value).length !== keys.length) return null;
  const out: Record<string, unknown> = {};
  for (const key of keys) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (!descriptor || !Object.hasOwn(descriptor, 'value')) return null;
    out[key] = descriptor.value;
  }
  return out;
}

export function readRegularPerturbationRequest(value: unknown): { request: { requestId: number; input: RegularPerturbationInput } } | { failure: RegularPerturbationFailure } {
  const data = regularPerturbationData(value, ['version', 'requestId', 'input']);
  if (!data || data.version !== 1 || typeof data.requestId !== 'number' || !Number.isSafeInteger(data.requestId) || data.requestId <= 0) return { failure: { version: 1, requestId: null, ok: false, error: 'invalid-request' } };
  const input = readRegularPerturbationInput(data.input);
  return input ? { request: { requestId: data.requestId, input } } : { failure: { version: 1, requestId: data.requestId, ok: false, error: 'invalid-input' } };
}

export function regularPerturbationSuccessHeader(value: unknown, input: RegularPerturbationInput, id: number): value is RegularPerturbationSuccess {
  const data = regularPerturbationData(value, ['version', 'requestId', 'ok', 'kind', 'input', 'inputKey', 'result']);
  if (!data || data.version !== 1 || data.requestId !== id || data.ok !== true || data.kind !== 'regular-perturbation' || data.inputKey !== regularPerturbationInputKey(input)) return false;
  const attributed = readRegularPerturbationInput(data.input);
  if (!attributed || regularPerturbationInputKey(attributed) !== regularPerturbationInputKey(input)) return false;
  if (!data.result || typeof data.result !== 'object') return false;
  const own = Object.getOwnPropertyDescriptor(data.result, 'input');
  const resultInput = own && Object.hasOwn(own, 'value') ? readRegularPerturbationInput(own.value) : null;
  return !!resultInput && regularPerturbationInputKey(resultInput) === regularPerturbationInputKey(input);
}
