/** Reader-safe transport metadata. Mathematical checking runs in a Worker. */
import { readSaddleFluctuationInput, type SaddleFluctuationInput } from './learning-saddle-fluctuation-input';

export type SaddleFluctuationResult = ReturnType<typeof import('./learning-saddle-fluctuation').calculateSaddleFluctuation>;
export const saddleFluctuationInputKey = (input: SaddleFluctuationInput) => JSON.stringify(readSaddleFluctuationInput(input));
export interface SaddleFluctuationSuccess {
  version: 1; requestId: number; ok: true; kind: 'saddle-fluctuation';
  input: SaddleFluctuationInput; inputKey: string; result: SaddleFluctuationResult;
}
export interface SaddleFluctuationFailure {
  version: 1; requestId: number | null; ok: false;
  error: 'invalid-request' | 'invalid-input' | 'resource-limit' | 'calculation-failed';
}
export interface SaddleFluctuationStarted { version: 1; requestId: number; phase: 'started'; kind: 'saddle-fluctuation'; }
export type SaddleFluctuationReceipt =
  | { version: 1; requestId: number; phase: 'validated'; kind: 'saddle-fluctuation'; inputKey: string; response: SaddleFluctuationSuccess }
  | { version: 1; requestId: number | null; phase: 'rejected'; error: 'invalid-request' | 'invalid-input' | 'invalid-response' | 'resource-limit' };

export function saddleFluctuationData(value: unknown, keys: readonly string[]): Record<string, unknown> | null {
  try {
  if (!value || typeof value !== 'object' || Object.getPrototypeOf(value) !== Object.prototype || Reflect.ownKeys(value).length !== keys.length) return null;
  const out: Record<string, unknown> = {};
  for (const key of keys) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (!descriptor || !Object.hasOwn(descriptor, 'value')) return null;
    Object.defineProperty(out, key, {value: descriptor.value, enumerable: true});
  }
  return out;
  } catch { return null; }
}

export function readSaddleFluctuationRequest(value: unknown): { request: { requestId: number; input: SaddleFluctuationInput } } | { failure: SaddleFluctuationFailure } {
  const data = saddleFluctuationData(value, ['version', 'requestId', 'input']);
  if (!data || data.version !== 1 || typeof data.requestId !== 'number' || !Number.isSafeInteger(data.requestId) || data.requestId <= 0) return { failure: { version: 1, requestId: null, ok: false, error: 'invalid-request' } };
  const input = readSaddleFluctuationInput(data.input);
  return input ? { request: { requestId: data.requestId, input } } : { failure: { version: 1, requestId: data.requestId, ok: false, error: 'invalid-input' } };
}

export function saddleFluctuationSuccessHeader(value: unknown, input: SaddleFluctuationInput, id: number): value is SaddleFluctuationSuccess {
  const data = saddleFluctuationData(value, ['version', 'requestId', 'ok', 'kind', 'input', 'inputKey', 'result']);
  if (!data || data.version !== 1 || data.requestId !== id || data.ok !== true || data.kind !== 'saddle-fluctuation' || data.inputKey !== saddleFluctuationInputKey(input)) return false;
  const attributed = readSaddleFluctuationInput(data.input);
  if (!attributed || saddleFluctuationInputKey(attributed) !== saddleFluctuationInputKey(input)) return false;
  if (!data.result || typeof data.result !== 'object') return false;
  const own = Object.getOwnPropertyDescriptor(data.result, 'input');
  const resultInput = own && Object.hasOwn(own, 'value') ? readSaddleFluctuationInput(own.value) : null;
  return !!resultInput && saddleFluctuationInputKey(resultInput) === saddleFluctuationInputKey(input);
}
