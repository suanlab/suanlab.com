/** Reader-safe transport metadata. Mathematical checking runs in a Worker. */
import { readDuffingSecularInput, type DuffingSecularInput } from './learning-duffing-secular-input';

export type DuffingSecularResult = ReturnType<typeof import('./learning-duffing-secular').calculateDuffingSecular>;
export const duffingSecularInputKey = (input: DuffingSecularInput) => JSON.stringify(input);
export interface DuffingSecularSuccess {
  version: 1; requestId: number; ok: true; kind: 'duffing-secular';
  input: DuffingSecularInput; inputKey: string; result: DuffingSecularResult;
}
export interface DuffingSecularFailure {
  version: 1; requestId: number | null; ok: false;
  error: 'invalid-request' | 'invalid-input' | 'resource-limit' | 'calculation-failed';
}
export interface DuffingSecularStarted { version: 1; requestId: number; phase: 'started'; kind: 'duffing-secular'; }
export type DuffingSecularReceipt =
  | { version: 1; requestId: number; phase: 'validated'; kind: 'duffing-secular'; inputKey: string; response: DuffingSecularSuccess }
  | { version: 1; requestId: number | null; phase: 'rejected'; error: 'invalid-request' | 'invalid-input' | 'invalid-response' | 'resource-limit' };

export function duffingSecularData(value: unknown, keys: readonly string[]): Record<string, unknown> | null {
  if (!value || typeof value !== 'object' || Object.getPrototypeOf(value) !== Object.prototype || Reflect.ownKeys(value).length !== keys.length) return null;
  const out: Record<string, unknown> = {};
  for (const key of keys) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (!descriptor || !Object.hasOwn(descriptor, 'value')) return null;
    out[key] = descriptor.value;
  }
  return out;
}

export function readDuffingSecularRequest(value: unknown): { request: { requestId: number; input: DuffingSecularInput } } | { failure: DuffingSecularFailure } {
  const data = duffingSecularData(value, ['version', 'requestId', 'input']);
  if (!data || data.version !== 1 || typeof data.requestId !== 'number' || !Number.isSafeInteger(data.requestId) || data.requestId <= 0) return { failure: { version: 1, requestId: null, ok: false, error: 'invalid-request' } };
  const input = readDuffingSecularInput(data.input);
  return input ? { request: { requestId: data.requestId, input } } : { failure: { version: 1, requestId: data.requestId, ok: false, error: 'invalid-input' } };
}

export function duffingSecularSuccessHeader(value: unknown, input: DuffingSecularInput, id: number): value is DuffingSecularSuccess {
  const data = duffingSecularData(value, ['version', 'requestId', 'ok', 'kind', 'input', 'inputKey', 'result']);
  if (!data || data.version !== 1 || data.requestId !== id || data.ok !== true || data.kind !== 'duffing-secular' || data.inputKey !== duffingSecularInputKey(input)) return false;
  const attributed = readDuffingSecularInput(data.input);
  if (!attributed || duffingSecularInputKey(attributed) !== duffingSecularInputKey(input)) return false;
  if (!data.result || typeof data.result !== 'object') return false;
  const own = Object.getOwnPropertyDescriptor(data.result, 'input');
  const resultInput = own && Object.hasOwn(own, 'value') ? readDuffingSecularInput(own.value) : null;
  return !!resultInput && duffingSecularInputKey(resultInput) === duffingSecularInputKey(input);
}
