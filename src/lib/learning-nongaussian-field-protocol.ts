/** Reader-safe transport metadata. Mathematical checking runs in a Worker. */
import { readNongaussianFieldInput, type NongaussianFieldInput } from './learning-nongaussian-field-input';

export type NongaussianFieldResult = ReturnType<typeof import('./learning-nongaussian-field').calculateNongaussianField>;
export const nongaussianFieldInputKey = (input: NongaussianFieldInput) => JSON.stringify(input);
export interface NongaussianFieldSuccess {
  version: 1; requestId: number; ok: true; kind: 'nongaussian-field';
  input: NongaussianFieldInput; inputKey: string; result: NongaussianFieldResult;
}
export interface NongaussianFieldFailure {
  version: 1; requestId: number | null; ok: false;
  error: 'invalid-request' | 'invalid-input' | 'resource-limit' | 'calculation-failed';
}
export interface NongaussianFieldStarted { version: 1; requestId: number; phase: 'started'; kind: 'nongaussian-field'; }
export type NongaussianFieldReceipt =
  | { version: 1; requestId: number; phase: 'validated'; kind: 'nongaussian-field'; inputKey: string; response: NongaussianFieldSuccess }
  | { version: 1; requestId: number | null; phase: 'rejected'; error: 'invalid-request' | 'invalid-input' | 'invalid-response' | 'resource-limit' };

export function nongaussianFieldData(value: unknown, keys: readonly string[]): Record<string, unknown> | null {
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

export function readNongaussianFieldRequest(value: unknown): { request: { requestId: number; input: NongaussianFieldInput } } | { failure: NongaussianFieldFailure } {
  const data = nongaussianFieldData(value, ['version', 'requestId', 'input']);
  if (!data || data.version !== 1 || typeof data.requestId !== 'number' || !Number.isSafeInteger(data.requestId) || data.requestId <= 0) return { failure: { version: 1, requestId: null, ok: false, error: 'invalid-request' } };
  const input = readNongaussianFieldInput(data.input);
  return input ? { request: { requestId: data.requestId, input } } : { failure: { version: 1, requestId: data.requestId, ok: false, error: 'invalid-input' } };
}

export function nongaussianFieldSuccessHeader(value: unknown, input: NongaussianFieldInput, id: number): value is NongaussianFieldSuccess {
  const data = nongaussianFieldData(value, ['version', 'requestId', 'ok', 'kind', 'input', 'inputKey', 'result']);
  if (!data || data.version !== 1 || data.requestId !== id || data.ok !== true || data.kind !== 'nongaussian-field' || data.inputKey !== nongaussianFieldInputKey(input)) return false;
  const attributed = readNongaussianFieldInput(data.input);
  if (!attributed || nongaussianFieldInputKey(attributed) !== nongaussianFieldInputKey(input)) return false;
  if (!data.result || typeof data.result !== 'object') return false;
  const own = Object.getOwnPropertyDescriptor(data.result, 'input');
  const resultInput = own && Object.hasOwn(own, 'value') ? readNongaussianFieldInput(own.value) : null;
  return !!resultInput && nongaussianFieldInputKey(resultInput) === nongaussianFieldInputKey(input);
}
