/** Reader-safe transport metadata. Mathematical checking runs in a Worker. */
import { readGaussianFieldInput, type GaussianFieldInput } from './learning-gaussian-field-input';

export type GaussianFieldResult = ReturnType<typeof import('./learning-gaussian-field').calculateGaussianField>;
export const gaussianFieldInputKey = (input: GaussianFieldInput) => JSON.stringify(input);
export interface GaussianFieldSuccess {
  version: 1; requestId: number; ok: true; kind: 'gaussian-field';
  input: GaussianFieldInput; inputKey: string; result: GaussianFieldResult;
}
export interface GaussianFieldFailure {
  version: 1; requestId: number | null; ok: false;
  error: 'invalid-request' | 'invalid-input' | 'resource-limit' | 'calculation-failed';
}
export interface GaussianFieldStarted { version: 1; requestId: number; phase: 'started'; kind: 'gaussian-field'; }
export type GaussianFieldReceipt =
  | { version: 1; requestId: number; phase: 'validated'; kind: 'gaussian-field'; inputKey: string; response: GaussianFieldSuccess }
  | { version: 1; requestId: number | null; phase: 'rejected'; error: 'invalid-request' | 'invalid-input' | 'invalid-response' | 'resource-limit' };

export function gaussianFieldData(value: unknown, keys: readonly string[]): Record<string, unknown> | null {
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

export function readGaussianFieldRequest(value: unknown): { request: { requestId: number; input: GaussianFieldInput } } | { failure: GaussianFieldFailure } {
  const data = gaussianFieldData(value, ['version', 'requestId', 'input']);
  if (!data || data.version !== 1 || typeof data.requestId !== 'number' || !Number.isSafeInteger(data.requestId) || data.requestId <= 0) return { failure: { version: 1, requestId: null, ok: false, error: 'invalid-request' } };
  const input = readGaussianFieldInput(data.input);
  return input ? { request: { requestId: data.requestId, input } } : { failure: { version: 1, requestId: data.requestId, ok: false, error: 'invalid-input' } };
}

export function gaussianFieldSuccessHeader(value: unknown, input: GaussianFieldInput, id: number): value is GaussianFieldSuccess {
  const data = gaussianFieldData(value, ['version', 'requestId', 'ok', 'kind', 'input', 'inputKey', 'result']);
  if (!data || data.version !== 1 || data.requestId !== id || data.ok !== true || data.kind !== 'gaussian-field' || data.inputKey !== gaussianFieldInputKey(input)) return false;
  const attributed = readGaussianFieldInput(data.input);
  if (!attributed || gaussianFieldInputKey(attributed) !== gaussianFieldInputKey(input)) return false;
  if (!data.result || typeof data.result !== 'object') return false;
  const own = Object.getOwnPropertyDescriptor(data.result, 'input');
  const resultInput = own && Object.hasOwn(own, 'value') ? readGaussianFieldInput(own.value) : null;
  return !!resultInput && gaussianFieldInputKey(resultInput) === gaussianFieldInputKey(input);
}
