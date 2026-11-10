/** Reader-safe transport metadata. Mathematical checking runs in a Worker. */
import { readEuclideanKernelInput, type EuclideanKernelInput } from './learning-euclidean-kernel-input';

export type EuclideanKernelResult = ReturnType<typeof import('./learning-euclidean-kernel').calculateEuclideanKernel>;
export const euclideanKernelInputKey = (input: EuclideanKernelInput) => JSON.stringify(readEuclideanKernelInput(input));
export interface EuclideanKernelSuccess {
  version: 1; requestId: number; ok: true; kind: 'euclidean-kernel';
  input: EuclideanKernelInput; inputKey: string; result: EuclideanKernelResult;
}
export interface EuclideanKernelFailure {
  version: 1; requestId: number | null; ok: false;
  error: 'invalid-request' | 'invalid-input' | 'resource-limit' | 'calculation-failed';
}
export interface EuclideanKernelStarted { version: 1; requestId: number; phase: 'started'; kind: 'euclidean-kernel'; }
export type EuclideanKernelReceipt =
  | { version: 1; requestId: number; phase: 'validated'; kind: 'euclidean-kernel'; inputKey: string; response: EuclideanKernelSuccess }
  | { version: 1; requestId: number | null; phase: 'rejected'; error: 'invalid-request' | 'invalid-input' | 'invalid-response' | 'resource-limit' };

export function euclideanKernelData(value: unknown, keys: readonly string[]): Record<string, unknown> | null {
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

export function readEuclideanKernelRequest(value: unknown): { request: { requestId: number; input: EuclideanKernelInput } } | { failure: EuclideanKernelFailure } {
  const data = euclideanKernelData(value, ['version', 'requestId', 'input']);
  if (!data || data.version !== 1 || typeof data.requestId !== 'number' || !Number.isSafeInteger(data.requestId) || data.requestId <= 0) return { failure: { version: 1, requestId: null, ok: false, error: 'invalid-request' } };
  const input = readEuclideanKernelInput(data.input);
  return input ? { request: { requestId: data.requestId, input } } : { failure: { version: 1, requestId: data.requestId, ok: false, error: 'invalid-input' } };
}

export function euclideanKernelSuccessHeader(value: unknown, input: EuclideanKernelInput, id: number): value is EuclideanKernelSuccess {
  const data = euclideanKernelData(value, ['version', 'requestId', 'ok', 'kind', 'input', 'inputKey', 'result']);
  if (!data || data.version !== 1 || data.requestId !== id || data.ok !== true || data.kind !== 'euclidean-kernel' || data.inputKey !== euclideanKernelInputKey(input)) return false;
  const attributed = readEuclideanKernelInput(data.input);
  if (!attributed || euclideanKernelInputKey(attributed) !== euclideanKernelInputKey(input)) return false;
  if (!data.result || typeof data.result !== 'object') return false;
  const own = Object.getOwnPropertyDescriptor(data.result, 'input');
  const resultInput = own && Object.hasOwn(own, 'value') ? readEuclideanKernelInput(own.value) : null;
  return !!resultInput && euclideanKernelInputKey(resultInput) === euclideanKernelInputKey(input);
}
