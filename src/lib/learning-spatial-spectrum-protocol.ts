/** Reader-safe transport metadata. Mathematical checking runs in a Worker. */
import { readSpatialSpectrumInput, type SpatialSpectrumInput } from './learning-spatial-spectrum-input';

export type SpatialSpectrumResult = ReturnType<typeof import('./learning-spatial-spectrum').calculateSpatialSpectrum>;
export const spatialSpectrumInputKey = (input: SpatialSpectrumInput) => JSON.stringify(input);
export interface SpatialSpectrumSuccess {
  version: 1; requestId: number; ok: true; kind: 'spatial-spectrum';
  input: SpatialSpectrumInput; inputKey: string; result: SpatialSpectrumResult;
}
export interface SpatialSpectrumFailure {
  version: 1; requestId: number | null; ok: false;
  error: 'invalid-request' | 'invalid-input' | 'resource-limit' | 'calculation-failed';
}
export interface SpatialSpectrumStarted { version: 1; requestId: number; phase: 'started'; kind: 'spatial-spectrum'; }
export type SpatialSpectrumReceipt =
  | { version: 1; requestId: number; phase: 'validated'; kind: 'spatial-spectrum'; inputKey: string; response: SpatialSpectrumSuccess }
  | { version: 1; requestId: number | null; phase: 'rejected'; error: 'invalid-request' | 'invalid-input' | 'invalid-response' | 'resource-limit' };

export function spatialSpectrumData(value: unknown, keys: readonly string[]): Record<string, unknown> | null {
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

export function readSpatialSpectrumRequest(value: unknown): { request: { requestId: number; input: SpatialSpectrumInput } } | { failure: SpatialSpectrumFailure } {
  const data = spatialSpectrumData(value, ['version', 'requestId', 'input']);
  if (!data || data.version !== 1 || typeof data.requestId !== 'number' || !Number.isSafeInteger(data.requestId) || data.requestId <= 0) return { failure: { version: 1, requestId: null, ok: false, error: 'invalid-request' } };
  const input = readSpatialSpectrumInput(data.input);
  return input ? { request: { requestId: data.requestId, input } } : { failure: { version: 1, requestId: data.requestId, ok: false, error: 'invalid-input' } };
}

export function spatialSpectrumSuccessHeader(value: unknown, input: SpatialSpectrumInput, id: number): value is SpatialSpectrumSuccess {
  const data = spatialSpectrumData(value, ['version', 'requestId', 'ok', 'kind', 'input', 'inputKey', 'result']);
  if (!data || data.version !== 1 || data.requestId !== id || data.ok !== true || data.kind !== 'spatial-spectrum' || data.inputKey !== spatialSpectrumInputKey(input)) return false;
  const attributed = readSpatialSpectrumInput(data.input);
  if (!attributed || spatialSpectrumInputKey(attributed) !== spatialSpectrumInputKey(input)) return false;
  if (!data.result || typeof data.result !== 'object') return false;
  const own = Object.getOwnPropertyDescriptor(data.result, 'input');
  const resultInput = own && Object.hasOwn(own, 'value') ? readSpatialSpectrumInput(own.value) : null;
  return !!resultInput && spatialSpectrumInputKey(resultInput) === spatialSpectrumInputKey(input);
}
