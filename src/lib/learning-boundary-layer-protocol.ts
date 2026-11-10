/** Reader-safe transport metadata. Mathematical checking runs in a Worker. */
import { readBoundaryLayerInput, type BoundaryLayerInput } from './learning-boundary-layer-input';

export type BoundaryLayerResult = ReturnType<typeof import('./learning-boundary-layer').calculateBoundaryLayer>;
export const boundaryLayerInputKey = (input: BoundaryLayerInput) => JSON.stringify(input);
export interface BoundaryLayerSuccess {
  version: 1; requestId: number; ok: true; kind: 'boundary-layer';
  input: BoundaryLayerInput; inputKey: string; result: BoundaryLayerResult;
}
export interface BoundaryLayerFailure {
  version: 1; requestId: number | null; ok: false;
  error: 'invalid-request' | 'invalid-input' | 'resource-limit' | 'calculation-failed';
}
export interface BoundaryLayerStarted { version: 1; requestId: number; phase: 'started'; kind: 'boundary-layer'; }
export type BoundaryLayerReceipt =
  | { version: 1; requestId: number; phase: 'validated'; kind: 'boundary-layer'; inputKey: string; response: BoundaryLayerSuccess }
  | { version: 1; requestId: number | null; phase: 'rejected'; error: 'invalid-request' | 'invalid-input' | 'invalid-response' | 'resource-limit' };

export function boundaryLayerData(value: unknown, keys: readonly string[]): Record<string, unknown> | null {
  if (!value || typeof value !== 'object' || Object.getPrototypeOf(value) !== Object.prototype || Reflect.ownKeys(value).length !== keys.length) return null;
  const out: Record<string, unknown> = {};
  for (const key of keys) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (!descriptor || !Object.hasOwn(descriptor, 'value')) return null;
    out[key] = descriptor.value;
  }
  return out;
}

export function readBoundaryLayerRequest(value: unknown): { request: { requestId: number; input: BoundaryLayerInput } } | { failure: BoundaryLayerFailure } {
  const data = boundaryLayerData(value, ['version', 'requestId', 'input']);
  if (!data || data.version !== 1 || typeof data.requestId !== 'number' || !Number.isSafeInteger(data.requestId) || data.requestId <= 0) return { failure: { version: 1, requestId: null, ok: false, error: 'invalid-request' } };
  const input = readBoundaryLayerInput(data.input);
  return input ? { request: { requestId: data.requestId, input } } : { failure: { version: 1, requestId: data.requestId, ok: false, error: 'invalid-input' } };
}

export function boundaryLayerSuccessHeader(value: unknown, input: BoundaryLayerInput, id: number): value is BoundaryLayerSuccess {
  const data = boundaryLayerData(value, ['version', 'requestId', 'ok', 'kind', 'input', 'inputKey', 'result']);
  if (!data || data.version !== 1 || data.requestId !== id || data.ok !== true || data.kind !== 'boundary-layer' || data.inputKey !== boundaryLayerInputKey(input)) return false;
  const attributed = readBoundaryLayerInput(data.input);
  if (!attributed || boundaryLayerInputKey(attributed) !== boundaryLayerInputKey(input)) return false;
  if (!data.result || typeof data.result !== 'object') return false;
  const own = Object.getOwnPropertyDescriptor(data.result, 'input');
  const resultInput = own && Object.hasOwn(own, 'value') ? readBoundaryLayerInput(own.value) : null;
  return !!resultInput && boundaryLayerInputKey(resultInput) === boundaryLayerInputKey(input);
}
