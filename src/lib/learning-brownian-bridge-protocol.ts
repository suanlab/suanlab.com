/** Reader-safe transport metadata. Mathematical checking runs in a Worker. */
import { readBrownianBridgeInput, type BrownianBridgeInput } from './learning-brownian-bridge-input';

export type BrownianBridgeResult = ReturnType<typeof import('./learning-brownian-bridge').calculateBrownianBridge>;
export const brownianBridgeInputKey = (input: BrownianBridgeInput) => JSON.stringify(readBrownianBridgeInput(input));
export interface BrownianBridgeSuccess {
  version: 1; requestId: number; ok: true; kind: 'brownian-bridge';
  input: BrownianBridgeInput; inputKey: string; result: BrownianBridgeResult;
}
export interface BrownianBridgeFailure {
  version: 1; requestId: number | null; ok: false;
  error: 'invalid-request' | 'invalid-input' | 'resource-limit' | 'calculation-failed';
}
export interface BrownianBridgeStarted { version: 1; requestId: number; phase: 'started'; kind: 'brownian-bridge'; }
export type BrownianBridgeReceipt =
  | { version: 1; requestId: number; phase: 'validated'; kind: 'brownian-bridge'; inputKey: string; response: BrownianBridgeSuccess }
  | { version: 1; requestId: number | null; phase: 'rejected'; error: 'invalid-request' | 'invalid-input' | 'invalid-response' | 'resource-limit' };

export function brownianBridgeData(value: unknown, keys: readonly string[]): Record<string, unknown> | null {
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

export function readBrownianBridgeRequest(value: unknown): { request: { requestId: number; input: BrownianBridgeInput } } | { failure: BrownianBridgeFailure } {
  const data = brownianBridgeData(value, ['version', 'requestId', 'input']);
  if (!data || data.version !== 1 || typeof data.requestId !== 'number' || !Number.isSafeInteger(data.requestId) || data.requestId <= 0) return { failure: { version: 1, requestId: null, ok: false, error: 'invalid-request' } };
  const input = readBrownianBridgeInput(data.input);
  return input ? { request: { requestId: data.requestId, input } } : { failure: { version: 1, requestId: data.requestId, ok: false, error: 'invalid-input' } };
}

export function brownianBridgeSuccessHeader(value: unknown, input: BrownianBridgeInput, id: number): value is BrownianBridgeSuccess {
  const data = brownianBridgeData(value, ['version', 'requestId', 'ok', 'kind', 'input', 'inputKey', 'result']);
  if (!data || data.version !== 1 || data.requestId !== id || data.ok !== true || data.kind !== 'brownian-bridge' || data.inputKey !== brownianBridgeInputKey(input)) return false;
  const attributed = readBrownianBridgeInput(data.input);
  if (!attributed || brownianBridgeInputKey(attributed) !== brownianBridgeInputKey(input)) return false;
  if (!data.result || typeof data.result !== 'object') return false;
  const own = Object.getOwnPropertyDescriptor(data.result, 'input');
  const resultInput = own && Object.hasOwn(own, 'value') ? readBrownianBridgeInput(own.value) : null;
  return !!resultInput && brownianBridgeInputKey(resultInput) === brownianBridgeInputKey(input);
}
