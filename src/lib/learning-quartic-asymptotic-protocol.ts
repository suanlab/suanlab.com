/** Reader-safe transport metadata. Mathematical checking runs in a Worker. */
import { readQuarticAsymptoticInput, type QuarticAsymptoticInput } from './learning-quartic-asymptotic-input';

export type QuarticAsymptoticResult = ReturnType<typeof import('./learning-quartic-asymptotic').calculateQuarticAsymptotic>;
export const quarticAsymptoticInputKey = (input: QuarticAsymptoticInput) => JSON.stringify(input);
export interface QuarticAsymptoticSuccess {
  version: 1; requestId: number; ok: true; kind: 'quartic-asymptotic';
  input: QuarticAsymptoticInput; inputKey: string; result: QuarticAsymptoticResult;
}
export interface QuarticAsymptoticFailure {
  version: 1; requestId: number | null; ok: false;
  error: 'invalid-request' | 'invalid-input' | 'resource-limit' | 'calculation-failed';
}
export interface QuarticAsymptoticStarted { version: 1; requestId: number; phase: 'started'; kind: 'quartic-asymptotic'; }
export type QuarticAsymptoticReceipt =
  | { version: 1; requestId: number; phase: 'validated'; kind: 'quartic-asymptotic'; inputKey: string; response: QuarticAsymptoticSuccess }
  | { version: 1; requestId: number | null; phase: 'rejected'; error: 'invalid-request' | 'invalid-input' | 'invalid-response' | 'resource-limit' };

export function quarticAsymptoticData(value: unknown, keys: readonly string[]): Record<string, unknown> | null {
  if (!value || typeof value !== 'object' || Object.getPrototypeOf(value) !== Object.prototype || Reflect.ownKeys(value).length !== keys.length) return null;
  const out: Record<string, unknown> = {};
  for (const key of keys) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    if (!descriptor || !Object.hasOwn(descriptor, 'value')) return null;
    out[key] = descriptor.value;
  }
  return out;
}

export function readQuarticAsymptoticRequest(value: unknown): { request: { requestId: number; input: QuarticAsymptoticInput } } | { failure: QuarticAsymptoticFailure } {
  const data = quarticAsymptoticData(value, ['version', 'requestId', 'input']);
  if (!data || data.version !== 1 || typeof data.requestId !== 'number' || !Number.isSafeInteger(data.requestId) || data.requestId <= 0) return { failure: { version: 1, requestId: null, ok: false, error: 'invalid-request' } };
  const input = readQuarticAsymptoticInput(data.input);
  return input ? { request: { requestId: data.requestId, input } } : { failure: { version: 1, requestId: data.requestId, ok: false, error: 'invalid-input' } };
}

export function quarticAsymptoticSuccessHeader(value: unknown, input: QuarticAsymptoticInput, id: number): value is QuarticAsymptoticSuccess {
  const data = quarticAsymptoticData(value, ['version', 'requestId', 'ok', 'kind', 'input', 'inputKey', 'result']);
  if (!data || data.version !== 1 || data.requestId !== id || data.ok !== true || data.kind !== 'quartic-asymptotic' || data.inputKey !== quarticAsymptoticInputKey(input)) return false;
  const attributed = readQuarticAsymptoticInput(data.input);
  if (!attributed || quarticAsymptoticInputKey(attributed) !== quarticAsymptoticInputKey(input)) return false;
  if (!data.result || typeof data.result !== 'object') return false;
  const own = Object.getOwnPropertyDescriptor(data.result, 'input');
  const resultInput = own && Object.hasOwn(own, 'value') ? readQuarticAsymptoticInput(own.value) : null;
  return !!resultInput && quarticAsymptoticInputKey(resultInput) === quarticAsymptoticInputKey(input);
}
