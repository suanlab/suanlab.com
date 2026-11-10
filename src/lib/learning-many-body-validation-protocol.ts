/** Trusted validation-worker receipts, separate from unverified calculation results.
 * The controller factory must create the application's validation worker. A receipt
 * is not an authentication token for data supplied by an arbitrary external port.
 */
import { manyBodyData, readManyBodyInput, manyBodyInputKey, type ManyBodyInput } from './learning-many-body-input';
import type { ManyBodySuccess } from './learning-many-body-protocol';
export type ManyBodyValidationRequest = { version: 1; requestId: number; input: ManyBodyInput; response: unknown };
export type ManyBodyValidationReceipt =
  | { version: 1; requestId: number; phase: 'validated'; kind: ManyBodyInput['kind']; inputKey: string; response: ManyBodySuccess }
  | { version: 1; requestId: number | null; phase: 'rejected'; error: 'invalid-request' | 'invalid-response' | 'validation-failed' };
export function readManyBodyValidationRequest(value: unknown): ManyBodyValidationRequest | null {
  const data = manyBodyData(value, ['version', 'requestId', 'input', 'response']);
  if (!data || data.version !== 1 || typeof data.requestId !== 'number' || !Number.isSafeInteger(data.requestId) || data.requestId <= 0) return null;
  const input = readManyBodyInput(data.input); return input ? { version: 1, requestId: data.requestId, input, response: data.response } : null;
}
/** Cheap header admission only, after the separate worker's mathematical gate. */
export function manyBodySuccessHeader(value: unknown, input: ManyBodyInput, requestId: number): value is ManyBodySuccess {
  try {
  const v = manyBodyData(value, ['version', 'requestId', 'ok', 'kind', 'inputKey', 'input', 'result']);
  if (!v || v.version !== 1 || v.requestId !== requestId || v.ok !== true || v.kind !== input.kind || v.inputKey !== manyBodyInputKey(input)) return false;
  const supplied = readManyBodyInput(v.input);
  return !!supplied && manyBodyInputKey(supplied) === v.inputKey && !!v.result && typeof v.result === 'object' && Object.getPrototypeOf(v.result) === Object.prototype;
  } catch { return false; }
}
