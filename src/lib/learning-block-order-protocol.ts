/** Transport types and attribution only. No runtime model or checker imports. */
import { blockOrderData, readBlockOrderInput, type BlockOrderInput } from './learning-block-order-input';
export type BlockOrderExact = ReturnType<typeof import('./learning-block-order').calculateBlockOrder>;
export type BlockOrderWireExact = Omit<BlockOrderExact, 'profiles'> & { profileChunks: BlockOrderExact['profiles'][] };
export type BlockOrderObservations = ReturnType<typeof import('./learning-block-order-observations').blockOrderObservations>;
export interface BlockOrderResult { exact: BlockOrderWireExact; observations: BlockOrderObservations }
export interface BlockOrderSuccess {
  version: 1; requestId: number; ok: true; kind: 'block-order'; input: BlockOrderInput; inputKey: string; result: BlockOrderResult;
}
export interface BlockOrderFailure {
  version: 1; requestId: number | null; ok: false; error: 'invalid-request' | 'invalid-input' | 'resource-limit' | 'calculation-failed';
}
export interface BlockOrderStarted { version: 1; requestId: number; phase: 'started'; kind: 'block-order' }
export type BlockOrderReceipt =
  | { version: 1; requestId: number; phase: 'validated'; kind: 'block-order'; inputKey: string; response: BlockOrderSuccess }
  | { version: 1; requestId: number | null; phase: 'rejected'; error: 'invalid-request' | 'invalid-input' | 'invalid-response' | 'resource-limit' };
export const blockOrderInputKey = (input: BlockOrderInput) => JSON.stringify(readBlockOrderInput(input));
export function readBlockOrderRequest(value: unknown): { request: { requestId: number; input: BlockOrderInput } } | { failure: BlockOrderFailure } {
  const d = blockOrderData(value, ['version','requestId','input']);
  if (!d || d.version !== 1 || typeof d.requestId !== 'number' || !Number.isSafeInteger(d.requestId) || d.requestId <= 0)
    return { failure: { version: 1, requestId: null, ok: false, error: 'invalid-request' } };
  const input = readBlockOrderInput(d.input);
  return input ? { request: { requestId: d.requestId, input } } : { failure: { version: 1, requestId: d.requestId, ok: false, error: 'invalid-input' } };
}
/** Attribution is deliberately separate from the mathematical validation gate. */
export function blockOrderSuccessHeader(value: unknown, input: BlockOrderInput, id: number): value is BlockOrderSuccess {
  const d = blockOrderData(value, ['version','requestId','ok','kind','input','inputKey','result']);
  if (!d || d.version !== 1 || d.requestId !== id || d.ok !== true || d.kind !== 'block-order' || d.inputKey !== blockOrderInputKey(input)) return false;
  const attributed = readBlockOrderInput(d.input);
  return !!attributed && blockOrderInputKey(attributed) === blockOrderInputKey(input);
}
