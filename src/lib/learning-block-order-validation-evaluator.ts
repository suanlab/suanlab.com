import { blockOrderData, readBlockOrderInput } from './learning-block-order-input';
import { blockOrderInputKey, blockOrderSuccessHeader, type BlockOrderReceipt } from './learning-block-order-protocol';
import { validateBlockOrderResult } from './learning-block-order-validation';
import { snapshotQubitWire, QubitWireLimit } from './learning-qubit-wire';
export function evaluateBlockOrderValidationRequest(value: unknown): BlockOrderReceipt {
  let id: number | null = null;
  try {
    const d = blockOrderData(value,['version','requestId','input','response']);
    if (!d || d.version !== 1 || typeof d.requestId !== 'number' || !Number.isSafeInteger(d.requestId) || d.requestId <= 0)
      return { version: 1, requestId: null, phase: 'rejected', error: 'invalid-request' };
    id = d.requestId;
    const input = readBlockOrderInput(d.input);
    if (!input) return { version: 1, requestId: id, phase: 'rejected', error: 'invalid-input' };
    const response = snapshotQubitWire(d.response);
    if (!blockOrderSuccessHeader(response,input,id)) return { version: 1, requestId: id, phase: 'rejected', error: 'invalid-response' };
    const result = validateBlockOrderResult(input,response.result);
    return result ? { version: 1, requestId: id, phase: 'validated', kind: 'block-order', inputKey: blockOrderInputKey(input), response: { ...response,result } }
      : { version: 1, requestId: id, phase: 'rejected', error: 'invalid-response' };
  } catch (error) {
    return { version: 1, requestId: id, phase: 'rejected', error: error instanceof QubitWireLimit ? 'resource-limit' : 'invalid-response' };
  }
}
