import { calculateBlockOrder } from './learning-block-order';
import { blockOrderObservations } from './learning-block-order-observations';
import { blockOrderInputKey, readBlockOrderRequest, type BlockOrderFailure, type BlockOrderStarted, type BlockOrderSuccess } from './learning-block-order-protocol';
import { snapshotQubitWire, QubitWireLimit } from './learning-qubit-wire';
export function evaluateBlockOrderRequest(value: unknown, started?: (message: BlockOrderStarted) => void): BlockOrderSuccess | BlockOrderFailure {
  const parsed = readBlockOrderRequest(value);
  if ('failure' in parsed) return parsed.failure;
  const { requestId, input } = parsed.request;
  started?.({ version: 1, requestId, phase: 'started', kind: 'block-order' });
  try {
    const raw = calculateBlockOrder(input), { profiles, ...rest } = raw;
    const profileChunks = Array.from({ length: Math.ceil(profiles.length / 128) }, (_, i) => profiles.slice(128*i, 128*(i+1)));
    return snapshotQubitWire({ version: 1, requestId, ok: true, kind: 'block-order', input, inputKey: blockOrderInputKey(input),
      result: { exact: { ...rest, profileChunks }, observations: blockOrderObservations(raw) } }) as BlockOrderSuccess;
  } catch (error) {
    return { version: 1, requestId, ok: false, error: error instanceof QubitWireLimit ? 'resource-limit' : 'calculation-failed' };
  }
}
