/** Heavy numerical dispatch belongs only to the worker. */
import { partitionCumulantModel } from './learning-partition-cumulants';
import { coupledPartitionModel } from './learning-coupled-partition';
import { energyLadderModel } from './learning-energy-ladder';
import { partitionRecord, isPartitionInput, snapshotPartitionInput, type PartitionResponse } from './learning-partition-protocol';
export function evaluatePartitionRequest(value: unknown): PartitionResponse {
  const requestId = partitionRecord(value) && Number.isSafeInteger(value.requestId) && (value.requestId as number) > 0 ? value.requestId as number : null;
  if (!partitionRecord(value) || value.version !== 1 || requestId === null) return { version: 1, requestId, ok: false, error: 'invalid-request' };
  if (!isPartitionInput(value.input)) return { version: 1, requestId, ok: false, error: 'invalid-input' };
  try {
    const input = snapshotPartitionInput(value.input);
    if (input.kind === 'finite-spectrum-partition-cumulant-generators') return { version: 1, requestId, ok: true, kind: input.kind, result: partitionCumulantModel(input.parameters) };
    if (input.kind === 'coupled-binary-sites-partition-factorization') return { version: 1, requestId, ok: true, kind: input.kind, result: coupledPartitionModel(input.parameters) };
    return { version: 1, requestId, ok: true, kind: input.kind, result: energyLadderModel(input.parameters) };
  } catch { return { version: 1, requestId, ok: false, error: 'calculation-failed' }; }
}
