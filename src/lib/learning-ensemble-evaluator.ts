/** Worker-side dispatch only. Do not import into UI components. */
import { finiteReservoirModel } from './learning-finite-reservoir';
import { canonicalBinaryModel } from './learning-canonical-binary';
import { grandCanonicalOccupancyModel } from './learning-grand-canonical-occupancy';
import { ensembleRecord, isEnsembleInput, snapshotEnsembleInput, type EnsembleResponse } from './learning-ensemble-protocol';
export function evaluateEnsembleRequest(value: unknown): EnsembleResponse {
  const requestId = ensembleRecord(value) && Number.isSafeInteger(value.requestId) && (value.requestId as number) > 0 ? value.requestId as number : null;
  if (!ensembleRecord(value) || value.version !== 1 || requestId === null) return { version: 1, requestId, ok: false, error: 'invalid-request' };
  if (!isEnsembleInput(value.input)) return { version: 1, requestId, ok: false, error: 'invalid-input' };
  try {
    const input = snapshotEnsembleInput(value.input);
    if (input.kind === 'finite-binary-reservoir-canonical-comparison') {
      const p = input.parameters;
      return { version: 1, requestId, ok: true, kind: input.kind, result: finiteReservoirModel(p.subsystemSites, p.reservoirSites, p.totalExcitations) };
    }
    if (input.kind === 'canonical-binary-energy-fluctuations') return { version: 1, requestId, ok: true, kind: input.kind, result: canonicalBinaryModel(input.parameters) };
    return { version: 1, requestId, ok: true, kind: input.kind, result: grandCanonicalOccupancyModel(input.parameters) };
  } catch {
    return { version: 1, requestId, ok: false, error: 'calculation-failed' };
  }
}
