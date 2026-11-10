/** Heavy numerical imports are confined to worker-side dispatch. */
import { weightedBoltzmannModel } from './learning-weighted-boltzmann';
import { finiteGibbsModel } from './learning-finite-gibbs';
import { finiteGibbsPath } from './learning-finite-gibbs-path';
import { quadraticGibbsModel } from './learning-quadratic-gibbs';
import { boltzmannRecord, isBoltzmannGibbsInput, snapshotBoltzmannGibbsInput, boltzmannGibbsInputKey, type BoltzmannGibbsResponse } from './learning-boltzmann-gibbs-protocol';
export function evaluateBoltzmannGibbsRequest(value: unknown): BoltzmannGibbsResponse {
  const requestId = boltzmannRecord(value) && Number.isSafeInteger(value.requestId) && (value.requestId as number) > 0 ? value.requestId as number : null;
  if (!boltzmannRecord(value) || value.version !== 1 || requestId === null) return { version: 1, requestId, ok: false, error: 'invalid-request' };
  if (!isBoltzmannGibbsInput(value.input)) return { version: 1, requestId, ok: false, error: 'invalid-input' };
  try {
    const input = snapshotBoltzmannGibbsInput(value.input), base = { version: 1 as const, requestId, ok: true as const, inputKey: boltzmannGibbsInputKey(input) };
    if (input.kind === 'reference-measure-boltzmann-softmax-identifiability') return { ...base, kind: input.kind, input, result: weightedBoltzmannModel(input.parameters) };
    if (input.kind === 'finite-gibbs-conditionals-scan-support') {
      const result = finiteGibbsModel(input.parameters);
      return { ...base, kind: input.kind, input, result, path: input.path === undefined ? null : finiteGibbsPath(result, input.path) };
    }
    return { ...base, kind: input.kind, input, result: quadraticGibbsModel(input.parameters) };
  } catch { return { version: 1, requestId, ok: false, error: 'calculation-failed' }; }
}
