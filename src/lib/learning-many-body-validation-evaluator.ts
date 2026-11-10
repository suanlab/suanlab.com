/** Heavy independent attribution. Import only in the validation worker or CLI tests. */
import { readManyBodyValidationRequest, type ManyBodyValidationReceipt } from './learning-many-body-validation-protocol';
import { readSpinChainSuccessForInput } from './learning-many-body-spin-validation';
import { readFermionFockSuccessForInput } from './learning-many-body-fermion-validation';
import { readBosonFockSuccessForInput } from './learning-many-body-boson-validation';
import { readBosonCoherentSuccessForInput } from './learning-many-body-coherent-validation';
import { readMpsSuccessForInput } from './learning-many-body-mps-validation';
import { readSchmidtSuccessForInput } from './learning-many-body-schmidt-validation';
import { manyBodyInputKey } from './learning-many-body-input';
const readers = { 'spin-chain': readSpinChainSuccessForInput, 'fermion-fock': readFermionFockSuccessForInput, 'boson-fock': readBosonFockSuccessForInput, 'boson-coherent': readBosonCoherentSuccessForInput, mps: readMpsSuccessForInput, schmidt: readSchmidtSuccessForInput };
export function evaluateManyBodyValidationRequest(value: unknown): ManyBodyValidationReceipt {
  let id: number | null = null;
  try {
    const request = readManyBodyValidationRequest(value); if (!request) return { version: 1, requestId: null, phase: 'rejected', error: 'invalid-request' };
    const { input, requestId } = request; id = requestId;
    const response = readers[input.kind](request.response, input, requestId);
    return response ? { version: 1, requestId, phase: 'validated', kind: input.kind, inputKey: manyBodyInputKey(input), response }
      : { version: 1, requestId, phase: 'rejected', error: 'invalid-response' };
  } catch { return { version: 1, requestId: id, phase: 'rejected', error: 'validation-failed' }; }
}
