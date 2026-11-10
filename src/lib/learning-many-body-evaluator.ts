/** Heavy synchronous engines. Import only from a worker or explicit CLI tests. */
import { spinChainModel } from './learning-spin-chain';
import { fermionFockModel } from './learning-fermion-fock';
import { bosonFockModel } from './learning-boson-fock';
import { bosonCoherentModel } from './learning-boson-coherent';
import { mpsModel } from './learning-mps';
import { schmidtModel } from './learning-schmidt';
import { manyBodyInputKey, readManyBodyRequest, type ManyBodyResponse, type ManyBodyStarted, type ManyBodySuccess } from './learning-many-body-protocol';

const calculations = {
  'spin-chain': spinChainModel, 'fermion-fock': fermionFockModel,
  'boson-fock': bosonFockModel, 'boson-coherent': bosonCoherentModel,
  mps: mpsModel, schmidt: schmidtModel,
};
export function evaluateManyBodyRequest(value: unknown, started?: (message: ManyBodyStarted) => void): ManyBodyResponse {
  const parsed = readManyBodyRequest(value);
  if ('failure' in parsed) return parsed.failure;
  const { requestId, input } = parsed.request;
  started?.({ version: 1, requestId, phase: 'started', kind: input.kind });
  try {
    const result = calculations[input.kind](input.parameters);
    // The dispatch key determines the result family; every parameters object is
    // already an owned frozen snapshot validated by that family's input module.
    return { version: 1, requestId, ok: true, kind: input.kind, input,
      inputKey: manyBodyInputKey(input), result } as ManyBodySuccess;
  } catch { return { version: 1, requestId, ok: false, error: 'calculation-failed' }; }
}
