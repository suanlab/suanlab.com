/** Transport types and owned requests; response validation is a separate gate.
 * Do not publish a worker result to the reader without the forthcoming result
 * shape/attribution/mathematical checks and cancellation controller.
 */
import { manyBodyData, readManyBodyInput, type ManyBodyInput } from './learning-many-body-input';
import type { spinChainModel } from './learning-spin-chain';
import type { fermionFockModel } from './learning-fermion-fock';
import type { bosonFockModel } from './learning-boson-fock';
import type { bosonCoherentModel } from './learning-boson-coherent';
import type { mpsModel } from './learning-mps';
import type { schmidtModel } from './learning-schmidt';
export { manyBodyKinds, isManyBodyInput, readManyBodyInput, snapshotManyBodyInput, manyBodyInputKey,
  parseManyBodyInputDraft, parseManyBodyStoredInput, type ManyBodyInput, type ManyBodyKind } from './learning-many-body-input';

export type ManyBodyResults = {
  'spin-chain': ReturnType<typeof spinChainModel>;
  'fermion-fock': ReturnType<typeof fermionFockModel>;
  'boson-fock': ReturnType<typeof bosonFockModel>;
  'boson-coherent': ReturnType<typeof bosonCoherentModel>;
  mps: ReturnType<typeof mpsModel>;
  schmidt: ReturnType<typeof schmidtModel>;
};
export type ManyBodySuccess = {
  [K in keyof ManyBodyResults]: {
    version: 1; requestId: number; ok: true; kind: K; inputKey: string;
    input: ManyBodyInput<K>; result: ManyBodyResults[K];
  }
}[keyof ManyBodyResults];
export type ManyBodyFailure = { version: 1; requestId: number | null; ok: false; error: 'invalid-request' | 'invalid-input' | 'calculation-failed' };
export type ManyBodyResponse = ManyBodySuccess | ManyBodyFailure;
export type ManyBodyStarted = { version: 1; requestId: number; phase: 'started'; kind: ManyBodyInput['kind'] };
export type ManyBodyRequest = Readonly<{ version: 1; requestId: number; input: ManyBodyInput }>;

export function readManyBodyRequest(value: unknown): { request: ManyBodyRequest } | { failure: ManyBodyFailure } {
  // Preserve a usable ID for errors without invoking an inherited/accessor ID.
  let requestId: number | null = null;
  try {
    const descriptor = value && typeof value === 'object' && Object.getPrototypeOf(value) === Object.prototype
      ? Object.getOwnPropertyDescriptor(value, 'requestId') : null;
    const id = descriptor && Object.hasOwn(descriptor, 'value') ? descriptor.value : null;
    if (typeof id === 'number' && Number.isSafeInteger(id) && id > 0) requestId = id;
  } catch { /* Invalid reflection is handled as an invalid envelope. */ }
  const envelope = manyBodyData(value, ['version', 'requestId', 'input']);
  if (!envelope || envelope.version !== 1 || typeof envelope.requestId !== 'number'
    || !Number.isSafeInteger(envelope.requestId) || envelope.requestId <= 0) {
    return { failure: { version: 1, requestId, ok: false, error: 'invalid-request' } };
  }
  // Use the same descriptor snapshot that will be used by the calculation.
  requestId = envelope.requestId;
  const input = readManyBodyInput(envelope.input);
  return input ? { request: Object.freeze({ version: 1, requestId, input }) }
    : { failure: { version: 1, requestId, ok: false, error: 'invalid-input' } };
}
