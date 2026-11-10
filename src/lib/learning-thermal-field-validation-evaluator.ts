/** Independent gate evaluator. Neither this entry nor its checker imports producer code. */
import { thermalFieldData, readThermalFieldInput } from './learning-thermal-field-input';
import { thermalFieldSuccessHeader, thermalFieldInputKey, type ThermalFieldReceipt } from './learning-thermal-field-protocol';
import { validateThermalFieldResult } from './learning-thermal-field-validation';
import { snapshotQubitWire, QubitWireLimit } from './learning-qubit-wire';
export function evaluateThermalFieldValidationRequest(value: unknown): ThermalFieldReceipt {
  let id: number | null = null;
  try {
    const d = thermalFieldData(value, ['version', 'requestId', 'input', 'response']);
    if (!d || d.version !== 1 || typeof d.requestId !== 'number' || !Number.isSafeInteger(d.requestId) || d.requestId <= 0) return { version: 1, requestId: null, phase: 'rejected', error: 'invalid-request' };
    id = d.requestId;
    const input = readThermalFieldInput(d.input);
    if (!input) return { version: 1, requestId: id, phase: 'rejected', error: 'invalid-input' };
    const owned = snapshotQubitWire(d.response);
    if (!thermalFieldSuccessHeader(owned, input, id)) return { version: 1, requestId: id, phase: 'rejected', error: 'invalid-response' };
    const result = validateThermalFieldResult(input, owned.result);
    return result ? { version: 1, requestId: id, phase: 'validated', kind: 'thermal-field', inputKey: thermalFieldInputKey(input), response: { ...owned, result } }
      : { version: 1, requestId: id, phase: 'rejected', error: 'invalid-response' };
  } catch (error) {
    return { version: 1, requestId: id, phase: 'rejected', error: error instanceof QubitWireLimit ? 'resource-limit' : 'invalid-response' };
  }
}
