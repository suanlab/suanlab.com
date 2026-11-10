/** Producer evaluator for the future isolated calculation Worker. */
import { calculateThermalField } from './learning-thermal-field';
import { thermalFieldObservations } from './learning-thermal-field-observations';
import { readThermalFieldRequest, thermalFieldInputKey, type ThermalFieldStarted, type ThermalFieldSuccess, type ThermalFieldFailure } from './learning-thermal-field-protocol';
import { snapshotQubitWire, QubitWireLimit } from './learning-qubit-wire';
export function evaluateThermalFieldRequest(value: unknown, started?: (message: ThermalFieldStarted) => void): ThermalFieldSuccess | ThermalFieldFailure {
  const parsed = readThermalFieldRequest(value);
  if ('failure' in parsed) return parsed.failure;
  const { requestId, input } = parsed.request;
  started?.({ version: 1, requestId, phase: 'started', kind: 'thermal-field' });
  try {
    const exact = calculateThermalField(input), observations = exact.kind === 'normalizable' ? thermalFieldObservations(exact) : null;
    return snapshotQubitWire({ version: 1, requestId, ok: true, kind: 'thermal-field', input, inputKey: thermalFieldInputKey(input), result: { exact, observations } }) as ThermalFieldSuccess;
  } catch (error) {
    return { version: 1, requestId, ok: false, error: error instanceof QubitWireLimit ? 'resource-limit' : 'calculation-failed' };
  }
}
