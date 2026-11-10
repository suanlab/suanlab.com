import { drivenTwoStateThermodynamics } from './learning-driven-two-state-thermodynamics';
import { reservoirTwoStateThermodynamics } from './learning-reservoir-two-state-thermodynamics';
import { thermalRecord, isThermalInput, isThermalRequestEnvelope, snapshotThermalInput, thermalInputKey, thermalKinds, type ThermalResponse } from './learning-thermal-protocol';
export function evaluateThermalRequest(value: unknown): ThermalResponse {
  const requestId = thermalRecord(value) && Number.isSafeInteger(value.requestId) && (value.requestId as number) > 0 ? value.requestId as number : null;
  if (!isThermalRequestEnvelope(value)) return { version: 1, requestId, ok: false, error: 'invalid-request' };
  if (!isThermalInput(value.input)) return { version: 1, requestId, ok: false, error: 'invalid-input' };
  try {
    const input = snapshotThermalInput(value.input), base = { version: 1 as const, requestId: value.requestId, ok: true as const, inputKey: thermalInputKey(input) };
    return input.kind === thermalKinds[0]
      ? { ...base, kind: input.kind, input, result: drivenTwoStateThermodynamics(input.parameters) }
      : { ...base, kind: input.kind, input, result: reservoirTwoStateThermodynamics(input.parameters) };
  } catch { return { version: 1, requestId, ok: false, error: 'calculation-failed' }; }
}
