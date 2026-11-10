/** Plain transport only. Mathematical engines must remain out of reader bundles. */
import { readThermalFieldInput, thermalFieldData, type ThermalFieldInput } from './learning-thermal-field-input';
export type ThermalFieldExact = ReturnType<typeof import('./learning-thermal-field').calculateThermalField>;
export type ThermalFieldObservations = ReturnType<typeof import('./learning-thermal-field-observations').thermalFieldObservations>;
export interface ThermalFieldResult { exact: ThermalFieldExact; observations: ThermalFieldObservations | null }
export const thermalFieldInputKey = (input: ThermalFieldInput) => JSON.stringify(readThermalFieldInput(input));
export interface ThermalFieldSuccess {
  version: 1; requestId: number; ok: true; kind: 'thermal-field';
  input: ThermalFieldInput; inputKey: string; result: ThermalFieldResult;
}
export interface ThermalFieldFailure {
  version: 1; requestId: number | null; ok: false;
  error: 'invalid-request' | 'invalid-input' | 'resource-limit' | 'calculation-failed';
}
export interface ThermalFieldStarted { version: 1; requestId: number; phase: 'started'; kind: 'thermal-field' }
export type ThermalFieldReceipt =
  | { version: 1; requestId: number; phase: 'validated'; kind: 'thermal-field'; inputKey: string; response: ThermalFieldSuccess }
  | { version: 1; requestId: number | null; phase: 'rejected'; error: 'invalid-request' | 'invalid-input' | 'invalid-response' | 'resource-limit' };

export function readThermalFieldRequest(value: unknown): { request: { requestId: number; input: ThermalFieldInput } } | { failure: ThermalFieldFailure } {
  const own = thermalFieldData(value, ['version', 'requestId', 'input']);
  if (!own || own.version !== 1 || typeof own.requestId !== 'number' || !Number.isSafeInteger(own.requestId) || own.requestId <= 0) {
    return { failure: { version: 1, requestId: null, ok: false, error: 'invalid-request' } };
  }
  const input = readThermalFieldInput(own.input);
  return input ? { request: { requestId: own.requestId, input } } : { failure: { version: 1, requestId: own.requestId, ok: false, error: 'invalid-input' } };
}
export function thermalFieldSuccessHeader(value: unknown, input: ThermalFieldInput, id: number): value is ThermalFieldSuccess {
  const own = thermalFieldData(value, ['version', 'requestId', 'ok', 'kind', 'input', 'inputKey', 'result']);
  if (!own || own.version !== 1 || own.requestId !== id || own.ok !== true || own.kind !== 'thermal-field' || own.inputKey !== thermalFieldInputKey(input)) return false;
  const attributed = readThermalFieldInput(own.input);
  return !!attributed && thermalFieldInputKey(attributed) === thermalFieldInputKey(input);
}
