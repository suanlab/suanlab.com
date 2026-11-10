/** Reader-safe types. All numerical dependencies below are type-only. */
import type { QubitDisplayComplex } from './learning-qubit-display';
import type { observeQubitDensity } from './learning-qubit-density-display';
export type QubitReportScalar = { readonly real: readonly string[]; readonly imaginary: readonly string[] };
export type QubitReportMatrix = readonly (readonly QubitReportScalar[])[];
export type QubitCoordinateTree = null | boolean | number | string | QubitDisplayComplex
  | QubitCoordinateTree[] | { [key: string]: QubitCoordinateTree };
export type QubitObservationState = {
  matrix: QubitReportMatrix;
  matrixCoordinates: QubitDisplayComplex[][];
  observation: ReturnType<typeof observeQubitDensity>;
};
export type QubitObservations = {
  schemaVersion: 1;
  /** Same structure as the exact report; scalar leaves become certificates. */
  coordinates: QubitCoordinateTree;
  /** Roles are explicit: never infer state semantics from a matrix shape. */
  states: Record<string, QubitObservationState | null>;
};
