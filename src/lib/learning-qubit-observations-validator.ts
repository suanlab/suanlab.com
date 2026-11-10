/** Independent validation of owned observations, after the exact report has
 * passed its own validator. No preparation/product matrix engine imports. */
import { qubitData, type QubitInput } from './learning-qubit-input';
import { validateQubitDisplayReal } from './learning-qubit-display-validator';
import { validateQubitDensityObservation } from './learning-qubit-density-display-validator';
import { decodeQubitReportScalar, decodeQubitReportMatrix } from './learning-qubit-observation-values';
import { mWire, plus, Z } from './learning-qubit-channel-algebra';
import type { QubitResults } from './learning-qubit-protocol';
import type { QubitDisplayComplex } from './learning-qubit-display';
import type { QubitReportScalar, QubitReportMatrix, QubitObservationState } from './learning-qubit-observations-schema';

const realKeys = ['sign', 'status', 'lower', 'upper', 'coordinate', 'exact'];
function complexShape(value: unknown): value is QubitDisplayComplex {
  const d = qubitData(value, ['real', 'imaginary']);
  return !!d && !!qubitData(d.real, realKeys) && !!qubitData(d.imaginary, realKeys);
}
function checkCoordinates(exact: unknown, observed: unknown): boolean {
  if (exact === null || typeof exact !== 'object') return Object.is(exact, observed);
  if (Array.isArray(exact)) return Array.isArray(observed) && exact.length === observed.length
    && exact.every((child, i) => checkCoordinates(child, observed[i]));
  const source = exact as Record<string, unknown>;
  if (Array.isArray(source.real) && Array.isArray(source.imaginary)) {
    if (!complexShape(observed)) return false;
    const z = decodeQubitReportScalar(source as QubitReportScalar);
    return validateQubitDisplayReal(z.re, observed.real) && validateQubitDisplayReal(z.im, observed.imaginary);
  }
  const actual = qubitData(observed, Object.keys(source));
  return !!actual && Object.entries(source).every(([key, child]) => checkCoordinates(child, actual[key]));
}
function checkState(matrix: QubitReportMatrix | null, value: unknown): boolean {
  if (matrix === null) return value === null;
  const state = qubitData(value, ['matrix', 'matrixCoordinates', 'observation']);
  if (!state || JSON.stringify(state.matrix) !== JSON.stringify(matrix)
    || !checkCoordinates(matrix, state.matrixCoordinates)) return false;
  const observation = qubitData(state.observation, ['dimension', 'isDensity', 'hermitian', 'positiveSemidefinite', 'unitTrace', 'purity', 'bloch']);
  if (!observation) return false;
  const observableShape = (v: unknown) => {
    const d = qubitData(v, ['exact', 'display']);
    return !!d && complexShape(d.display);
  };
  if (observation.purity !== null && !observableShape(observation.purity)) return false;
  if (observation.bloch !== null && (!Array.isArray(observation.bloch) || observation.bloch.length !== 3
    || !observation.bloch.every(observableShape))) return false;
  return validateQubitDensityObservation(decodeQubitReportMatrix(matrix), (value as QubitObservationState).observation);
}
export function validateQubitObservations(input: QubitInput, result: QubitResults[keyof QubitResults], value: unknown): boolean {
  try {
    const envelope = qubitData(value, ['schemaVersion', 'coordinates', 'states']);
    if (!envelope || envelope.schemaVersion !== 1 || !checkCoordinates(result, envelope.coordinates)) return false;
    const expected: Record<string, QubitReportMatrix | null> = {};
    if (input.kind !== 'circuit') expected.input = input.state;
    switch (input.kind) {
      case 'circuit': break;
      case 'channel': {
        const r = result as QubitResults['channel'];
        expected.output = r.output; expected.bellOutput = r.bellOutput; expected.referenceMarginal = r.referenceMarginal;
        break;
      }
      case 'environment': {
        const r = result as QubitResults['environment'];
        expected.joint = r.joint; expected.system = r.system; expected.environment = r.environment;
        expected.referenceOutput = r.referenceOutput; expected.referenceMarginal = r.referenceMarginal;
        break;
      }
      case 'instrument': {
        const r = result as QubitResults['instrument'];
        expected.forgottenBeforeFeedback = r.forgotten; expected.systemAfterFeedback = r.feedforward;
        expected.classicalQuantumBeforeFeedback = r.classicalQuantum;
        const outputs = r.branches.map((branch, outcome) => {
          expected[`conditionalBeforeFeedback${outcome}`] = branch.conditional;
          const permute = (matrix: QubitReportMatrix) => {
            const exact = decodeQubitReportMatrix(matrix);
            return exact.map((row, i) => row.map((z, j) => outcome === 1 && input.correctZ ? exact[1 - i][1 - j] : z));
          };
          expected[`conditionalAfterFeedback${outcome}`] = branch.conditional === null ? null : mWire(permute(branch.conditional));
          return permute(branch.unnormalized);
        });
        expected.classicalQuantumAfterFeedback = mWire(Array.from({ length: 4 }, (_, i) => Array.from({ length: 4 }, (_, j) =>
          Math.floor(i / 2) === Math.floor(j / 2) ? outputs[Math.floor(i / 2)][i % 2][j % 2] : Z)));
        break;
      }
      case 'deferred': {
        const r = result as QubitResults['deferred'];
        expected.coherentJoint = r.coherent; expected.dephasedJoint = r.dephased;
        expected.coherentSystem = r.system; expected.coherentRecord = r.record;
        const matrix = decodeQubitReportMatrix(r.dephased);
        expected.dephasedSystem = mWire([0, 1].map(i => [0, 1].map(j => plus(matrix[2 * i][2 * j], matrix[2 * i + 1][2 * j + 1]))));
        expected.dephasedRecord = mWire([0, 1].map(i => [0, 1].map(j => plus(matrix[i][j], matrix[2 + i][2 + j]))));
        expected.recordSystem = r.recordSystem; expected.classicalCorrectedSystem = r.classicalCorrected;
        break;
      }
    }
    const states = qubitData(envelope.states, Object.keys(expected));
    return !!states && Object.entries(expected).every(([role, matrix]) => checkState(matrix, states[role]));
  } catch { return false; }
}
