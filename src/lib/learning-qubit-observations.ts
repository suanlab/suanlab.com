/** Calculation Worker display preparation. Not a reader dependency. */
import { observeQubitComplex } from './learning-qubit-display';
import { observeQubitDensity } from './learning-qubit-density-display';
import { decodeQubitReportScalar, decodeQubitReportMatrix } from './learning-qubit-observation-values';
import { mWire, product, tensor, mAdd, partialTrace, Z, O, type M } from './learning-qubit-channel-algebra';
import type { QubitInput } from './learning-qubit-input';
import type { QubitResults } from './learning-qubit-protocol';
import type { QubitCoordinateTree, QubitObservations, QubitReportScalar, QubitReportMatrix } from './learning-qubit-observations-schema';

function coordinates(value: unknown): QubitCoordinateTree {
  if (value === null || typeof value === 'boolean' || typeof value === 'number' || typeof value === 'string') return value;
  if (Array.isArray(value)) return value.map(coordinates);
  const record = value as Record<string, unknown>;
  if (Array.isArray(record.real) && Array.isArray(record.imaginary))
    return observeQubitComplex(decodeQubitReportScalar(record as QubitReportScalar));
  return Object.fromEntries(Object.entries(record).map(([key, child]) => [key, coordinates(child)]));
}
export function prepareQubitObservations(input: QubitInput, result: QubitResults[keyof QubitResults]): QubitObservations {
  const states: QubitObservations['states'] = {};
  const add = (role: string, matrix: QubitReportMatrix | null) => {
    if (matrix === null) { states[role] = null; return; }
    const exact = decodeQubitReportMatrix(matrix);
    states[role] = { matrix, matrixCoordinates: exact.map(row => row.map(z => observeQubitComplex(z))), observation: observeQubitDensity(exact) };
  };
  if (input.kind !== 'circuit') add('input', input.state);
  switch (input.kind) {
    case 'circuit': break; // Prefix purity/local Bloch already belong to the exact report.
    case 'channel': {
      const r = result as QubitResults['channel'];
      add('output', r.output); add('bellOutput', r.bellOutput); add('referenceMarginal', r.referenceMarginal);
      break;
    }
    case 'environment': {
      const r = result as QubitResults['environment'];
      for (const role of ['joint', 'system', 'environment', 'referenceOutput', 'referenceMarginal'] as const) add(role, r[role]);
      break;
    }
    case 'instrument': {
      const r = result as QubitResults['instrument'];
      add('forgottenBeforeFeedback', r.forgotten);
      add('systemAfterFeedback', r.feedforward);
      add('classicalQuantumBeforeFeedback', r.classicalQuantum);
      const X: M = [[Z, O], [O, Z]];
      const outputs = r.branches.map((branch, index) => {
        const unnormalized = decodeQubitReportMatrix(branch.unnormalized);
        const correction = index === 1 && input.correctZ;
        add(`conditionalBeforeFeedback${index}`, branch.conditional);
        const conditional = branch.conditional === null ? null : decodeQubitReportMatrix(branch.conditional);
        add(`conditionalAfterFeedback${index}`, conditional === null ? null : mWire(correction ? product(product(X, conditional), X) : conditional));
        return correction ? product(product(X, unnormalized), X) : unnormalized;
      });
      add('classicalQuantumAfterFeedback', mWire(mAdd(tensor([[O, Z], [Z, Z]], outputs[0]), tensor([[Z, Z], [Z, O]], outputs[1]))));
      break;
    }
    case 'deferred': {
      const r = result as QubitResults['deferred'];
      add('coherentJoint', r.coherent); add('dephasedJoint', r.dephased);
      add('coherentSystem', r.system); add('coherentRecord', r.record);
      const dephased = decodeQubitReportMatrix(r.dephased);
      add('dephasedSystem', mWire(partialTrace(dephased, 2, 2, 'first')));
      add('dephasedRecord', mWire(partialTrace(dephased, 2, 2, 'second')));
      add('recordSystem', r.recordSystem); add('classicalCorrectedSystem', r.classicalCorrected);
      break;
    }
  }
  return { schemaVersion: 1, coordinates: coordinates(result), states };
}
