/** Fixed preparations: independent of the saved interactive experiment. */
import { partitionPresetInput, isPartitionKind, type PartitionKind } from './learning-partition';
import type { PartitionInput } from './learning-partition-protocol';
const indices: Record<PartitionKind, readonly (readonly number[])[]> = {
  'finite-spectrum-partition-cumulant-generators': [[0], [0, 6], [0, 1, 2], [3, 4, 5, 7]],
  'coupled-binary-sites-partition-factorization': [[0], [1, 2], [0], [7, 6]],
  'infinite-energy-ladder-normalization-cutoff': [[0], [0, 1], [2, 3], [4, 5, 6, 7]],
};
export function partitionConceptInputs(kind: PartitionKind, step: number): readonly PartitionInput[] {
  if (!isPartitionKind(kind) || !Number.isInteger(step) || step < 0 || step > 3) throw new RangeError('Invalid partition concept');
  return Object.freeze(indices[kind][step].map(i => partitionPresetInput(kind, i)));
}
