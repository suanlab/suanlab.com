/** Public finite relative-level API; the numerical core also accepts exact
 * Hamiltonians assembled from several binary inputs without rounding them. */
import { binaryProduct, binarySum } from './learning-binary-products';
import { exactFinitePartition } from './learning-partition-exact-core';
export type { PartitionObservation, PartitionRational } from './learning-partition-exact-core';
export interface PartitionCumulantParameters { energyLevels: readonly number[]; energyOffset: number; temperature: number }
export function partitionCumulantModel(parameters: PartitionCumulantParameters) {
  const { energyLevels, energyOffset, temperature } = parameters;
  if (!Array.isArray(energyLevels) || energyLevels.length < 1 || energyLevels.length > 32 || !Array.from(energyLevels).every(v => typeof v === 'number' && Number.isFinite(v)) || !Number.isFinite(energyOffset) || !Number.isFinite(temperature)) throw new RangeError('Expected 1..32 finite relative levels, a finite common offset and finite temperature');
  const actual = energyLevels.map(level => binarySum([binaryProduct(energyOffset), binaryProduct(level)]));
  const result = exactFinitePartition(actual, temperature);
  return { ...result, scope: 'finite-explicit-microstates-fixed-hamiltonian-with-common-offset' as const,
    parameters: { energyLevels: [...energyLevels], energyOffset, temperature },
    states: result.states.map((state, i) => ({ ...state, relativeEnergy: energyLevels[i] })) };
}
