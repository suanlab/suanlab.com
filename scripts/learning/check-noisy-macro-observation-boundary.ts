/** Optional expensive range/performance diagnostic; not a fixed wall-time test. */
import { noisyMacroObservationModel } from '../../src/lib/learning-noisy-macro-observation';
const preparation = Array.from({ length: 16 }, (_, i) => i % 2 ? Number.MAX_VALUE / (i + 1) : Number.MIN_VALUE * (i + 1));
const rows = Array.from({ length: 16 }, (_, i) => Array.from({ length: 16 }, (_, j) => j === i ? Number.MAX_VALUE / (i + 1) : Number.MIN_VALUE * (j + 1)));
const started = performance.now();
const result = noisyMacroObservationModel(preparation, rows);
if (!Object.values(result.normalization).every(Boolean) || result.mutualInformation.sign < 0)
  throw new Error('Boundary normalization or information failed');
console.log(JSON.stringify({ inputStates: 16, outputLabels: 16, weightPattern: 'alternating-min-max-prior-and-dominant-diagonal-rows',
  milliseconds: performance.now() - started, normalization: result.normalization, informationArithmeticBits: result.informationArithmeticBits,
  mutualInformation: result.mutualInformation, jsonBytes: JSON.stringify(result).length }));
