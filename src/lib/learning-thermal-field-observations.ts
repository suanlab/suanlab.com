/** Directed finite-Gaussian log Z/F/entropy enclosures. Producer-side only. */
import { f } from './learning-quantum-hilbert-core';
import { createTranscendentalContext } from './learning-interval-transcendentals';
import { createRealTimeIntervalContext } from './learning-real-time-interval';
import type { calculateThermalField } from './learning-thermal-field';

type Normal = Extract<ReturnType<typeof calculateThermalField>, { kind: 'normalizable' }>;
const read = (s: string) => { const [n, d = '1'] = s.split('/'); return f(BigInt(n), BigInt(d)); };
const context = createRealTimeIntervalContext(1024), logs = createTranscendentalContext(1024);
/** Apply range reduction to the exact ratio before any fixed-point conversion.
 * A valid 32-site determinant may be far smaller than 2^-1024. */
const logRational = (s: string) => {
  const q = read(s);
  return logs.logRatio({ coefficient: q.n, exponent: 0 }, { coefficient: q.d, exponent: 0 });
};
export function thermalFieldObservations(exact: Normal) {
  const { logPartition: form } = exact, T = read(form.temperature), q = context.integer(form.dimension);
  const logTwoPi = context.log(context.scale(context.pi(), 2));
  const amplitude = context.divideInteger(context.multiply(q, context.add(logTwoPi, logRational(form.temperature))), 2);
  const logZ = context.add(context.sub(context.sub(amplitude, context.divideInteger(logRational(form.restrictedDeterminant), 2)),
    context.scale(logRational(form.referenceScale), form.dimension)), context.from(read(form.constant)));
  const freeEnergy = context.multiply(context.neg(context.from(T)), logZ);
  const entropy = context.add(logZ, context.from(read(exact.entropyRationalAddend)));
  const energy = read(exact.expectedEnergy), gibbsAddend = f(energy.n * T.d, energy.d * T.n);
  const gibbsEntropy = context.add(logZ, context.from(gibbsAddend));
  return {
    precisionBits: 512 as const,
    interpretation: 'finite-gaussian-fixed-reference-measure' as const,
    logPartition: context.observe(logZ), freeEnergy: context.observe(freeEnergy),
    thermodynamicEntropy: context.observe(entropy), gibbsEntropy: context.observe(gibbsEntropy),
  };
}
