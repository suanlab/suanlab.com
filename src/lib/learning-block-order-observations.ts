/** Producer-side directed logarithms; the checker uses different arithmetic. */
import { createRealTimeIntervalContext } from './learning-real-time-interval';
import { createTranscendentalContext } from './learning-interval-transcendentals';
import type { BlockOrderExact } from './learning-block-order-protocol';

const context = createRealTimeIntervalContext(1024), logs = createTranscendentalContext(1024);
const read = (s: string) => { const [n, d = '1'] = s.split('/'); return { n: BigInt(n), d: BigInt(d) }; };
export function blockOrderObservations(exact: BlockOrderExact) {
  const cache = new Map<string, ReturnType<typeof context.log>>();
  const log = (s: string) => {
    let result = cache.get(s);
    if (!result) {
      const q = read(s);
      result = logs.logRatio({ coefficient: q.n, exponent: 0 }, { coefficient: q.d, exponent: 0 });
      cache.set(s, result);
    }
    return result;
  };
  const scaled = (s: string, c: string) => context.multiply(log(s), context.from(read(c)));
  const form = exact.logarithmOperands;
  const lx = log(form.disagreementActivity), lz = log(exact.relativePartition);
  let fieldMean = context.integer(0), baseline = context.multiply(lx, context.from(read(form.absoluteLogPartitionBaseline.logXCoefficient)));
  for (let a = 0; a < exact.blocks; a++) {
    fieldMean = context.add(fieldMean, scaled(form.blockUpActivities[a], form.meanUpCounts[a]));
    baseline = context.add(baseline, scaled(form.blockUpActivities[a], form.absoluteLogPartitionBaseline.logYCoefficients[a]));
  }
  const logZ = context.add(lz, baseline), minusMeanDLogX = context.neg(context.multiply(lx, context.from(read(exact.meanDisagreements))));
  const microscopic = context.sub(context.add(lz, minusMeanDLogX), fieldMean);
  let profile = context.integer(0), hidden = context.integer(0);
  // Direct positive conditional forms, rather than subtracting two entropies.
  for (const row of exact.profiles) {
    const probability = context.from(read(row.probability));
    profile = context.sub(profile, context.multiply(probability, log(row.probability)));
    const conditional = context.sub(log(row.bondPolynomial), context.multiply(lx, context.from(read(row.conditionalMeanDisagreements))));
    hidden = context.add(hidden, context.multiply(probability, conditional));
  }
  const row = exact.profiles[exact.selected.profileIndex];
  const selectedHidden = context.sub(log(row.bondPolynomial), context.multiply(lx, context.from(read(row.conditionalMeanDisagreements))));
  const selectedFreeEnergy = context.neg(context.add(log(row.relativeWeight), baseline));
  const selectedEnergy = context.add(selectedFreeEnergy, selectedHidden);
  const residual = context.divideInteger(log(exact.selected.meanProductResidualPower), BigInt(exact.selected.meanProductResidualDivisor));
  return { precisionBits: 512 as const, interpretation: 'finite-discrete-profile-counting-measure' as const,
    logRelativePartition: context.observe(lz), logPartition: context.observe(logZ), freeEnergyOverTemperature: context.observe(context.neg(logZ)),
    microscopicEntropy: context.observe(microscopic), profileEntropy: context.observe(profile), meanHiddenEntropy: context.observe(hidden),
    selectedHiddenEntropy: context.observe(selectedHidden), selectedFreeEnergyOverTemperature: context.observe(selectedFreeEnergy),
    selectedMeanEnergyOverTemperature: context.observe(selectedEnergy), meanProductLogWeightResidual: context.observe(residual),
    localityLogCrossRatio: exact.localityWitness ? context.observe(log(exact.localityWitness.crossRatio)) : null };
}
