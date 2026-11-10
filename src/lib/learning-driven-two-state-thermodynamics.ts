/** Unregistered numerical preparation: an isothermal two-quench protocol.
 * Energies are kBT ln(r), work is on the system, heat is into the system.
 * Endpoint atoms integrate every intervening jump count; they are not samples.
 */
import { fraction as f, zero, one, add, sub, mul, div, equal, exact, point, logarithm, plus, minus, negative, scale, square, observe, floor, ceil, min, max, entropyScale as S, type F, type I } from './learning-entropy-production-math';
import { createTranscendentalContext } from './learning-interval-transcendentals';
import { drivenThermalParameterKeys as keys, isDrivenThermalParameters, type DrivenThermalParameters } from './learning-thermal-input';
export { isDrivenThermalParameters, type DrivenThermalParameters } from './learning-thermal-input';
const Z = BigInt(0), ctx = createTranscendentalContext(256);
function product(a: I, b: I): I {
  const values = [a.lower * b.lower, a.lower * b.upper, a.upper * b.lower, a.upper * b.upper];
  return { lower: floor(values.reduce(min), S), upper: ceil(values.reduce(max), S) };
}
function logRange(a: I): I {
  if (a.lower <= Z) throw new RangeError('Positive probability lower bound required');
  return { lower: logarithm(f(a.lower, S)).lower, upper: logarithm(f(a.upper, S)).upper };
}
function nonnegative(a: I): I {
  if (a.upper < Z) throw new RangeError('Nonnegative thermodynamic enclosure inconsistent');
  return { lower: max(Z, a.lower), upper: a.upper };
}
type Probability = { rational: F | null; range: I };
const probability = (x: F): Probability => ({ rational: x, range: point(x) });
const complement = (x: Probability): Probability => ({ rational: x.rational ? sub(one, x.rational) : null, range: minus(point(one), x.range) });
const logProbability = (x: Probability) => x.rational ? logarithm(x.rational) : logRange(x.range);
const observedProbability = (x: Probability) => x.rational ? exact(x.rational) : observe(x.range);
function shannon(x: Probability) { const y = complement(x); return negative(plus(product(x.range, logProbability(x)), product(y.range, logProbability(y)))); }
function relativeEntropy(x: Probability, equilibrium: F): I {
  if (x.rational && equal(x.rational, equilibrium)) return point(zero);
  const y = complement(x);
  return nonnegative(plus(product(x.range, minus(logProbability(x), logarithm(equilibrium))), product(y.range, minus(logProbability(y), logarithm(sub(one, equilibrium))))));
}
function hold(start: Probability, ratio: F, clock: F, time: F) {
  const pi = div(one, add(one, ratio)), mean = mul(clock, time), decay = mean.n === Z ? point(one) : ctx.expNegativeRatio({ coefficient: mean.n, exponent: 0 }, { coefficient: mean.d, exponent: 0 });
  const mixing = minus(point(one), decay), equilibriumStart = start.rational !== null && equal(start.rational, pi);
  const end = time.n === Z ? start : equilibriumStart ? probability(pi) : { rational: null, range: plus(point(pi), product(minus(start.range, point(pi)), decay)) };
  if (end.range.lower <= Z || end.range.upper >= S) throw new RangeError('Strictly positive binary law lost');
  const transition = time.n === Z ? [[point(one), point(zero)], [point(zero), point(one)]] : [[plus(point(sub(one, pi)), scale(decay, pi)), scale(mixing, pi)], [scale(mixing, sub(one, pi)), plus(point(pi), scale(decay, sub(one, pi)))]];
  const energy = logarithm(ratio), heat = time.n === Z || equilibriumStart ? point(zero) : product(energy, minus(end.range, start.range));
  const sigma = time.n === Z || equilibriumStart ? point(zero) : nonnegative(minus(minus(shannon(end), shannon(start)), heat));
  return { start, end, pi, decay, transition, energy, heat, sigma, zeroProduction: time.n === Z || equilibriumStart,
    generator: [[negative(point(mul(clock, pi))), point(mul(clock, pi))], [point(mul(clock, sub(one, pi))), negative(point(mul(clock, sub(one, pi))))]] };
}
export function drivenTwoStateThermodynamics(input: DrivenThermalParameters) {
  if (!isDrivenThermalParameters(input)) throw new RangeError('Eighteen owned integer fields; positive odds/clocks/temperature/denominators, full-support initial law, each hold0..4s, temperature<=1e6K, field magnitudes<=1e9');
  const parameters = Object.freeze(Object.fromEntries(keys.map(k => [k, input[k] + 0]))) as unknown as Readonly<DrivenThermalParameters>;
  const p = parameters, q = (name: string) => f(BigInt(p[(name + 'Numerator') as keyof DrivenThermalParameters]), BigInt(p[(name + 'Denominator') as keyof DrivenThermalParameters]));
  const r = [q('r0'), q('r1'), q('r2')], energies = r.map(logarithm), initial = probability(f(BigInt(p.initialOneNumerator), BigInt(p.initialDenominator)));
  const times = [q('time1'), q('time2')], clocks = [q('clock1'), q('clock2')], first = hold(initial, r[1], clocks[0], times[0]), second = hold(first.end, r[2], clocks[1], times[1]);
  const equilibrium = r.map(x => div(one, add(one, x))), Zs = r.map(x => add(one, div(one, x)));
  const temperature = q('temperature'), boltzmann = f(BigInt(1380649), BigInt(10) ** BigInt(29)), thermalEnergy = mul(temperature, boltzmann);
  const exactlyZeroWork = (equal(r[0], r[1]) && equal(r[1], r[2])) || (times[0].n === Z && equal(r[0], r[2]));
  const exactlyZeroMeanWork = exactlyZeroWork || (equal(initial.rational!, first.pi) && equal(r[0], r[2]));
  const w1 = product(initial.range, logarithm(div(r[1], r[0]))), w2 = product(first.end.range, logarithm(div(r[2], r[1]))), work = exactlyZeroMeanWork ? point(zero) : plus(w1, w2), heat = plus(first.heat, second.heat);
  const meanSigma = plus(first.sigma, second.sigma), initialH = shannon(initial), initialU = product(initial.range, energies[0]);
  const stages = [
    { label: 'initial', gap: 0, law: initial, work: point(zero), heat: point(zero), sigma: point(zero), time: zero },
    { label: 'first-quench', gap: 1, law: initial, work: w1, heat: point(zero), sigma: point(zero), time: zero },
    { label: 'first-hold', gap: 1, law: first.end, work: w1, heat: first.heat, sigma: first.sigma, time: times[0] },
    { label: 'second-quench', gap: 2, law: first.end, work, heat: first.heat, sigma: first.sigma, time: times[0] },
    { label: 'second-hold', gap: 2, law: second.end, work, heat, sigma: meanSigma, time: add(times[0], times[1]) },
  ].map(s => {
    const U = product(s.law.range, energies[s.gap]), H = shannon(s.law), Fnoneq = minus(U, H);
    return { label: s.label, time: exact(s.time), stateOneProbability: observedProbability(s.law), meanEnergy: observe(U), meanWork: observe(s.work), heatIntoSystem: observe(s.heat), bathEntropy: observe(negative(s.heat)), systemEntropy: observe(H), totalEntropyProduction: observe(s.sigma), nonequilibriumFreeEnergy: observe(Fnoneq), canonicalFreeEnergy: observe(negative(logarithm(Zs[s.gap]))), relativeEntropyToCanonical: observe(relativeEntropy(s.law, equilibrium[s.gap])), firstLawResidual: exact(zero), independentFirstLawResidual: observe(minus(minus(minus(U, initialU), s.work), s.heat)), meanEnergyJ: observe(scale(U, thermalEnergy)), meanWorkJ: observe(scale(s.work, thermalEnergy)), heatIntoSystemJ: observe(scale(s.heat, thermalEnergy)) };
  });
  const initialLaw = [complement(initial), initial], finalLaw = [complement(second.end), second.end];
  const workAtoms = [0, 1].flatMap(x => [0, 1].map(y => {
    const reachable = times[0].n !== Z || x === y;
    const mass = reachable ? scale(first.transition[x][y], initialLaw[x].rational!) : point(zero);
    const inverseWorkRatio = mul(x ? div(r[0], r[1]) : one, y ? div(r[1], r[2]) : one);
    const atomWork = logarithm(div(one, inverseWorkRatio));
    const reverseMass = reachable ? scale(first.transition[y][x], y ? equilibrium[2] : sub(one, equilibrium[2])) : point(zero);
    const canonicalX = x ? equilibrium[0] : sub(one, equilibrium[0]), canonicalY = y ? equilibrium[2] : sub(one, equilibrium[2]);
    const transitionRatio = x === y ? one : x === 0 ? div(one, r[1]) : r[1];
    const forwardReverseRatio = reachable ? mul(div(initialLaw[x].rational!, canonicalY), transitionRatio) : null;
    const crooksPrediction = div(div(Zs[2], Zs[0]), inverseWorkRatio), initialCorrection = div(initialLaw[x].rational!, canonicalX);
    if (forwardReverseRatio && !equal(forwardReverseRatio, mul(crooksPrediction, initialCorrection))) throw new RangeError('Crooks endpoint ratio identity failed');
    return { x, y, reachable, mass, atomWork, inverseWorkRatio, reverseMass, forwardReverseRatio, crooksPrediction, initialCorrection };
  }));
  const workMoment = workAtoms.reduce((total, a) => plus(total, scale(a.mass, a.inverseWorkRatio)), point(zero));
  let workVariance = point(zero);
  for (let i = 0; i < workAtoms.length; i++) for (let j = i + 1; j < workAtoms.length; j++) {
    const a = workAtoms[i], b = workAtoms[j];
    if (!a.reachable || !b.reachable) continue;
    const difference = logarithm(div(b.inverseWorkRatio, a.inverseWorkRatio));
    workVariance = plus(workVariance, product(product(a.mass, b.mass), square(difference)));
  }
  const endpointAtoms = [0, 1].flatMap(x => [0, 1].flatMap(y => [0, 1].map(z => {
    const reachable = (times[0].n !== Z || x === y) && (times[1].n !== Z || y === z);
    if (!reachable) return { x, y, z, reachable, probability: exact(zero), work: null, heatIntoSystem: null, energyChange: null, bathEntropy: null, systemEntropyChange: null, totalEntropy: null, inverseEntropyContribution: exact(zero), firstLawResidual: null };
    const mass = scale(product(first.transition[x][y], second.transition[y][z]), initialLaw[x].rational!);
    const power = (v: F, exponent: number) => exponent === 0 ? one : exponent > 0 ? v : div(one, v);
    const heatRatio = mul(power(r[1], y - x), power(r[2], z - y));
    const workRatio = div(one, workAtoms[x * 2 + y].inverseWorkRatio), deltaURatio = div(z ? r[2] : one, x ? r[0] : one);
    if (!equal(mul(workRatio, heatRatio), deltaURatio)) throw new RangeError('Endpoint first-law ratio identity failed');
    const W = logarithm(workRatio), Q = logarithm(heatRatio), deltaU = logarithm(deltaURatio);
    const finalRational = finalLaw[z].rational;
    const system = finalRational ? logarithm(div(initialLaw[x].rational!, finalRational)) : minus(logProbability(initialLaw[x]), logProbability(finalLaw[z]));
    const total = finalRational ? logarithm(div(div(initialLaw[x].rational!, finalRational), heatRatio)) : minus(system, Q);
    // Conditional detailed balance cancels the forward initial weight exactly.
    // This is the reverse-protocol endpoint probability, not an empirical sum.
    const inverseContribution = product(finalLaw[z].range, product(second.transition[z][y], first.transition[y][x]));
    return { x, y, z, reachable, probability: observe(mass), work: observe(W), heatIntoSystem: observe(Q), energyChange: observe(deltaU), bathEntropy: observe(negative(Q)), systemEntropyChange: observe(system), totalEntropy: observe(total), inverseEntropyContribution: observe(inverseContribution), firstLawResidual: exact(zero) };
  })));
  const computedIFT = endpointAtoms.reduce((total, a) => {
    // The same independently computed reverse probabilities are summed with
    // fixed-point outward rounding; normalization is also certified separately.
    const zLaw = finalLaw[a.z].range;
    if (!a.reachable) return total;
    return plus(total, product(zLaw, product(second.transition[a.z][a.y], first.transition[a.y][a.x])));
  }, point(zero));
  const deltaF = negative(logarithm(div(Zs[2], Zs[0]))), initialKL = relativeEntropy(initial, equilibrium[0]), finalKL = relativeEntropy(second.end, equilibrium[2]);
  const dissipatedWork = minus(work, deltaF), noneqDissipatedWork = minus(work, minus(minus(product(second.end.range, energies[2]), shannon(second.end)), minus(initialU, initialH)));
  return {
    parameters, precisionBits: 256, temperatureK: exact(temperature), thermalEnergyJ: exact(thermalEnergy), boltzmannConstantJPerK: exact(boltzmann), gapInThermalUnits: energies.map(observe), equilibriumOneProbabilities: equilibrium.map(exact), stages,
    holds: [first, second].map(h => ({ stationaryOneProbability: exact(h.pi), decay: observe(h.decay), transition: h.transition.map(row => row.map(observe)), generator: h.generator.map(row => row.map(observe)), heatIntoSystem: observe(h.heat), totalEntropyProduction: observe(h.sigma), exactlyZeroProduction: h.zeroProduction })),
    workAtoms: workAtoms.map(a => ({ initialState: a.x, secondQuenchState: a.y, reachable: a.reachable, probability: observe(a.mass), work: a.reachable ? observe(a.atomWork) : null, inverseWorkRatio: exact(a.inverseWorkRatio), reverseCanonicalProbability: observe(a.reverseMass), forwardToReverseMassRatio: a.forwardReverseRatio ? exact(a.forwardReverseRatio) : null, canonicalCrooksPrediction: a.reachable ? exact(a.crooksPrediction) : null, initialDensityCorrection: exact(a.initialCorrection) })), endpointAtoms,
    meanWork: observe(work), heatIntoSystem: observe(heat), meanWorkVariance: observe(workVariance), totalEntropyProduction: observe(meanSigma), nonequilibriumDissipatedWork: observe(noneqDissipatedWork), canonicalFreeEnergyChange: observe(deltaF), canonicalDissipatedWork: observe(dissipatedWork), initialCanonicalRelativeEntropy: observe(initialKL), finalCanonicalRelativeEntropy: observe(finalKL),
    initialDistributionIsCanonical: equal(initial.rational!, equilibrium[0]), jarzynskiApplicable: equal(initial.rational!, equilibrium[0]), workExponentialMoment: observe(workMoment), jarzynskiIdentity: equal(initial.rational!, equilibrium[0]) ? exact(div(Zs[2], Zs[0])) : null,
    entropyIFTIdentity: exact(one), computedReverseNormalization: observe(computedIFT), firstLawResidual: exact(zero), freeEnergyBalanceResidual: exact(zero),
    classification: { fullSupportInitialLaw: true, specifiedConstantTemperatureBath: true, workOnSystem: true, heatIntoSystem: true, markovianEvenStates: true, sampledPaths: false, normalizedOrTruncatedPathSum: false, endpointBinsIntegrateAllJumpCounts: true, exactlyZeroWork, zeroMeanWorkIdentityApplies: exactlyZeroMeanWork, registeredReader: false },
    approximationContract: 'Two quenches separated/followed by exact two-state Markov holds satisfying q01/q10=1/r and E1-E0=kBT ln(r), constant specified T. Work is into the system, heat is into the system, bath entropy=-Q/(kBT). Probabilities propagate with directed256-bit exponential/log enclosures; exact rational stationary/zero-time laws are preserved. Known nonnegativity only intersects entropy/KL enclosures; no law is renormalized. Four work atoms and eight endpoint atoms integrate all intervening jump counts because heat telescopes during a constant gap. They are not sampled paths, path densities or truncated trajectory sums. IFT normalization and energy identities have separate algebraic certificates; the reverse normalization is independently enclosed. Canonical JE/Crooks require their stated equilibrium initialization. All primary energies are in kBT and entropies in nats, with SI joules from the exact Boltzmann constant and specified T. Unregistered numerical preparation; no worker/UI or research-completion claim.',
  };
}
