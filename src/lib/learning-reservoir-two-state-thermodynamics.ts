/** Unregistered stationary two-state heat conductor with resolved bath labels.
 * A/B are names, not temperature order. Ensemble initialization is always pi;
 * initialState/bathCode/jumpCount select only a deterministic example path.
 */
import { fraction as f, zero, one, add, sub, mul, div, neg, equal, exact, logarithm, observeRange, entropyScale as S, type F } from './learning-entropy-production-math';
type Range = { lower: F; upper: F };
const point = (x: F): Range => ({ lower: x, upper: x });
const plus = (a: Range, b: Range): Range => ({ lower: add(a.lower, b.lower), upper: add(a.upper, b.upper) });
const negative = (a: Range): Range => ({ lower: neg(a.upper), upper: neg(a.lower) });
const minus = (a: Range, b: Range) => plus(a, negative(b));
const scale = (a: Range, x: F): Range => x.n >= BigInt(0) ? { lower: mul(a.lower, x), upper: mul(a.upper, x) } : { lower: mul(a.upper, x), upper: mul(a.lower, x) };
const log = (x: F): Range => { const v = logarithm(x); return { lower: f(v.lower, S), upper: f(v.upper, S) }; };
const observation = (x: Range) => observeRange(x.lower, x.upper);
const sum = (xs: F[]) => xs.reduce(add, zero);
const sumRanges = (xs: Range[]) => xs.reduce(plus, point(zero));
const power = (x: F, exponent: number): F => exponent >= 0 ? f(x.n ** BigInt(exponent), x.d ** BigInt(exponent)) : f(x.d ** BigInt(-exponent), x.n ** BigInt(-exponent));
import { reservoirThermalParameterKeys, isReservoirThermalParameters, type ReservoirThermalParameters } from './learning-thermal-input';
export { reservoirThermalParameterKeys, isReservoirThermalParameters, type ReservoirThermalParameters } from './learning-thermal-input';
export function reservoirTwoStateThermodynamics(input: ReservoirThermalParameters) {
  if (!isReservoirThermalParameters(input)) throw new RangeError('Sixteen owned integer fields; positive odds/clocks/temperature/denominators, beta1..8, time0..4s,1..64 records,0..16 jumps,0..1 example state, no unused bath bits, zero time allows no jumps; magnitudes<=1e9, reference temperature<=1e6K');
  const parameters = Object.freeze(Object.fromEntries(reservoirThermalParameterKeys.map(k => [k, input[k] + 0]))) as unknown as Readonly<ReservoirThermalParameters>;
  const p = parameters, ratio = f(BigInt(p.oddsNumerator), BigInt(p.oddsDenominator)), betas = [p.betaA, p.betaB];
  const clocks = [f(BigInt(p.clockANumerator), BigInt(p.clockADenominator)), f(BigInt(p.clockBNumerator), BigInt(p.clockBDenominator))];
  const rateOdds = betas.map(b => power(ratio, b)), up = rateOdds.map((r, i) => div(clocks[i], add(one, r))), down = rateOdds.map((r, i) => mul(up[i], r));
  const aggregateUp = sum(up), aggregateDown = sum(down), totalClock = add(aggregateUp, aggregateDown), pi = [div(aggregateDown, totalClock), div(aggregateUp, totalClock)];
  const generator = [[neg(aggregateUp), aggregateUp], [aggregateDown, neg(aggregateDown)]];
  const flowsUp = up.map(q => mul(pi[0], q)), flowsDown = down.map(q => mul(pi[1], q)), currents = flowsUp.map((q, i) => sub(q, flowsDown[i]));
  if (sum(currents).n !== BigInt(0)) throw new RangeError('Stationary current conservation failed');
  const referenceTemperature = f(BigInt(p.temperatureNumerator), BigInt(p.temperatureDenominator)), boltzmann = f(BigInt(1380649), BigInt(10) ** BigInt(29)), thermalEnergy = mul(boltzmann, referenceTemperature);
  const logOdds = log(ratio), energy = logOdds, H = negative(sumRanges(pi.map(q => scale(log(q), q))));
  const heatRates = currents.map(j => scale(energy, j)), bathRates = currents.map((j, b) => scale(energy, mul(j, f(BigInt(-betas[b])))));
  const sigma = scale(logOdds, mul(currents[0], f(BigInt(p.betaB - p.betaA))));
  const pairedContributions = currents.map((j, b) => scale(log(div(flowsUp[b], flowsDown[b])), j));
  const bathChoicesUp = up.map(q => div(q, aggregateUp)), bathChoicesDown = down.map(q => div(q, aggregateDown));
  const conditionalJeffreys = sumRanges(bathChoicesUp.map((q, b) => scale(log(div(q, bathChoicesDown[b])), sub(q, bathChoicesDown[b]))));
  const commonAggregateTraffic = mul(pi[0], aggregateUp), conditionalEntropyRate = scale(conditionalJeffreys, commonAggregateTraffic);
  const detailedBalanceChannels = equal(ratio, one) || p.betaA === p.betaB;
  if (!detailedBalanceChannels && (sigma.lower.n <= BigInt(0) || sigma.upper.n <= BigInt(0))) throw new RangeError('Positive physical entropy-rate certificate lost');
  const adjointUp = down.map(q => div(mul(pi[1], q), pi[0])), adjointDown = up.map(q => div(mul(pi[0], q), pi[1]));
  const adjointGenerator = [[neg(sum(adjointUp)), sum(adjointUp)], [sum(adjointDown), neg(sum(adjointDown))]];
  if (!generator.every((row, i) => row.every((q, j) => equal(q, adjointGenerator[i][j])))) throw new RangeError('Reversed-film aggregate generator identity failed');
  const time = f(BigInt(p.timeNumerator), BigInt(p.timeDenominator));
  const history = Array.from({ length: p.sampleSteps + 1 }, (_, step) => {
    const t = mul(time, f(BigInt(step), BigInt(p.sampleSteps)));
    return { step, time: exact(t), meanHeatIntoSystemA: observation(scale(heatRates[0], t)), meanHeatIntoSystemB: observation(scale(heatRates[1], t)), meanHeatIntoSystemAJ: observation(scale(heatRates[0], mul(t, thermalEnergy))), meanHeatIntoSystemBJ: observation(scale(heatRates[1], mul(t, thermalEnergy))), meanBathEntropyA: observation(scale(bathRates[0], t)), meanBathEntropyB: observation(scale(bathRates[1], t)), meanTotalEntropyProduction: observation(scale(sigma, t)), meanSystemEntropyChange: exact(zero), stationaryShannonEntropy: observation(H), meanEnergyChange: exact(zero), meanWork: exact(zero), aggregateObservedEntropy: exact(zero), firstLawResidual: exact(zero) };
  });
  const states = [p.initialState], bathIndices: number[] = [];
  let heatCountA = 0, heatCountB = 0, rateRatio = one, forwardWord = one, reverseWord = one;
  const jumps = Array.from({ length: p.jumpCount }, (_, index) => {
    const from = states[index], to = 1 - from, bath = (p.bathCode >> index) & 1, delta = to - from;
    states.push(to); bathIndices.push(bath); if (bath === 0) heatCountA += delta; else heatCountB += delta;
    const rate = from === 0 ? up[bath] : down[bath], reversedRate = from === 0 ? down[bath] : up[bath], aggregateRate = from === 0 ? aggregateUp : aggregateDown, aggregateReversedRate = from === 0 ? aggregateDown : aggregateUp;
    const ratioForJump = div(rate, reversedRate), fluxRatio = div(mul(pi[from], rate), mul(pi[to], reversedRate));
    rateRatio = mul(rateRatio, ratioForJump); forwardWord = mul(forwardWord, div(rate, aggregateRate)); reverseWord = mul(reverseWord, div(reversedRate, aggregateReversedRate));
    return { index, from, to, bath: bath === 0 ? 'A' as const : 'B' as const, rate: exact(rate), reverseRate: exact(reversedRate), heatIntoSystem: observation(scale(energy, f(BigInt(delta)))), bathEntropy: observation(scale(energy, f(BigInt(-betas[bath] * delta)))), totalEntropyIncrement: observation(log(fluxRatio)), aggregateObservedEntropyIncrement: exact(zero), cumulativeHeatIntoSystemA: observation(scale(energy, f(BigInt(heatCountA)))), cumulativeHeatIntoSystemB: observation(scale(energy, f(BigInt(heatCountB)))), cumulativeTotalEntropy: observation(log(mul(div(pi[states[0]], pi[to]), rateRatio))) };
  });
  const boundaryRatio = div(pi[states[0]], pi[states.at(-1)!]), totalRatio = mul(boundaryRatio, rateRatio), channelWordRatio = div(forwardWord, reverseWord);
  if (!equal(totalRatio, channelWordRatio) || heatCountA + heatCountB !== states.at(-1)! - states[0]) throw new RangeError('Marked-path likelihood/energy telescope failed');
  const dwell = div(time, f(BigInt(p.jumpCount + 1))), exits = [aggregateUp, aggregateDown], survivalAction = mul(dwell, sum(states.map(i => exits[i])));
  const forwardLog = minus(plus(log(pi[states[0]]), sumRanges(bathIndices.map((bath, i) => log(states[i] === 0 ? up[bath] : down[bath])))), point(survivalAction));
  const reverseLog = minus(plus(log(pi[states.at(-1)!]), sumRanges(bathIndices.map((bath, i) => log(states[i] === 0 ? down[bath] : up[bath])))), point(survivalAction));
  const heatA = scale(energy, f(BigInt(heatCountA))), heatB = scale(energy, f(BigInt(heatCountB))), total = log(totalRatio);
  const bathEntropyRatio = power(ratio, -p.betaA * heatCountA - p.betaB * heatCountB);
  return {
    parameters, precisionBits: 256, referenceTemperatureK: exact(referenceTemperature), reservoirTemperaturesK: betas.map(b => exact(div(referenceTemperature, f(BigInt(b))))), boltzmannConstantJPerK: exact(boltzmann), referenceThermalEnergyJ: exact(thermalEnergy), gapInReferenceThermalUnits: observation(energy), gapJ: observation(scale(energy, thermalEnergy)),
    generator: generator.map(row => row.map(exact)), reversedFilmAggregateGenerator: adjointGenerator.map(row => row.map(exact)), stationaryLaw: pi.map(exact), stationaryShannonEntropy: observation(H), stationaryEnergyInReferenceThermalUnits: observation(scale(energy, pi[1])),
    channels: ['A', 'B'].map((name, b) => ({ name, beta: betas[b], temperatureK: exact(div(referenceTemperature, f(BigInt(betas[b])))), clock: exact(clocks[b]), equilibriumOneProbabilityAlone: exact(div(one, add(one, rateOdds[b]))), upRate: exact(up[b]), downRate: exact(down[b]), forwardTraffic: exact(flowsUp[b]), reverseTraffic: exact(flowsDown[b]), current: exact(currents[b]), heatIntoSystemRate: observation(heatRates[b]), heatIntoSystemRateJPerSecond: observation(scale(heatRates[b], thermalEnergy)), bathEntropyRate: observation(bathRates[b]), fullEntropyContributionRate: observation(pairedContributions[b]), reversedFilmUpRate: exact(adjointUp[b]), reversedFilmDownRate: exact(adjointDown[b]) })),
    aggregateStationaryCurrent: exact(sum(currents)), aggregateObservedEntropyRate: exact(zero), physicalEntropyRate: observation(sigma), pairedPhysicalEntropyRate: observation(sumRanges(pairedContributions)), commonAggregateTraffic: exact(commonAggregateTraffic), conditionalBathChoicesUp: bathChoicesUp.map(exact), conditionalBathChoicesDown: bathChoicesDown.map(exact), conditionalJeffreysDivergence: observation(conditionalJeffreys), conditionalLabelEntropyRate: observation(conditionalEntropyRate), energyRateResidual: exact(zero), stationaryShannonEntropyRate: exact(zero), entropyIFTIdentity: exact(one), entropyIFTMethod: 'analytic-normalization-of-stationary-marked-time-reversal' as const,
    history, current: history.at(-1)!,
    path: { states, reversedStates: [...states].reverse(), baths: bathIndices.map(b => b === 0 ? 'A' : 'B'), reversedBaths: [...bathIndices].reverse().map(b => b === 0 ? 'A' : 'B'), jumps, jumpTimes: jumps.map((_, i) => exact(mul(dwell, f(BigInt(i + 1))))), equalDwellTime: exact(dwell), survivalAction: exact(survivalAction), systemBoundaryEntropy: observation(log(boundaryRatio)), bathEntropyA: observation(scale(energy, f(BigInt(-p.betaA * heatCountA)))), bathEntropyB: observation(scale(energy, f(BigInt(-p.betaB * heatCountB)))), bathEntropy: observation(log(bathEntropyRatio)), heatIntoSystemA: observation(heatA), heatIntoSystemB: observation(heatB), heatIntoSystemAJ: observation(scale(heatA, thermalEnergy)), heatIntoSystemBJ: observation(scale(heatB, thermalEnergy)), energyChange: observation(scale(energy, f(BigInt(states.at(-1)! - states[0])))), work: exact(zero), firstLawResidual: exact(zero), totalEntropy: observation(total), reversedTotalEntropy: observation(negative(total)), totalRatio: exact(totalRatio), aggregateObservedEntropy: exact(zero), conditionalBathWordProbability: exact(forwardWord), reversedConditionalBathWordProbability: exact(reverseWord), conditionalBathWordRatio: exact(channelWordRatio), logForwardDensityInSeconds: observation(forwardLog), logReversedDensityInSeconds: observation(reverseLog), independentLogDensityDifference: observation(minus(forwardLog, reverseLog)), densityReference: 'Lebesgue measure on ordered jump times in seconds and counting measure on bath labels; density unit s^-jumpCount, not exact-path probability. Survival uses both channels together.' },
    classification: { stationaryInitialization: true, evenTimeReversalStates: true, bothChannelsPositive: true, aggregateDetailedBalance: true, channelDetailedBalance: detailedBalanceChannels, physicalEntropyRateExactSign: detailedBalanceChannels ? 0 : 1, zeroEnergyGap: equal(ratio, one), hotterChannel: p.betaA === p.betaB ? 'equal-temperatures' : p.betaA < p.betaB ? 'A' : 'B', heatFlowDirection: detailedBalanceChannels ? 'none' : p.betaA < p.betaB ? 'A-to-B' : 'B-to-A', sampledPaths: false, registeredReader: false },
    approximationContract: 'Stationary autonomous two-state marked CTMC, even states, two positive channels. Energy gap=kBTref ln(r), Tb=Tref/beta_b with integer beta1..8; qb01/qb10=r^-beta_b imposes local detailed balance. All rates, stationary weights, currents and path ratios are exact rational. Directed256-bit logarithm endpoints are scaled/summed as exact rationals, without final fixed-quantum rounding; tiny positive heat/entropy is not replaced by zero. Ensemble laws always start at pi; initialState and bath bits choose only a deterministic equal-dwell path. Bath entropy can be negative individually, while total production is nonnegative. State-only stationary paths have zero observed entropy because their aggregate binary process is reversible; resolved bath words retain heat conduction. Marked time reversal changes channel allocations while leaving the aggregate generator unchanged; it is not a swap of bath names. IFT=1 is an analytic normalization identity, not a finite example-path sum or empirical test; no general tilted moment is computed. No rates, path laws or probabilities are renormalized. Unregistered numerical preparation; no UI or Worker verification yet.',
  };
}
