/** Reversible monatomic ideal gas with two constant-temperature reservoirs.
 * Work is by the gas; heat is into it. Energies are per cycle, not powers.
 * T on each adiabat is linear in leg progress; isothermal log(V) is linear.
 */
import { binaryProduct as product, binarySum as sum, binaryNegate as negative,
  binaryMultiply as multiply, binaryRatioObservation as observe,
  type BinaryValue, type BinaryObservation } from './learning-binary-products';
import { idealGasConstant as R } from './learning-ideal-gas-paths';

export interface CarnotParameters {
  moles: number;
  hotTemperature: number;
  coldTemperature: number;
  initialVolume: number;
  expansionRatio: number;
}
export type CarnotDirection = 'forward' | 'reverse';
const one = product(1);
const difference = (a: BinaryValue, b: BinaryValue) => sum([a, negative(b)]);
const zero = () => observe(product(0), one);
function bound(v: number, low: number, high: number, name: string) {
  if (!Number.isFinite(v) || v < low || v > high) throw new RangeError(`Invalid ${name}`);
}
function validate(p: CarnotParameters, direction: CarnotDirection) {
  bound(p.moles, .25, 2, 'amount');
  bound(p.hotTemperature, 250, 800, 'hot temperature'); bound(p.coldTemperature, 250, 800, 'cold temperature');
  bound(p.initialVolume, .005, .04, 'initial volume'); bound(p.expansionRatio, 1, 4, 'expansion ratio');
  if (p.hotTemperature < p.coldTemperature) throw new RangeError('Hot reservoir is colder than cold reservoir');
  if (direction !== 'forward' && direction !== 'reverse') throw new RangeError('Invalid cycle direction');
}
function logRatio(n: BinaryValue, d: BinaryValue) {
  const delta = difference(n, d), total = sum([n, d]), a = observe(delta, total).value ?? 0;
  const square = a * a;
  const gain = Math.abs(a) < 1e-5 ? 1 + square * (1 / 3 + square * (1 / 5 + square / 7)) : Math.atanh(a) / a;
  return { numerator: multiply(delta, product(2, gain)), denominator: total };
}
function expGain(v: number) {
  return Math.abs(v) < 1e-5 ? 1 + v * (.5 + v * (1 / 6 + v * (1 / 24 + v / 120))) : Math.expm1(v) / v;
}
function state(p: CarnotParameters, direction: CarnotDirection, progress: number) {
  const index = Math.min(3, Math.floor(4 * progress)), u = 4 * progress - index;
  const hot = p.hotTemperature, cold = p.coldTemperature, gap = difference(product(hot), product(cold));
  const forward = direction === 'forward', sign = forward ? 1 : -1;
  const isothermal = forward ? index % 2 === 0 : index % 2 === 1;
  let temperature: BinaryValue, hotFraction: number, coldFraction: number, volumeExpansionFraction: number;
  if (forward) {
    temperature = index === 0 ? product(hot) : index === 1 ? sum([product(hot), negative(multiply(gap, product(u)))]) : index === 2 ? product(cold) : sum([product(cold), multiply(gap, product(u))]);
    hotFraction = index === 0 ? u : 1;
    coldFraction = index < 2 ? 0 : index === 2 ? u : 1;
    volumeExpansionFraction = index === 0 ? u : index === 1 ? 1 : index === 2 ? 1 - u : 0;
  } else {
    // Evaluate reverse legs directly: 1-progress would round to1 for tiny progress.
    temperature = index === 0 ? sum([product(hot), negative(multiply(gap, product(u)))]) : index === 1 ? product(cold) : index === 2 ? sum([product(cold), multiply(gap, product(u))]) : product(hot);
    hotFraction = index === 3 ? u : 0;
    coldFraction = index === 0 ? 0 : index === 1 ? u : 1;
    volumeExpansionFraction = index === 0 ? 0 : index === 1 ? u : index === 2 ? 1 : 1 - u;
  }
  const temperatureChange = difference(temperature, product(hot));
  const logExpansion = Math.log1p(p.expansionRatio - 1);
  const logThermal = logRatio(product(hot), temperature);
  const logVolumeNumerator = sum([multiply(logThermal.numerator, product(1.5)), multiply(logThermal.denominator, product(logExpansion, volumeExpansionFraction))]);
  const logVolumeRatio = observe(logVolumeNumerator, logThermal.denominator);
  const z = logVolumeRatio.value ?? 0;
  const volumeChangeNumerator = multiply(logVolumeNumerator, product(p.initialVolume, expGain(z)));
  const volumeRaw = sum([multiply(product(p.initialVolume), logThermal.denominator), volumeChangeNumerator]);
  const volume = observe(volumeRaw, logThermal.denominator).value!;
  const energyRaw = multiply(temperatureChange, product(p.moles, R, 1.5));
  const heatHotRaw = product(p.moles, R, logExpansion, sign, hot, hotFraction);
  const heatColdRaw = product(p.moles, R, logExpansion, -sign, cold, coldFraction);
  // Form the full rational temperature bracket before multiplying by log(r).
  // This retains tiny endpoint work and exact reverse heat/work crossings.
  const bracket = difference(product(hot, hotFraction), product(cold, coldFraction));
  const netHeatRaw = multiply(bracket, product(p.moles, R, logExpansion, sign));
  const workRaw = difference(netHeatRaw, energyRaw);
  const entropyFraction = difference(product(hotFraction), product(coldFraction));
  const entropyGasRaw = multiply(entropyFraction, product(p.moles, R, logExpansion, sign));
  return {
    direction, progress, legIndex: index, legProgress: u,
    stage: isothermal ? 'isothermal' as const : 'adiabatic' as const,
    temperature: observe(temperature, one).value!, volume,
    pressure: observe(multiply(multiply(temperature, product(p.moles, R)), logThermal.denominator), volumeRaw).value!,
    temperatureChange: observe(temperatureChange, one),
    volumeChange: observe(volumeChangeNumerator, logThermal.denominator), logVolumeRatio,
    energyChange: observe(energyRaw, one), heatHotIntoGas: observe(heatHotRaw, one), heatColdIntoGas: observe(heatColdRaw, one),
    netHeat: observe(netHeatRaw, one), workByGas: observe(workRaw, one), entropyChangeGas: observe(entropyGasRaw, one),
    hotReservoirEntropyChange: observe(product(-sign, p.moles, R, logExpansion, hotFraction), one),
    coldReservoirEntropyChange: observe(product(sign, p.moles, R, logExpansion, coldFraction), one),
    entropyProduction: 0,
  };
}
export function carnotCycleState(p: CarnotParameters, direction: CarnotDirection, progress: number) {
  validate(p, direction); bound(progress, 0, 1, 'progress');
  return state(p, direction, progress);
}
export function carnotCycle(p: CarnotParameters, direction: CarnotDirection) {
  validate(p, direction);
  const gap = difference(product(p.hotTemperature), product(p.coldTemperature)), logExpansion = Math.log1p(p.expansionRatio - 1);
  const active = p.hotTemperature > p.coldTemperature && p.expansionRatio > 1;
  const hot = product(p.moles, R, p.hotTemperature, logExpansion), cold = product(-p.moles, R, p.coldTemperature, logExpansion);
  const adiabat = multiply(product(p.moles, R, 1.5), gap), entropy = product(p.moles, R, logExpansion);
  const forwardLegs = [
    { heat: hot, work: hot, energy: product(0), entropy },
    { heat: product(0), work: adiabat, energy: negative(adiabat), entropy: product(0) },
    { heat: cold, work: cold, energy: product(0), entropy: negative(entropy) },
    { heat: product(0), work: negative(adiabat), energy: adiabat, entropy: product(0) },
  ];
  const legs = (direction === 'forward' ? forwardLegs : [...forwardLegs].reverse().map(leg => ({ heat:negative(leg.heat), work:negative(leg.work), energy:negative(leg.energy), entropy:negative(leg.entropy) })))
    .map((leg, index) => ({ index, heat:observe(leg.heat, one), work:observe(leg.work, one), energyChange:observe(leg.energy, one), entropyChange:observe(leg.entropy, one) }));
  return {
    direction, regime: active ? direction === 'forward' ? 'heat-engine' as const : 'refrigerator' as const : 'degenerate' as const,
    complete: state(p, direction, 1), corners:[0, .25, .5, .75].map(s => state(p, direction, s)), legs,
    carnotTemperatureBound: observe(gap, product(p.hotTemperature)),
    workPerHotHeat: p.expansionRatio > 1 ? observe(gap, product(p.hotTemperature)) : null,
    engineEfficiency: active && direction === 'forward' ? observe(gap, product(p.hotTemperature)) : null,
    refrigeratorCOP: active && direction === 'reverse' ? observe(product(p.coldTemperature), gap) : null,
    heatingCOP: active && direction === 'reverse' ? observe(product(p.hotTemperature), gap) : null,
    totalEnergyChange: zero(), totalEntropyChange: zero(),
  };
}
export type CarnotObservation = BinaryObservation;
