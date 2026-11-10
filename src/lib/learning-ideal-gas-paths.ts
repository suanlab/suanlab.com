/** Closed monatomic ideal gas; frictionless quasistatic volume control and
 * reversible heat exchange. Progress is path position, not physical time.
 * Linear T/V interpolation retains exact supplied binary inputs internally.
 */
import { binaryProduct as product, binarySum as sum, binaryNegate as negate, binaryMultiply as multiply,
  binaryRatioObservation as observe, type BinaryValue, type BinaryObservation } from './learning-binary-products';

export const idealGasConstant = 8.31446261815324; // binary64 approximation, J/(mol K)
export interface IdealGasPathParameters {
  moles: number;
  initialTemperature: number;
  finalTemperature: number;
  initialVolume: number;
  finalVolume: number;
}
export type IdealGasPath = 'A' | 'B';
const one = product(1);
const difference = (a: BinaryValue, b: BinaryValue) => sum([a, negate(b)]);
const power = (a: BinaryValue, n: number) => { let result = one; for (let i = 0; i < n; i++) result = multiply(result, a); return result; };
function bound(value: number, low: number, high: number, name: string) {
  if (!Number.isFinite(value) || value < low || value > high) throw new RangeError(`Invalid ${name}`);
}
function validate(p: IdealGasPathParameters) {
  bound(p.moles, .25, 2, 'moles');
  bound(p.initialTemperature, 250, 350, 'initialTemperature'); bound(p.finalTemperature, 250, 350, 'finalTemperature');
  bound(p.initialVolume, .005, .04, 'initialVolume'); bound(p.finalVolume, .005, .04, 'finalVolume');
}
interface Fraction { numerator: BinaryValue; denominator: BinaryValue }
/** log(N/D) = 2*a*atanh(a)/a; its sign and structural zero follow exact N-D. */
function logarithm(numerator: BinaryValue, denominator: BinaryValue): Fraction {
  const delta = difference(numerator, denominator), total = sum([numerator, denominator]);
  const a = observe(delta, total).value ?? 0;
  const square = a * a;
  const gain = Math.abs(a) < 1e-5 ? 1 + square * (1 / 3 + square * (1 / 5 + square / 7)) : Math.atanh(a) / a;
  return { numerator: multiply(delta, product(2, gain)), denominator: total };
}
const quantity = (f: Fraction, ...scale: number[]) => observe(multiply(f.numerator, product(...scale)), f.denominator);

// Outward integer bounds on logarithms, used for cancellation in Q/(nR).
// Reducing N/D into [1,2) leaves a in [0,1/3]. The positive atanh series
// has a bounded geometric tail. Integer floor/ceil preserves enclosure.
const ceilDiv = (a: bigint, b: bigint) => (a + b - BigInt(1)) / b;
const logTwoCache = new Map<number, [bigint, bigint]>();
function atanhBounds(numerator: bigint, denominator: bigint, bits: number): [bigint, bigint] {
  if (numerator === BigInt(0)) return [BigInt(0), BigInt(0)];
  const scale = BigInt(1) << BigInt(bits);
  const low = numerator * scale / denominator, high = ceilDiv(numerator * scale, denominator);
  const squareLow = low * low / scale, squareHigh = ceilDiv(high * high, scale);
  let termLow = low, termHigh = high, totalLow = BigInt(0), totalHigh = BigInt(0);
  const terms = Math.ceil((bits + 16) / Math.log2(9));
  for (let i = 0; i < terms; i++) {
    const odd = BigInt(2 * i + 1);
    totalLow += termLow / odd; totalHigh += ceilDiv(termHigh, odd);
    termLow = termLow * squareLow / scale; termHigh = ceilDiv(termHigh * squareHigh, scale);
  }
  // 2*a^(2M+1)/((2M+1)*(1-a*a)) < 2^-bits for a<=1/3.
  return [BigInt(2) * totalLow, BigInt(2) * totalHigh + BigInt(1)];
}
function logarithmBounds(n: BinaryValue, d: BinaryValue, bits: number): [bigint, bigint] {
  const exp = n.exponent - d.exponent;
  let numerator = exp >= 0 ? n.coefficient << BigInt(exp) : n.coefficient;
  let denominator = exp < 0 ? d.coefficient << BigInt(-exp) : d.coefficient;
  const sign = numerator < denominator ? -1 : 1;
  if (sign < 0) [numerator, denominator] = [denominator, numerator];
  let shift = numerator.toString(2).length - denominator.toString(2).length;
  if (numerator < (denominator << BigInt(shift))) shift--;
  const reduced = denominator << BigInt(shift);
  const remainder = atanhBounds(numerator - reduced, numerator + reduced, bits);
  let ln2 = logTwoCache.get(bits);
  if (!ln2) { ln2 = atanhBounds(BigInt(1), BigInt(3), bits); logTwoCache.set(bits, ln2); }
  const low = BigInt(shift) * ln2[0] + remainder[0], high = BigInt(shift) * ln2[1] + remainder[1];
  return sign < 0 ? [-high, -low] : [low, high];
}
interface HeatResult { heat: BinaryObservation | null; heatPrecisionBits: number | null }
function heatQuantity(p: IdealGasPathParameters, deltaTemperature: BinaryValue, isothermalTemperature: number, volume: BinaryValue, logVolume: Fraction): HeatResult {
  const sensible = multiply(deltaTemperature, product(1.5));
  const workNumerator = multiply(logVolume.numerator, product(isothermalTemperature));
  const approximate = sum([multiply(sensible, logVolume.denominator), workNumerator]);
  if (sensible.coefficient === BigInt(0) || workNumerator.coefficient === BigInt(0) ||
      (sensible.coefficient > BigInt(0)) === (workNumerator.coefficient > BigInt(0))) {
    return { heat: observe(multiply(approximate, product(p.moles, idealGasConstant)), logVolume.denominator), heatPrecisionBits: null };
  }
  for (const bits of [128, 256, 512, 1024, 2048]) {
    const [lo, hi] = logarithmBounds(volume, product(p.initialVolume), bits);
    const lower = sum([sensible, multiply(product(isothermalTemperature), { coefficient: lo, exponent: -bits })]);
    const upper = sum([sensible, multiply(product(isothermalTemperature), { coefficient: hi, exponent: -bits })]);
    if (lower.coefficient === BigInt(0) || upper.coefficient === BigInt(0) ||
        (lower.coefficient > BigInt(0)) !== (upper.coefficient > BigInt(0))) continue;
    const midpoint = multiply(sum([lower, upper]), product(.5));
    const relativeWidth = observe(difference(upper, lower), midpoint).value;
    if (relativeWidth !== null && Math.abs(relativeWidth) > 1e-14) continue;
    return { heat: observe(multiply(midpoint, product(p.moles, idealGasConstant)), one), heatPrecisionBits: bits };
  }
  // Unresolved sign/accuracy is distinct from an exact zero or a range-limited value.
  return { heat: null, heatPrecisionBits: 2048 };
}

export function idealGasPathState(p: IdealGasPathParameters, path: IdealGasPath, progress: number) {
  validate(p); bound(progress, 0, 1, 'progress');
  if (path !== 'A' && path !== 'B') throw new RangeError('Invalid path');
  const first = Math.min(2 * progress, 1), second = Math.max(2 * progress - 1, 0);
  const temperatureProgress = path === 'A' ? first : second, volumeProgress = path === 'A' ? second : first;
  const deltaTemperature = multiply(difference(product(p.finalTemperature), product(p.initialTemperature)), product(temperatureProgress));
  const deltaVolume = multiply(difference(product(p.finalVolume), product(p.initialVolume)), product(volumeProgress));
  const temperature = sum([product(p.initialTemperature), deltaTemperature]);
  const volume = sum([product(p.initialVolume), deltaVolume]);
  const logVolume = logarithm(volume, product(p.initialVolume));
  const isothermalTemperature = path === 'A' ? p.finalTemperature : p.initialTemperature;
  const heat = heatQuantity(p, deltaTemperature, isothermalTemperature, volume, logVolume);
  // Combine the state entropy into one log of a positive rational. This avoids
  // cancellation between separate temperature and volume logarithms.
  const entropyNumerator = multiply(power(temperature, 3), power(volume, 2));
  const entropyDenominator = product(p.initialTemperature, p.initialTemperature, p.initialTemperature, p.initialVolume, p.initialVolume);
  return {
    path, progress, temperatureProgress, volumeProgress,
    stage: (path === 'A' ? progress <= .5 : progress > .5) ? 'isochoric' as const : 'isothermal' as const,
    temperature: observe(temperature, one).value!, volume: observe(volume, one).value!,
    pressure: observe(multiply(temperature, product(p.moles, idealGasConstant)), volume).value!,
    temperatureChange: observe(deltaTemperature, one), volumeChange: observe(deltaVolume, one),
    energyChange: observe(multiply(deltaTemperature, product(p.moles, idealGasConstant, 1.5)), one),
    work: quantity(logVolume, p.moles, idealGasConstant, isothermalTemperature),
    entropyChange: quantity(logarithm(entropyNumerator, entropyDenominator), p.moles, idealGasConstant, .5),
    entropyProduction: 0,
    ...heat,
    heatStatus: heat.heat === null ? 'unresolved' as const : 'resolved' as const,
  };
}

export function idealGasPathComparison(p: IdealGasPathParameters) {
  const a = idealGasPathState(p, 'A', 1), b = idealGasPathState(p, 'B', 1);
  const logVolume = logarithm(product(p.finalVolume), product(p.initialVolume));
  const workDifference = observe(multiply(logVolume.numerator,
    product(p.moles, idealGasConstant, p.finalTemperature - p.initialTemperature)), logVolume.denominator);
  return { a, b, workDifference, cycleHeat: workDifference };
}
