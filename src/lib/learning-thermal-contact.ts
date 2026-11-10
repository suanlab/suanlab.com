/** Lumped, insulated, fixed-volume bodies: C in J/K, conductance in W/K, time in s.
 * Exact binary input products precede projection. Transcendental functions and
 * the positive entropy integral use ordinary floating point, not exact arithmetic.
 */
import {
  binaryProduct, binarySum, binaryNegate, binaryMultiply, binaryRatioObservation,
  type BinaryValue, type BinaryObservation,
} from './learning-binary-products';

export type ThermalObservation = BinaryObservation;
export interface ThermalContactParameters {
  capacity1: number;
  capacity2: number;
  initialTemperature1: number;
  initialTemperature2: number;
  conductance: number;
}
const product = binaryProduct;
const one = product(1);
const observe = (numerator: BinaryValue, denominator: BinaryValue = one) => binaryRatioObservation(numerator, denominator);
const negative = (v: ThermalObservation): ThermalObservation => ({ value: v.value === null ? null : v.value === 0 ? 0 : -v.value, sign: v.sign === 0 ? 0 : -v.sign, logMagnitude: v.logMagnitude });
function bound(value: number, low: number, high: number, name: string) {
  if (!Number.isFinite(value) || value < low || value > high) throw new RangeError(`Invalid ${name}`);
}
function validate(p: ThermalContactParameters) {
  bound(p.capacity1, .5, 4, 'capacity1');
  bound(p.capacity2, .5, 4, 'capacity2');
  bound(p.initialTemperature1, 250, 350, 'initialTemperature1');
  bound(p.initialTemperature2, 250, 350, 'initialTemperature2');
  bound(p.conductance, 0, 2, 'conductance');
}
function exponent(p: ThermalContactParameters, time: number) {
  return observe(binaryMultiply(product(p.conductance, time), binarySum([product(p.capacity1), product(p.capacity2)])), product(p.capacity1, p.capacity2));
}
/** (1-exp(-z))/z, including its removable limit. z is in [0,64]. */
function transferGain(z: number) {
  return z < 1e-5 ? 1 + z * (-.5 + z * (1 / 6 + z * (-1 / 24 + z / 120))) : -Math.expm1(-z) / z;
}
function logGain(z: number) {
  return Math.abs(z) < 1e-5 ? 1 + z * (-.5 + z * (1 / 3 + z * (-.25 + z / 5))) : Math.log1p(z) / z;
}

// Generate positive Gauss-Legendre weights on [0,1] from the Legendre recurrence.
// With the declared temperature bounds the integrand has no nearby singularity.
const quadrature = (() => {
  const nodes: { x: number; weight: number }[] = [];
  const n = 16;
  for (let i = 0; i < n / 2; i++) {
    let x = Math.cos(Math.PI * (i + .75) / (n + .5));
    let derivative = 0;
    for (let iteration = 0; iteration < 32; iteration++) {
      let a = 1, b = x;
      for (let j = 2; j <= n; j++) { const next = ((2 * j - 1) * x * b - (j - 1) * a) / j; a = b; b = next; }
      derivative = n * (x * b - a) / (x * x - 1);
      const step = b / derivative;
      x -= step;
      if (Math.abs(step) <= Number.EPSILON) break;
    }
    // Recompute the derivative at the converged root, not the previous iterate.
    let a = 1, b = x;
    for (let j = 2; j <= n; j++) { const next = ((2 * j - 1) * x * b - (j - 1) * a) / j; a = b; b = next; }
    derivative = n * (x * b - a) / (x * x - 1);
    const weight = 1 / ((1 - x * x) * derivative * derivative);
    nodes.push({ x: (1 - x) / 2, weight }, { x: (1 + x) / 2, weight });
  }
  return nodes;
})();
function entropyIntegral(t1: number, t2: number, difference: number, fraction: number, p: ThermalContactParameters) {
  const sum = p.capacity1 + p.capacity2;
  return quadrature.reduce((total, node) => {
    const u = fraction * node.x;
    const a = t1 - p.capacity2 / sum * difference * u;
    const b = t2 + p.capacity1 / sum * difference * u;
    return total + node.weight * (1 - u) / (a * b);
  }, 0);
}
function interval(p: ThermalContactParameters, start: number, end: number) {
  const duration = end - start;
  const initialDifference = p.initialTemperature1 - p.initialTemperature2;
  const startExponent = exponent(p, start).value ?? 0;
  const z = exponent(p, duration).value ?? 0;
  const decay = Math.exp(-startExponent);
  const gain = transferGain(z);
  const startHeat = product(p.conductance, start, initialDifference, transferGain(startExponent));
  const startTemperature = (initial: number, capacity: number, sign: number) => observe(binarySum([
    product(initial, capacity), sign === 1 ? startHeat : binaryNegate(startHeat),
  ]), product(capacity)).value!;
  const temperature1 = startTemperature(p.initialTemperature1, p.capacity1, -1);
  const temperature2 = startTemperature(p.initialTemperature2, p.capacity2, 1);
  const difference = initialDifference * decay;
  const heatRaw = product(p.conductance, duration, initialDifference, decay, gain);
  const heat = observe(heatRaw);
  const fraction = -Math.expm1(-z);
  const entropyRaw = product(p.conductance, duration, initialDifference, initialDifference, decay, decay, gain,
    entropyIntegral(temperature1, temperature2, difference, fraction, p));
  const ratio1 = observe(binaryNegate(heatRaw), product(p.capacity1, temperature1)).value ?? 0;
  const ratio2 = observe(heatRaw, product(p.capacity2, temperature2)).value ?? 0;
  const entropy1 = observe(binaryMultiply(binaryNegate(heatRaw), product(logGain(ratio1))), product(temperature1));
  const entropy2 = observe(binaryMultiply(heatRaw, product(logGain(ratio2))), product(temperature2));
  return {
    start, end, duration, heat, energyChange1: negative(heat), energyChange2: heat,
    temperatureChange1: observe(binaryNegate(heatRaw), product(p.capacity1)),
    temperatureChange2: observe(heatRaw, product(p.capacity2)),
    entropyChange1: entropy1, entropyChange2: entropy2, entropyChange: observe(entropyRaw),
    roundedEntropySum: entropy1.value === null || entropy2.value === null ? null : entropy1.value + entropy2.value,
  };
}

export function thermalContactWindow(p: ThermalContactParameters, start: number, end: number) {
  validate(p); bound(start, 0, 8, 'start'); bound(end, 0, 8, 'end');
  if (end < start) throw new RangeError('Window end precedes start');
  return interval(p, start, end);
}

export function thermalContactState(p: ThermalContactParameters, time: number) {
  validate(p); bound(time, 0, 8, 'time');
  const history = interval(p, 0, time);
  const d = p.initialTemperature1 - p.initialTemperature2;
  const z = exponent(p, time).value ?? 0;
  const decay = Math.exp(-z);
  const heatRaw = product(p.conductance, time, d, transferGain(z));
  const temperature1 = observe(binarySum([product(p.initialTemperature1, p.capacity1), binaryNegate(heatRaw)]), product(p.capacity1)).value!;
  const temperature2 = observe(binarySum([product(p.initialTemperature2, p.capacity2), heatRaw]), product(p.capacity2)).value!;
  const capacitySum = binarySum([product(p.capacity1), product(p.capacity2)]);
  return {
    time, temperature1, temperature2,
    equilibriumTemperature: observe(binarySum([product(p.capacity1, p.initialTemperature1), product(p.capacity2, p.initialTemperature2)]), capacitySum).value!,
    connected: p.conductance > 0,
    relaxationTime: p.conductance === 0 ? null : observe(product(p.capacity1, p.capacity2), binaryMultiply(product(p.conductance), capacitySum)),
    relaxationRate: observe(binaryMultiply(product(p.conductance), capacitySum), product(p.capacity1, p.capacity2)),
    temperatureDifference: observe(product(d, decay)),
    heatRate: observe(product(p.conductance, d, decay)),
    entropyProductionRate: observe(product(p.conductance, d, d, decay, decay), product(temperature1, temperature2)),
    ...history,
  };
}
