/** Reader-safe thermal preparation contracts. No numerical engine or browser imports. */
export interface DrivenThermalParameters {
  r0Numerator: number; r0Denominator: number; r1Numerator: number; r1Denominator: number; r2Numerator: number; r2Denominator: number;
  clock1Numerator: number; clock1Denominator: number; clock2Numerator: number; clock2Denominator: number;
  time1Numerator: number; time1Denominator: number; time2Numerator: number; time2Denominator: number;
  initialOneNumerator: number; initialDenominator: number; temperatureNumerator: number; temperatureDenominator: number;
}
export const drivenThermalParameterKeys = ['r0Numerator', 'r0Denominator', 'r1Numerator', 'r1Denominator', 'r2Numerator', 'r2Denominator', 'clock1Numerator', 'clock1Denominator', 'clock2Numerator', 'clock2Denominator', 'time1Numerator', 'time1Denominator', 'time2Numerator', 'time2Denominator', 'initialOneNumerator', 'initialDenominator', 'temperatureNumerator', 'temperatureDenominator'] as const;
export function isDrivenThermalParameters(value: unknown): value is DrivenThermalParameters {
  if (!value || typeof value !== 'object' || Object.getPrototypeOf(value) !== Object.prototype || Reflect.ownKeys(value).length !== drivenThermalParameterKeys.length) return false;
  const p = value as Record<string, number>;
  return drivenThermalParameterKeys.every(k => Object.prototype.hasOwnProperty.call(p, k) && Number.isSafeInteger(p[k]) && p[k] >= (k === 'time1Numerator' || k === 'time2Numerator' ? 0 : 1) && p[k] <= 1000000000)
    && p.initialOneNumerator < p.initialDenominator && p.time1Numerator <= 4 * p.time1Denominator && p.time2Numerator <= 4 * p.time2Denominator
    && p.temperatureNumerator <= 1000000 * p.temperatureDenominator;
}

export interface ReservoirThermalParameters {
  oddsNumerator: number; oddsDenominator: number; betaA: number; betaB: number;
  clockANumerator: number; clockADenominator: number; clockBNumerator: number; clockBDenominator: number;
  temperatureNumerator: number; temperatureDenominator: number; timeNumerator: number; timeDenominator: number;
  sampleSteps: number; initialState: number; bathCode: number; jumpCount: number;
}
export const reservoirThermalParameterKeys = ['oddsNumerator', 'oddsDenominator', 'betaA', 'betaB', 'clockANumerator', 'clockADenominator', 'clockBNumerator', 'clockBDenominator', 'temperatureNumerator', 'temperatureDenominator', 'timeNumerator', 'timeDenominator', 'sampleSteps', 'initialState', 'bathCode', 'jumpCount'] as const;
export function isReservoirThermalParameters(value: unknown): value is ReservoirThermalParameters {
  if (!value || typeof value !== 'object' || Object.getPrototypeOf(value) !== Object.prototype || Reflect.ownKeys(value).length !== reservoirThermalParameterKeys.length) return false;
  const p = value as Record<string, number>;
  if (!reservoirThermalParameterKeys.every(k => Object.prototype.hasOwnProperty.call(p, k) && Number.isSafeInteger(p[k]) && p[k] >= (['timeNumerator', 'initialState', 'bathCode', 'jumpCount'].includes(k) ? 0 : 1) && p[k] <= (k === 'sampleSteps' ? 64 : k === 'initialState' ? 1 : k === 'jumpCount' ? 16 : k === 'betaA' || k === 'betaB' ? 8 : 1000000000))) return false;
  return p.timeNumerator <= 4 * p.timeDenominator && p.temperatureNumerator <= 1000000 * p.temperatureDenominator
    && p.bathCode < 2 ** p.jumpCount && (p.timeNumerator > 0 || p.jumpCount === 0);
}
