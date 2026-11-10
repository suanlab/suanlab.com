/** Client-safe adapters for the three thermodynamic-law experiments.
 * No registered curriculum/data imports. A curve is a sampled equilibrium/path
 * comparison, never an assertion of elapsed physical time.
 */
import { carnotCycle, carnotCycleState, type CarnotParameters, type CarnotDirection } from './learning-carnot-cycle';
import { reservoirCycle, type ReservoirCycleParameters } from './learning-reservoir-cycle';
import { gappedSpectrum, type GappedSpectrumParameters } from './learning-gapped-spectrum';
import type { BinaryObservation } from './learning-binary-products';

export const thermodynamicLawKinds = [
  'carnot-cycle-engine-refrigerator',
  'two-reservoir-cycle-entropy-admissibility',
  'gapped-spectrum-low-temperature-entropy',
] as const;
export type ThermodynamicLawKind = typeof thermodynamicLawKinds[number];
export const isThermodynamicLawKind = (id: string): id is ThermodynamicLawKind =>
  (thermodynamicLawKinds as readonly string[]).includes(id);

// Carnot: n, Th, Tc, Va, r, direction (0 forward / 1 reverse), progress.
// Reservoir: Th, Tc, Qh, relative entropy generation.
// Spectrum: independent copies, g0, g1, gap/kB, temperature.
export const carnotControlBounds = [[.25, 2], [250, 800], [250, 800], [.005, .04], [1, 4], [0, 1], [0, 1]] as const;
export const reservoirControlBounds = [[250, 800], [250, 800], [0, 4000], [-1, 4]] as const;
export const spectrumControlBounds = [[1, 64], [1, 16], [1, 16], [0, 1000], [0, 800]] as const;
function valid(v: unknown, bounds: readonly (readonly [number, number])[], integers: readonly number[]): v is number[] {
  return Array.isArray(v) && v.length === bounds.length && Array.from(v).every((x, i) =>
    typeof x === 'number' && Number.isFinite(x) && x >= bounds[i][0] && x <= bounds[i][1]
    && (!integers.includes(i) || Number.isInteger(x)));
}
// These validate stored control vectors, including repairable reservoir order.
export const validCarnotValues = (v: unknown): v is number[] => valid(v, carnotControlBounds, [5]);
export const validReservoirValues = (v: unknown): v is number[] => valid(v, reservoirControlBounds, []);
export const validSpectrumValues = (v: unknown): v is number[] => valid(v, spectrumControlBounds, [0, 1, 2]);
export const thermodynamicLawDefaults = {
  'carnot-cycle-engine-refrigerator': [1, 600, 300, .01, 2, 0, .375],
  'two-reservoir-cycle-entropy-admissibility': [600, 300, 1000, .25],
  'gapped-spectrum-low-temperature-entropy': [1, 1, 3, 200, 100],
} satisfies Record<ThermodynamicLawKind, number[]>;

/** Fixed teaching examples, deliberately independent of saved lab controls. */
export const thermodynamicLawConceptInputs: Record<ThermodynamicLawKind, number[][]> = {
  'carnot-cycle-engine-refrigerator': [
    [1, 600, 300, .01, 2, 0, 1], [1, 600, 300, .01, 2, 0, 1],
    [1, 600, 300, .01, 2, 1, 1], [1, 600, 300, .01, 1, 0, 1],
  ],
  'two-reservoir-cycle-entropy-admissibility': [
    [600, 300, 1000, 0], [600, 300, 1000, .25], [600, 300, 1000, -.25], [600, 300, 1000, 2],
  ],
  'gapped-spectrum-low-temperature-entropy': [
    [1, 1, 3, 200, 100], [1, 4, 3, 200, 0], [1, 4, 3, 0, 0], [1, 1, 3, 1000, Number.MIN_VALUE],
  ],
};

export type LawPoint = [number, number];
/** Split unavailable observations. A nonzero underflow is never drawn at zero.
 * Null also breaks a curve when a quantity is undefined (e.g. an efficiency).
 */
export function lawObservationSegments<T>(samples: readonly T[], x: (v: T) => number,
  quantity: (v: T) => BinaryObservation | null, scale: 'value' | 'logMagnitude' = 'value'): LawPoint[][] {
  const segments: LawPoint[][] = [];
  let current: LawPoint[] = [];
  for (const sample of samples) {
    const horizontal = x(sample), q = quantity(sample), vertical = q?.[scale];
    if (Number.isFinite(horizontal) && typeof vertical === 'number' && Number.isFinite(vertical)) {
      current.push([horizontal, vertical]);
    } else if (current.length) {
      segments.push(current); current = [];
    }
  }
  if (current.length) segments.push(current);
  return segments;
}

export function evaluateCarnotLaw(values: number[]) {
  if (!validCarnotValues(values)) throw new RangeError('Invalid Carnot controls');
  const parameters: CarnotParameters = { moles: values[0], hotTemperature: values[1], coldTemperature: values[2],
    initialVolume: values[3], expansionRatio: values[4] };
  const direction: CarnotDirection = values[5] === 0 ? 'forward' : 'reverse';
  if (parameters.hotTemperature < parameters.coldTemperature)
    return { status: 'invalid-reservoir-order' as const, parameters, direction, query: null, cycle: null, history: [] };
  return { status: 'ready' as const, parameters, direction,
    query: carnotCycleState(parameters, direction, values[6]), cycle: carnotCycle(parameters, direction),
    // Include every exact corner and current marker; never infer reverse by 1-s.
    history: [...new Set([...Array.from({ length: 129 }, (_, i) => i / 128), values[6]])]
      .sort((a, b) => a - b).map(s => carnotCycleState(parameters, direction, s)) };
}

export function evaluateReservoirLaw(values: number[]) {
  if (!validReservoirValues(values)) throw new RangeError('Invalid reservoir controls');
  const parameters: ReservoirCycleParameters = { hotTemperature: values[0], coldTemperature: values[1],
    heatHot: values[2], relativeEntropyGeneration: values[3] };
  if (parameters.hotTemperature < parameters.coldTemperature)
    return { status: 'invalid-reservoir-order' as const, parameters, query: null, history: [] };
  return { status: 'ready' as const, parameters, query: reservoirCycle(parameters),
    // This is a family of complete-cycle budgets, NOT a piston or time path.
    // Include selected/reversible cases and the rounded zero-work threshold.
    // Its actual sign still comes from the exact binary-input model below.
    history: [...new Set([...Array.from({ length: 129 }, (_, i) => -1 + 5 * i / 128),
      0, parameters.relativeEntropyGeneration,
      (parameters.hotTemperature - parameters.coldTemperature) / parameters.coldTemperature])]
      .filter(s => s >= -1 && s <= 4).sort((a, b) => a - b)
      .map(relativeEntropyGeneration => reservoirCycle({ ...parameters, relativeEntropyGeneration })) };
}

/** Positive-temperature logarithmic mesh, with local gap-scale refinement.
 * Zero temperature is a SEPARATE limit observation, never a log-axis endpoint.
 * The query may be far from the transition; keep both without joining an
 * unresolved transition using only a coarse full-range linear-temperature mesh.
 */
function spectrumTemperatures(gap: number, query: number) {
  const minLog = Math.log(Number.MIN_VALUE), maxLog = Math.log(800);
  const center = Math.log(gap > 0 ? gap : 1);
  const low = Math.max(minLog, Math.min(center - Math.log(1000), query > 0 ? Math.log(query) : center));
  const high = Math.min(maxLog, Math.max(center + Math.log(1000), query > 0 ? Math.log(query) : center));
  const points = [Math.exp(low), Math.exp(high), query];
  for (let i = 0; i <= 160; i++) points.push(Math.exp(low + (high - low) * i / 160));
  // A second local mesh resolves the transition even for a distant query.
  for (let i = 0; i <= 128; i++) {
    const logT = center - Math.log(1000) + Math.log(1e6) * i / 128;
    if (logT >= low && logT <= high) points.push(Math.exp(logT));
  }
  if (gap > 0 && gap <= 800) points.push(gap);
  return { temperatures: [...new Set(points.filter(T => T > 0 && T <= 800 && Number.isFinite(T)))].sort((a, b) => a - b),
    // Domain is explicit, including when a selected series has no finite values.
    logTemperatureDomain: [low / Math.LN10, high / Math.LN10] as [number, number] };
}
export function evaluateSpectrumLaw(values: number[]) {
  if (!validSpectrumValues(values)) throw new RangeError('Invalid spectrum controls');
  const parameters: GappedSpectrumParameters = { copies: values[0], groundDegeneracy: values[1],
    upperDegeneracy: values[2], gapTemperature: values[3], temperature: values[4] };
  const mesh = spectrumTemperatures(parameters.gapTemperature, parameters.temperature);
  return { parameters, query: gappedSpectrum(parameters),
    zeroTemperatureLimit: gappedSpectrum({ ...parameters, temperature: 0 }),
    logTemperatureDomain: mesh.logTemperatureDomain,
    history: mesh.temperatures.map(temperature => gappedSpectrum({ ...parameters, temperature })) };
}
