/** SI units throughout. Pure browser-safe calculations, without registry imports.
 * Bounds define the supported educational experiment, not physical universality.
 */
export interface ForcedOscillator {
  mass: number;
  stiffness: number;
  damping: number;
  forceAmplitude: number;
  frequency: number;
}
export interface OscillatorInitialState { position: number; velocity: number; }
type Matrix = number[][];
const identity = (): Matrix => Array.from({length: 4}, (_, i) => Array.from({length: 4}, (_, j) => Number(i === j)));
const multiply = (a: Matrix, b: Matrix): Matrix => a.map(row => b[0].map((_, j) => row.reduce((sum, value, k) => sum + value * b[k][j], 0)));
function validate(s: ForcedOscillator) {
  const values = [s.mass, s.stiffness, s.damping, s.forceAmplitude, s.frequency];
  const bounds = [[.25, 4], [.25, 4], [0, 8], [0, 4], [0, 8]];
  values.forEach((value, i) => {
    if (!Number.isFinite(value) || value < bounds[i][0] || value > bounds[i][1]) throw new RangeError(`Oscillator parameter ${i} is outside the supported range.`);
  });
}
function validateQuery(initial: OscillatorInitialState, time: number) {
  if (![initial.position, initial.velocity].every(v => Number.isFinite(v) && Math.abs(v) <= 4)) throw new RangeError('Initial position and velocity must be finite and between -4 and 4.');
  if (!Number.isFinite(time) || time < 0 || time > 32) throw new RangeError('Time must be finite and between 0 and 32 seconds.');
}
/** Scaling/squaring Taylor exponential: ||scaled A||∞ <= 1/2.
 * Fixed 24 terms give a truncation tail below binary64 roundoff at that norm.
 * This evaluates the linear flow; drawing samples are not integration steps.
 */
function exponential(a: Matrix): Matrix {
  const norm = Math.max(...a.map(row => row.reduce((s, v) => s + Math.abs(v), 0)));
  const squarings = norm > .5 ? Math.ceil(Math.log2(2 * norm)) : 0;
  const scaled = a.map(row => row.map(v => v / 2 ** squarings));
  let result = identity(), term = identity();
  for (let n = 1; n <= 24; n++) {
    term = multiply(term, scaled).map(row => row.map(v => v / n));
    result = result.map((row, i) => row.map((v, j) => v + term[i][j]));
  }
  for (let i = 0; i < squarings; i++) result = multiply(result, result);
  return result;
}
function stateUnchecked(s: ForcedOscillator, initial: OscillatorInitialState, time: number) {
  // Augment by cos(Omega*t), sin(Omega*t). No resonance denominator is used.
  const a = [[0, 1, 0, 0], [-s.stiffness / s.mass, -s.damping / s.mass, s.forceAmplitude / s.mass, 0], [0, 0, 0, -s.frequency], [0, 0, s.frequency, 0]];
  const flow = exponential(a.map(row => row.map(v => v * time)));
  const vector = [initial.position, initial.velocity, 1, 0];
  const [position, velocity] = flow.slice(0, 2).map(row => row.reduce((sum, v, j) => sum + v * vector[j], 0));
  const force = s.forceAmplitude * Math.cos(s.frequency * time);
  return {time, position, velocity, acceleration: (force - s.damping * velocity - s.stiffness * position) / s.mass,
    force, energy: (s.mass * velocity ** 2 + s.stiffness * position ** 2) / 2,
    inputPower: force * velocity, heatPower: s.damping * velocity ** 2};
}
export function forcedOscillatorState(s: ForcedOscillator, initial: OscillatorInitialState, time: number) {
  validate(s); validateQuery(initial, time);
  return stateUnchecked(s, initial, time);
}

/** Intrinsic susceptibility uses exp(+i Omega t); measured phase needs nonzero input.
 * At DC, mean power is the asymptotic static value, not a finite-cycle average.
 */
export function oscillatorFrequencyResponse(s: ForcedOscillator) {
  validate(s);
  const {mass: m, stiffness: k, damping: c, forceAmplitude: f, frequency: w} = s;
  const real = k - m * w * w, imaginary = c * w, denominator = real * real + imaginary * imaginary;
  const naturalFrequency = Math.sqrt(k / m), discriminant = c * c - 4 * m * k;
  // The canonical sqrt value names the exact natural-frequency marker. Its
  // floating-point square need not round back to k/m. No neighborhood is clamped.
  const singular = c === 0 && (real === 0 || w === naturalFrequency);
  const dampingRegime = discriminant < 0 ? 'underdamped' : discriminant === 0 ? 'critical' : 'overdamped';
  const gain = singular ? null : 1 / Math.sqrt(denominator);
  const intrinsicLag = singular ? null : Math.atan2(imaginary, real);
  const amplitude = f === 0 ? 0 : gain === null ? null : f * gain;
  const phaseLag = w > 0 && f > 0 && amplitude !== null && amplitude > 0 ? intrinsicLag : null;
  const peakSquared = k / m - c * c / (2 * m * m);
  const power = singular ? (f === 0 ? 0 : null) : w === 0 ? 0 : f * f * c * w * w / (2 * denominator);
  const root = Math.sqrt(c * c + 4 * m * k);
  // Rationalized lower root avoids subtraction cancellation at large damping.
  const halfPower = c > 0 && f > 0 ? {lower: 2 * k / (root + c), upper: (root + c) / (2 * m), width: c / m} : null;
  return {naturalFrequency, dampingRegime,
    dampedFreeFrequency: discriminant < 0 ? Math.sqrt(-discriminant) / (2 * m) : null,
    status: singular ? (f === 0 ? 'singular-transfer-zero-input' : 'no-bounded-periodic-response') : c > 0 ? 'attracting-particular' : 'nonattracting-particular',
    susceptibility: singular ? null : {real: real / denominator, imaginary: -imaginary / denominator},
    gain, amplitude, intrinsicLag, phaseLag, period: w > 0 ? 2 * Math.PI / w : null,
    meanPower: power, meanStoredEnergy: w > 0 && amplitude !== null ? (m * w * w + k) * amplitude ** 2 / 4 : null,
    heatPerCycle: w > 0 && power !== null ? power * 2 * Math.PI / w : null,
    displacementPeakFrequency: c > 0 && peakSquared > 0 ? Math.sqrt(peakSquared) : null,
    displacementPeakStatus: c === 0 ? 'unbounded-resonance' : peakSquared > 0 ? 'positive-interior-peak' : 'maximum-at-dc',
    powerPeakFrequency: c > 0 && f > 0 ? naturalFrequency : null,
    qualityFactor: c > 0 ? m * naturalFrequency / c : null, halfPower};
}

/** Independently integrate both physical power functions. Seed panels resolve the
 * fastest supported oscillation and decay scale before adaptive subdivision.
 * Simpson estimates are diagnostics, not certified global error bounds.
 */
function energyQuadrature(s: ForcedOscillator, initial: OscillatorInitialState, duration: number) {
  const absoluteTolerance = 1e-8, relativeTolerance = 1e-8, evaluationBudget = 32768;
  let evaluations = 0, completed = true, reservedEvaluations = 0;
  const value = (time: number) => {
    evaluations++;
    const p = stateUnchecked(s, initial, time);
    return [p.inputPower, p.heatPower];
  };
  const simpson = (a: number, b: number, left: number[], mid: number[], right: number[]) => left.map((v, i) => (b - a) * (v + 4 * mid[i] + right[i]) / 6);
  const refine = (a: number, b: number, left: number[], mid: number[], right: number[], coarse: number[], tolerance: number, depth: number): {integrals: number[]; estimates: number[]} => {
    if (evaluations + 2 + reservedEvaluations > evaluationBudget || depth >= 20) {
      completed = false;
      return {integrals: coarse, estimates: [0, 0]};
    }
    const center = (a + b) / 2, lm = value((a + center) / 2), rm = value((center + b) / 2);
    const l = simpson(a, center, left, lm, mid), r = simpson(center, b, mid, rm, right);
    const fine = l.map((v, i) => v + r[i]), delta = fine.map((v, i) => v - coarse[i]);
    if (delta.every((v, i) => Math.abs(v) <= 15 * (tolerance + relativeTolerance * Math.abs(fine[i])))) {
      return {integrals: fine.map((v, i) => v + delta[i] / 15), estimates: delta.map(v => Math.abs(v) / 15)};
    }
    const lo = refine(a, center, left, lm, mid, l, tolerance / 2, depth + 1);
    const hi = refine(center, b, mid, rm, right, r, tolerance / 2, depth + 1);
    return {integrals: lo.integrals.map((v, i) => v + hi.integrals[i]), estimates: lo.estimates.map((v, i) => v + hi.estimates[i])};
  };
  const panels = Math.max(16, Math.ceil(duration * Math.max(s.frequency, Math.sqrt(s.stiffness / s.mass), s.damping / (4 * s.mass)) * 2));
  let work = 0, heat = 0, workEstimate = 0, heatEstimate = 0;
  if (duration > 0) for (let i = 0; i < panels; i++) {
    reservedEvaluations = 3 * (panels - i - 1);
    const a = duration * i / panels, b = duration * (i + 1) / panels;
    const left = value(a), mid = value((a + b) / 2), right = value(b);
    const r = refine(a, b, left, mid, right, simpson(a, b, left, mid, right), absoluteTolerance / panels, 0);
    work += r.integrals[0]; heat += r.integrals[1]; workEstimate += r.estimates[0]; heatEstimate += r.estimates[1];
  }
  return {work, heat, method: 'adaptive-simpson', completed, evaluations, evaluationBudget, seedPanels: panels,
    absoluteTolerance, relativeTolerance, workErrorEstimate: completed ? workEstimate : null, heatErrorEstimate: completed ? heatEstimate : null};
}

/** Drawing samples and adaptive energy integration are separate calculations. */
export function sinusoidalForcingTransient(s: ForcedOscillator, initial: OscillatorInitialState, duration: number) {
  validate(s); validateQuery(initial, duration);
  const response = oscillatorFrequencyResponse(s);
  const intervals = 256;
  const states = Array.from({length: intervals + 1}, (_, i) => stateUnchecked(s, initial, duration * i / intervals));
  const quadrature = energyQuadrature(s, initial, duration);
  const {work, heat} = quadrature, terminal = states[intervals];
  // Zero forcing always admits the zero particular solution, even when the
  // intrinsic transfer function is singular at the selected frequency.
  const particular = s.forceAmplitude === 0 ? {position: 0, velocity: 0} : response.susceptibility === null ? null : {
    position: s.forceAmplitude * response.susceptibility.real,
    velocity: -s.frequency * s.forceAmplitude * response.susceptibility.imaginary};
  return {system: {...s}, initial: {...initial}, duration, response, states, terminal,
    particularInitial: particular,
    homogeneousInitial: particular === null ? null : {position: initial.position - particular.position, velocity: initial.velocity - particular.velocity},
    work, heat, energyChange: terminal.energy - states[0].energy,
    balanceResidual: terminal.energy - states[0].energy - work + heat,
    quadrature,
    sampling: {method: 'linear-flow-exponential', drawingIntervals: intervals, maximumPhaseAdvance: Math.max(s.frequency, response.naturalFrequency) * duration / intervals},
    finiteWindowMeanInputPower: duration > 0 ? work / duration : null};
}
