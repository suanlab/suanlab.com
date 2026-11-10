/** Smooth prescribed one-dimensional flows, constant cross-section, SI units.
 * Analytic identities and quadrature differences are not independent certificates.
 * No general Euler solver, shocks, viscous stress, heat conduction or vacuum solve.
 */
export type ContinuumValue = number | null;
type V = ContinuumValue;
export type ConservedTriple = [V, V, V];
function bound(x: number, lo: number, hi: number, label: string) {
  if (!Number.isFinite(x) || x < lo || x > hi) throw new RangeError(`${label} outside [${lo}, ${hi}]`);
}
function mul(...xs: V[]): V {
  if (xs.includes(0)) return 0;
  if (xs.includes(null)) return null;
  let result = 1;
  for (const x of xs) {
    result *= x!;
    if (!Number.isFinite(result) || result === 0) {
      // A later factor may rescue an intermediate underflow/overflow. Keep
      // the ordinary path unchanged and use a logarithmic fallback only here.
      const sign = xs.reduce<number>((a, b) => a * Math.sign(b!), 1);
      const log = xs.reduce<number>((a, b) => a + Math.log(Math.abs(b!)), 0);
      const fallback = log <= Math.log(Number.MIN_VALUE) - Math.log(2) ? 0 : sign * Math.exp(log);
      return fallback === 0 || !Number.isFinite(fallback) ? null : fallback;
    }
  }
  return result;
}
function sum(...xs: V[]): V {
  if (xs.includes(null)) return null;
  let s = 0, error = 0;
  for (const x of xs) { const t = s + x!; error += Math.abs(s) >= Math.abs(x!) ? (s - t) + x! : (x! - t) + s; s = t; }
  const result = s + error;
  return Number.isFinite(result) ? result || 0 : null;
}
/** Sum complete products before projecting to the nullable linear range. */
function sumProducts(terms: number[][]): V {
  const linear = terms.map(xs => mul(...xs));
  if (!linear.includes(null)) return sum(...linear);
  let sign = 0, log = 0;
  for (const xs of terms) {
    if (xs.includes(0)) continue;
    const nextSign = xs.reduce((a, x) => a * Math.sign(x), 1);
    const nextLog = xs.reduce((a, x) => a + Math.log(Math.abs(x)), 0);
    if (!sign) { sign = nextSign; log = nextLog; continue; }
    const larger = log >= nextLog;
    const hi = larger ? log : nextLog, lo = larger ? nextLog : log;
    const hiSign = larger ? sign : nextSign;
    if (sign === nextSign) log = hi + Math.log1p(Math.exp(lo - hi));
    else if (hi === lo) return null; // equal rounded logs do not prove exact cancellation
    else { sign = hiSign; log = hi + Math.log(-Math.expm1(lo - hi)); }
  }
  if (!sign) return 0;
  const value = log <= Math.log(Number.MIN_VALUE) - Math.log(2) ? 0 : sign * Math.exp(log);
  return value === 0 || !Number.isFinite(value) ? null : value;
}
const neg = (x: V) => mul(-1, x);
function div(x: V, y: V): V {
  if (x === null || y === null || y === 0) return null;
  const q = x / y;
  return !Number.isFinite(q) || x !== 0 && q === 0 ? null : q || 0;
}
const sin = (x: V) => x === null ? null : Math.sin(x);
const cos = (x: V) => x === null ? null : Math.cos(x);
function sinc(x: V): V {
  if (x === null) return null;
  if (Math.abs(x) < .01) { const q = x * x; return 1 + q * (-1 / 6 + q * (1 / 120 - q / 5040)); }
  return Math.sin(x) / x;
}
const state = (xs: V[]) => xs.includes(null) ? 'arithmetic-range' as const : 'finite' as const;
function time(t: number) { bound(t, -1, 1, 'Time s'); }

export interface DensityTransport {
  /** Reference density; the k=0 family has actual uniform density rho0(1+contrast). */
  meanDensity: number; contrast: number; wavenumber: number; velocity: number; area: number;
}
export interface TranslatingMassWindow { center: number; halfWidth: number; speed: number; }
function transportParameters(s: DensityTransport) {
  bound(s.meanDensity, .25, 4, 'Density kg/m3'); bound(s.contrast, -.8, .8, 'Density contrast');
  bound(s.wavenumber, 0, 8, 'Wavenumber rad/m'); bound(s.velocity, -2, 2, 'Fluid velocity m/s'); bound(s.area, .25, 4, 'Area m2');
}
function windowParameters(w: TranslatingMassWindow) {
  bound(w.center, -1, 1, 'Initial window center m'); bound(w.halfWidth, 0, 1, 'Initial half width m'); bound(w.speed, -2, 2, 'Window speed m/s');
}
export function transportedDensityState(s: DensityTransport, x: number, t: number) {
  transportParameters(s); bound(x, -8, 8, 'Position m'); time(t);
  const phase = mul(s.wavenumber, sum(x, neg(mul(s.velocity, t))));
  const density = mul(s.meanDensity, sum(1, mul(s.contrast, cos(phase))));
  const densityX = mul(-s.meanDensity, s.contrast, s.wavenumber, sin(phase));
  const densityT = mul(-s.velocity, densityX), massFlux = mul(density, s.velocity);
  return {x, time: t, phase, density, velocity: s.velocity, massFlux, densityX, densityT,
    massFluxX: mul(s.velocity, densityX), continuityResidual: sum(densityT, mul(s.velocity, densityX)),
    arithmeticStatus: state([phase, density, massFlux, densityX, densityT]),
    model: 'prescribed-smooth-constant-speed-density-wave' as const};
}
export function movingMassWindowState(s: DensityTransport, w: TranslatingMassWindow, t: number) {
  transportParameters(s); windowParameters(w); time(t);
  const center = sum(w.center, mul(w.speed, t)), h = w.halfWidth;
  const left = sum(center, -h), right = sum(center, h), relativeSpeed = s.velocity - w.speed;
  const phase = mul(s.wavenumber, sum(w.center, mul(w.speed - s.velocity, t))), shape = sinc(mul(s.wavenumber, h));
  const analyticMeanDensity = mul(s.meanDensity, sum(1, mul(s.contrast, cos(phase), shape)));
  const volume = mul(2, s.area, h), mass = mul(volume, analyticMeanDensity);
  // Cancel the equal background before evaluating the boundary-flow contrast.
  const stableNetInflow = mul(2, s.area, s.meanDensity, s.contrast, relativeSpeed, sin(phase), sin(mul(s.wavenumber, h)));
  const densityLeft = left === null ? null : transportedDensityState(s, left, t).density;
  const densityRight = right === null ? null : transportedDensityState(s, right, t).density;
  const leftOutward = h === 0 ? null : mul(-s.area, densityLeft, relativeSpeed);
  const rightOutward = h === 0 ? null : mul(s.area, densityRight, relativeSpeed);
  const ordinaryNetInflow = h === 0 ? null : neg(sum(leftOutward, rightOutward));
  const wrongLabFrameNetInflow = mul(2, s.area, s.meanDensity, s.contrast, s.velocity, sin(phase), sin(mul(s.wavenumber, h)));
  return {time: t, center, left, right, halfWidth: h, area: s.area, volume, mass,
    analyticMeanDensity, observedMeanDensity: div(mass, volume), densityLeft, densityRight,
    fluidVelocity: s.velocity, boundaryVelocity: w.speed, relativeSpeed, leftOutward, rightOutward,
    stableNetInflow, massRate: stableNetInflow, ordinaryNetInflow, wrongLabFrameNetInflow,
    wrongFluxResidual: sum(stableNetInflow, neg(wrongLabFrameNetInflow)), materialWindow: w.speed === s.velocity,
    geometryStatus: h === 0 ? 'zero-length' as const : left === null || right === null ? 'arithmetic-range' as const : left === right ? 'width-below-coordinate-resolution' as const : 'finite' as const,
    normalizationStatus: h === 0 ? 'zero-volume-undefined' as const : state([volume, div(mass, volume)]),
    diagnosticMeaning: 'analytic-family-balance-not-independent-verification' as const};
}

/** Gauss-Legendre quadrature with two orders for observation, not a certified bound. */
function quadrature(f: (t: number) => V, start: number, end: number, order: number): V {
  if (start === end) return 0;
  const values: V[] = [], midpoint = (start + end) / 2, half = (end - start) / 2;
  if (half === 0) return null;
  for (let i = 0; i < order / 2; i++) {
    let z = Math.cos(Math.PI * (i + .75) / (order + .5)), derivative = 0;
    for (let iteration = 0; iteration < 20; iteration++) {
      let p = 1, previous = 0;
      for (let j = 1; j <= order; j++) { const next = ((2 * j - 1) * z * p - (j - 1) * previous) / j; previous = p; p = next; }
      derivative = order * (z * p - previous) / (z * z - 1);
      const next = z - p / derivative;
      if (Math.abs(next - z) <= 4 * Number.EPSILON) { z = next; break; }
      z = next;
    }
    const weight = 2 / ((1 - z * z) * derivative * derivative);
    values.push(mul(weight, f(midpoint - half * z)), mul(weight, f(midpoint + half * z)));
  }
  return mul(half, sum(...values));
}
export function movingMassWindowBalance(s: DensityTransport, w: TranslatingMassWindow, start: number, end: number) {
  const initial = movingMassWindowState(s, w, start), selected = movingMassWindowState(s, w, end), dt = end - start;
  const phaseMid = mul(s.wavenumber, sum(w.center, mul(w.speed - s.velocity, start + end, .5)));
  const halfPhaseChange = mul(.5, s.wavenumber, w.speed - s.velocity, dt);
  let massChange = mul(-4, s.area, w.halfWidth, s.meanDensity, s.contrast,
    sinc(mul(s.wavenumber, w.halfWidth)), sin(phaseMid), sin(halfPhaseChange));
  if (massChange === null) {
    // sin(delta/2)=(delta/2)sinc(delta/2). Compose the midpoint phase
    // from its initial and drift parts without first rounding tiny angles to0.
    const relative = w.speed - s.velocity, dt = end - start;
    const A = s.wavenumber * w.center, B = .5 * s.wavenumber * relative * (start + end);
    const half = .5 * s.wavenumber * relative * dt;
    const common = [-2, s.area, w.halfWidth, s.meanDensity, s.contrast,
      sinc(s.wavenumber * w.halfWidth)!, s.wavenumber, relative, dt, sinc(half)!];
    massChange = sumProducts([
      [...common, s.wavenumber, w.center, sinc(A)!, Math.cos(B)],
      [...common, .5, s.wavenumber, relative, start + end, sinc(B)!, Math.cos(A)],
    ]);
  }
  const flow = (t: number) => movingMassWindowState(s, w, t).stableNetInflow;
  const zeroFlow = w.speed === s.velocity || s.contrast === 0 || s.wavenumber === 0 || w.halfWidth === 0;
  const integratedNetInflow = zeroFlow ? 0 : quadrature(flow, start, end, 32), coarse = zeroFlow ? 0 : quadrature(flow, start, end, 16);
  return {start, end, initial, selected, massChange, roundedMassChange: sum(selected.mass, neg(initial.mass)),
    integratedNetInflow, residual: sum(massChange, neg(integratedNetInflow)), quadratureOrderDifference: sum(integratedNetInflow, neg(coarse)),
    diagnosticMeaning: 'selected-quadrature-comparison-not-a-certified-global-error-bound' as const};
}

export interface AffineGas {
  initialDensity: number; initialPressure: number; expansionRate: number; drift: number; gamma: number; area: number;
}
export interface AffineControlWindow extends TranslatingMassWindow { mode: 'fixed' | 'translating' | 'material'; }
function gasParameters(s: AffineGas) {
  bound(s.initialDensity, .25, 4, 'Initial density kg/m3'); bound(s.initialPressure, 0, 4, 'Initial pressure Pa');
  bound(s.expansionRate, -.5, .5, 'Expansion rate 1/s'); bound(s.drift, -2, 2, 'Initial drift m/s');
  bound(s.gamma, 1.1, 2, 'Gamma'); bound(s.area, .25, 4, 'Area m2');
}
export function affineGasState(s: AffineGas, x: number, t: number) {
  gasParameters(s); bound(x, -8, 8, 'Position m'); time(t);
  const scale = 1 + s.expansionRate * t, gradient = s.expansionRate / scale;
  const density = s.initialDensity / scale, velocity = sum(s.drift, mul(s.expansionRate, x));
  const u = div(velocity, scale), pressure = mul(s.initialPressure, Math.exp(-s.gamma * Math.log1p(s.expansionRate * t)));
  const internalEnergyDensity = div(pressure, s.gamma - 1), kineticEnergyDensity = mul(.5, density, u, u);
  const energyDensity = sum(internalEnergyDensity, kineticEnergyDensity), momentumDensity = mul(density, u);
  const conserved: ConservedTriple = [density, momentumDensity, energyDensity];
  const flux: ConservedTriple = [momentumDensity, sum(mul(density, u, u), pressure), mul(sum(energyDensity, pressure), u)];
  const densityT = mul(-gradient, density), momentumT = mul(-2, gradient, momentumDensity);
  const energyT = mul(-gradient, sum(mul(s.gamma, internalEnergyDensity), mul(3, kineticEnergyDensity)));
  const fluxX: ConservedTriple = [mul(density, gradient), mul(2, gradient, momentumDensity), neg(energyT)];
  return {x, time: t, scale, density, velocity: u, velocityGradient: gradient, pressure,
    internalEnergyDensity, kineticEnergyDensity, energyDensity, momentumDensity, conserved, flux,
    timeDerivatives: [densityT, momentumT, energyT] as ConservedTriple, fluxDerivatives: fluxX,
    analyticResiduals: [sum(densityT, fluxX[0]), sum(momentumT, fluxX[1]), sum(energyT, fluxX[2])] as ConservedTriple,
    missingPressureWorkResidual: mul(-1, pressure, gradient),
    arithmeticStatus: state([...conserved, ...flux]),
    model: 'smooth-affine-positive-density-ideal-gas-Euler-family' as const};
}
export function affineGasWindowState(s: AffineGas, w: AffineControlWindow, t: number) {
  gasParameters(s); windowParameters(w); time(t);
  if (!['fixed', 'translating', 'material'].includes(w.mode)) throw new RangeError('Unknown control-volume mode');
  const a = s.expansionRate, scale = 1 + a * t, material = w.mode === 'material';
  const speed = w.mode === 'fixed' ? 0 : w.speed;
  const center = material ? sum(mul(scale, w.center), mul(s.drift, t)) : sum(w.center, mul(speed, t));
  const halfWidth = material ? mul(scale, w.halfWidth) : w.halfWidth;
  const left = sum(center, neg(halfWidth)), right = sum(center, halfWidth);
  const at = (x: V) => x === null ? null : affineGasState(s, x, t);
  const q = at(center), L = at(left), R = at(right), h = halfWidth;
  const leftSpeed = material ? sum(s.drift, mul(a, w.center - w.halfWidth)) : speed;
  const rightSpeed = material ? sum(s.drift, mul(a, w.center + w.halfWidth)) : speed;
  const volume = mul(2, s.area, h), mass = mul(2, s.area, h, q?.density ?? null), uc = q?.velocity ?? null, H = a / scale;
  const momentum = mul(mass, uc), internalEnergy = mul(2, s.area, h, q?.internalEnergyDensity ?? null);
  const variance = mul(H, H, h, h, 1 / 3), kineticEnergy = mul(.5, mass, sum(mul(uc, uc), variance));
  const totalEnergy = sum(internalEnergy, kineticEnergy), storage: ConservedTriple = [mass, momentum, totalEnergy];
  function relativeFlux(point: ReturnType<typeof affineGasState> | null, boundarySpeed: V): ConservedTriple {
    const u = point?.velocity ?? null, p = point?.pressure ?? null;
    const relative = material ? 0 : sum(u, neg(boundarySpeed));
    return [mul(point?.density ?? null, relative), sum(mul(point?.momentumDensity ?? null, relative), p),
      sum(mul(point?.energyDensity ?? null, relative), mul(p, u))];
  }
  const leftFlux = relativeFlux(L, leftSpeed), rightFlux = relativeFlux(R, rightSpeed);
  const leftOutward = leftFlux.map(x => w.halfWidth === 0 ? null : mul(-s.area, x)) as ConservedTriple;
  const rightOutward = rightFlux.map(x => w.halfWidth === 0 ? null : mul(s.area, x)) as ConservedTriple;
  const ordinaryNetInflow = leftOutward.map((x, i) => neg(sum(x, rightOutward[i]))) as ConservedTriple;
  const pressureWorkNet = mul(-2, s.area, q?.pressure ?? null, H, h);
  const massRate = material ? 0 : mul(-H, mass);
  const momentumRate = material ? 0 : mul(mass, H, sum(speed, mul(-2, uc)));
  const kineticRate = material ? 0 : mul(-H, mass,
    sum(mul(.5, uc, uc), mul(uc, sum(uc, -speed)), mul(.5, H, H, h, h)));
  const internalRate = material ? pressureWorkNet : mul(-s.gamma, H, internalEnergy);
  const storageRates: ConservedTriple = [massRate, momentumRate, sum(internalRate, kineticRate)];
  return {time: t, mode: w.mode, scale, center, halfWidth, left, right, leftSpeed, rightSpeed,
    area: s.area, volume, query: q, leftState: L, rightState: R, storage, internalEnergy, kineticEnergy,
    leftOutward, rightOutward, ordinaryNetInflow, analyticNetInflow: [...storageRates] as ConservedTriple, storageRates,
    pressureWorkNet, missingPressureWorkNetInflow: sum(storageRates[2], neg(pressureWorkNet)),
    missingPressureWorkResidual: pressureWorkNet,
    geometryStatus: w.halfWidth === 0 ? 'zero-length' as const : h === null || left === null || right === null ? 'arithmetic-range' as const : left === right ? 'width-below-coordinate-resolution' as const : 'finite' as const,
    arithmeticStatus: state([...storage, ...storageRates]),
    diagnosticMeaning: 'analytic-Euler-family-balance-not-independent-certification' as const};
}
export function affineGasWindowBalance(s: AffineGas, w: AffineControlWindow, start: number, end: number) {
  const initial = affineGasWindowState(s, w, start), selected = affineGasWindowState(s, w, end);
  const ds = mul(s.expansionRate, end - start), s0 = initial.scale, s1 = selected.scale;
  // The dimensionless ratio may round to zero although the weighted change
  // is representable. Factor x out of log1p(x), and z out of expm1(z), before
  // multiplying physical factors. Their analytic ratios tend to one at zero.
  const ratio = s.expansionRate * (end - start) / s0;
  const logRelative = ratio === 0 ? 1 : Math.log1p(ratio) / ratio;
  const weightedPowerChange = (weightFactors: number[], power: number): V => {
    const z = power * Math.log1p(ratio);
    const expRelative = Math.abs(z) < .01
      ? 1 + z * (.5 + z * (1 / 6 + z * (1 / 24 + z * (1 / 120 + z / 720))))
      : Math.expm1(z) / z;
    return mul(...weightFactors, power, s.expansionRate, end - start, 1 / s0, logRelative, expRelative);
  };
  const material = w.mode === 'material', speed = w.mode === 'fixed' ? 0 : w.speed;
  const u0 = initial.query?.velocity ?? null, H0 = s.expansionRate / s0, h = w.halfWidth;
  const massChange = material ? 0 : mul(-1, initial.storage[0], div(ds, s1));
  const momentumChange = material ? 0 : mul(initial.storage[0], div(ds, s1 * s1), sum(speed * s0, mul(-1, u0, sum(2 * s0, ds))));
  const internalChange = weightedPowerChange([2, s.area, w.halfWidth, s.initialPressure,
    1 / (s.gamma - 1), Math.exp((material ? 1 - s.gamma : -s.gamma) * Math.log(s0))], material ? 1 - s.gamma : -s.gamma);
  const kineticChange = material ? 0 : mul(.5, initial.storage[0], div(ds, s1 ** 3),
    sum(mul(2, u0, speed, s0, s0), mul(speed, speed, ds, s0),
      neg(mul(sum(mul(u0, u0), mul(H0, H0, h, h, 1 / 3)), sum(3 * s0 * s0, mul(3 * s0, ds), mul(ds, ds))))));
  const changes: ConservedTriple = [massChange, momentumChange, sum(internalChange, kineticChange)];
  const integrate = (i: number, order: number) => s.expansionRate === 0 || w.halfWidth === 0 || material && (i < 2 || s.initialPressure === 0)
    ? 0 : quadrature(t => affineGasWindowState(s, w, t).analyticNetInflow[i], start, end, order);
  const integratedNetInflows = [0, 1, 2].map(i => integrate(i, 32)) as ConservedTriple;
  return {start, end, initial, selected, changes, internalChange, kineticChange,
    roundedChanges: selected.storage.map((x, i) => sum(x, neg(initial.storage[i]))) as ConservedTriple,
    integratedNetInflows, residuals: changes.map((x, i) => sum(x, neg(integratedNetInflows[i]))) as ConservedTriple,
    quadratureOrderDifferences: integratedNetInflows.map((x, i) => sum(x, neg(integrate(i, 16)))) as ConservedTriple,
    diagnosticMeaning: 'stable-family-changes-and-selected-quadrature-not-rigorous-error-bounds' as const};
}
