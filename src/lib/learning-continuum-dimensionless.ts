/** Two steady, constant-coefficient scalar boundary-value families; not a general PDE solver. */
export interface TransportObservation {
  value: number | null;
  sign: number;
  logMagnitude: number | null;
}
interface Q { sign: number; log: number }
const zero: Q = { sign: 0, log: 0 };
const ln2 = Math.log(2);
function product(...factors: number[]): Q {
  return factors.includes(0) ? zero : {
    sign: factors.reduce((s, v) => s * Math.sign(v), 1),
    log: factors.reduce((s, v) => s + Math.log(Math.abs(v)), 0),
  };
}
function shift(q: Q, logScale: number): Q {
  return q.sign === 0 ? zero : { sign: q.sign, log: q.log + logScale };
}
function multiply(a: Q, b: Q): Q {
  return a.sign === 0 || b.sign === 0 ? zero : { sign: a.sign * b.sign, log: a.log + b.log };
}
function sum(...terms: Q[]): Q {
  const nonzero = terms.filter(v => v.sign !== 0);
  if (!nonzero.length) return zero;
  const top = Math.max(...nonzero.map(v => v.log));
  const scaled = nonzero.reduce((s, v) => s + v.sign * Math.exp(v.log - top), 0);
  return scaled === 0 ? zero : { sign: Math.sign(scaled), log: top + Math.log(Math.abs(scaled)) };
}
function observe(q: Q): TransportObservation {
  if (q.sign === 0) return { value: 0, sign: 0, logMagnitude: null };
  // exp(log(MIN_VALUE/2)) may round upward after the logarithm itself rounds.
  // Retain the signed log, rather than inventing a representable positive value at the zero tie.
  if (q.log <= Math.log(Number.MIN_VALUE) - ln2) return { value: null, sign: q.sign, logMagnitude: q.log };
  const n = q.sign * Math.exp(q.log);
  return { value: n === 0 || !Number.isFinite(n) ? null : n, sign: q.sign, logMagnitude: q.log };
}
function bound(v: number, low: number, high: number, name: string) {
  if (!Number.isFinite(v) || v < low || v > high) throw new RangeError(`Invalid ${name}`);
}
interface Geometry { diffusivity: number; length: number; area: number }
function geometry(p: Geometry) {
  bound(p.diffusivity, 1e-4, 4, 'diffusivity');
  bound(p.length, .25, 4, 'length');
  bound(p.area, .25, 4, 'area');
}
/** log(1-exp(-z)) from log(z); retains a positive z even if products underflow. */
function logOneMinusExpNegative(logZ: number) {
  if (logZ < -36) return logZ;
  return Math.log(-Math.expm1(-Math.exp(logZ)));
}
function logSinh(logZ: number) {
  if (logZ < -10) {
    const square = Math.exp(2 * logZ);
    return logZ + Math.log1p(square / 6 + square * square / 120);
  }
  const z = Math.exp(logZ);
  return z - ln2 + Math.log(-Math.expm1(-2 * z));
}
function logCosh(z: number) {
  if (z < 1) return Math.log1p(2 * Math.sinh(z / 2) ** 2);
  return z - ln2 + Math.log1p(Math.exp(-2 * z));
}

export interface PecletParameters extends Geometry {
  velocity: number;
  leftConcentration: number;
  rightConcentration: number;
}
/** SI: D m²/s, L m, A m², U m/s, c kg/m³; coordinate xi=x/L. */
export function pecletAt(p: PecletParameters, xi: number) {
  geometry(p);
  bound(p.velocity, -2, 2, 'velocity');
  bound(p.leftConcentration, 0, 4, 'left concentration');
  bound(p.rightConcentration, 0, 4, 'right concentration');
  bound(xi, 0, 1, 'dimensionless position');
  const { diffusivity: D, length: L, velocity: U, area: A } = p;
  const left = p.leftConcentration, right = p.rightConcentration, contrast = right - left;
  const pe = shift(product(U, L), -Math.log(D)), absPe = U === 0 ? 0 : Math.exp(pe.log);
  const signedPe = Math.sign(U) * absPe;
  let wLeft: Q, wRight: Q, derivative: Q, correction: Q;
  if (U === 0) {
    wLeft = product(1 - xi); wRight = product(xi); derivative = product(1); correction = zero;
  } else {
    const denominator = logOneMinusExpNegative(pe.log);
    const numerator = (fraction: number) => fraction === 0 ? zero :
      { sign: 1, log: logOneMinusExpNegative(pe.log + Math.log(fraction)) - denominator };
    wLeft = numerator(1 - xi); wRight = numerator(xi);
    if (U > 0) wRight = shift(wRight, -absPe * (1 - xi));
    else wLeft = shift(wLeft, -absPe * xi);
    derivative = { sign: 1, log: pe.log - denominator - absPe * (U > 0 ? 1 - xi : xi) };
    if (xi === 0 || xi === 1) correction = zero;
    else if (absPe < 1e-4) {
      const factor = 1 + signedPe * (2 * xi - 1) / 6 + signedPe ** 2 * xi * (xi - 1) / 12;
      correction = multiply(pe, product(xi, xi - 1, .5, factor));
    } else correction = sum(wRight, product(-xi));
  }
  // Equal boundary data are physically constant, even if separate rounded weights do not sum to one.
  const concentration = contrast === 0 ? product(left) : sum(multiply(product(left), wLeft), multiply(product(right), wRight));
  const gradient = shift(multiply(product(contrast), derivative), -Math.log(L));
  const curvature = shift(multiply(product(U), gradient), -Math.log(D));
  const advective = multiply(product(U), concentration), diffusive = multiply(product(-D), gradient);
  let flux: Q;
  if (U === 0) flux = shift(product(-D, contrast), -Math.log(L));
  else if (contrast === 0) flux = product(U, left);
  else {
    const denominator = logOneMinusExpNegative(pe.log);
    // Anchor at the upstream concentration; retain the exponentially small other-boundary term.
    const tail = shift(product(U, contrast, U > 0 ? -1 : 1), -absPe - denominator);
    flux = sum(product(U, U > 0 ? left : right), tail);
  }
  const concentrationValue = observe(concentration).value, gradientValue = observe(gradient).value;
  const roundedFlux = concentrationValue === null || gradientValue === null ? null : U * concentrationValue - D * gradientValue;
  return {
    xi, position: observe(product(L, xi)), peclet: observe(pe),
    basisShape: observe(wRight), leftWeight: observe(wLeft),
    observedNormalizedShape: contrast === 0 ? null : observe(wRight),
    normalizationStatus: contrast === 0 ? 'zero-boundary-contrast' as const : 'defined' as const,
    stableShapeCorrection: observe(correction),
    stableConcentrationCorrection: observe(multiply(product(contrast), correction)),
    concentration: observe(concentration), gradient: observe(gradient), curvature: observe(curvature),
    advectiveFlux: observe(advective), diffusiveFlux: observe(diffusive), totalFlux: observe(flux),
    totalTransport: observe(multiply(product(A), flux)), roundedTotalFlux: roundedFlux,
    diffusionTime: observe(shift(product(L, L), -Math.log(D))),
    advectionTime: U === 0 ? null : observe(shift(product(L), -Math.log(Math.abs(U)))),
    advectionTimeStatus: U === 0 ? 'absent-zero-velocity' as const : 'defined' as const,
    boundaryLayerLength: U === 0 ? null : observe(shift(product(D), -Math.log(Math.abs(U)))),
    residualMeaning: 'exact-family-observations-not-independent-validation' as const,
  };
}

export interface DamkohlerParameters extends Geometry {
  reactionRate: number;
  surfaceConcentration: number;
}
/** Half-slab: c(0)=c0, c_x(L)=0. The input length is HALF the full symmetric plate thickness. */
export function damkohlerAt(p: DamkohlerParameters, xi: number) {
  geometry(p);
  bound(p.reactionRate, 0, 128, 'first-order reaction rate');
  bound(p.surfaceConcentration, 0, 4, 'surface concentration');
  bound(xi, 0, 1, 'dimensionless position');
  const { diffusivity: D, length: L, area: A, reactionRate: k, surfaceConcentration: c0 } = p;
  const da = shift(product(k, L, L), -Math.log(D));
  const logPhi = da.log / 2, phi = k === 0 ? 0 : Math.exp(logPhi);
  let shape: Q = product(1), depletion: Q = zero, slope: Q = zero;
  let effectiveness: Q = product(1), effectivenessDeficit: Q = zero;
  if (k > 0) {
    const denominator = logCosh(phi), a = phi * (1 - xi);
    if (xi !== 0) {
      depletion = { sign: 1, log: ln2 + logSinh(logPhi + Math.log(1 - xi / 2)) + logSinh(logPhi + Math.log(xi) - ln2) - denominator };
      const d = observe(depletion).value;
      const logShape = d !== null && d < .5 ? Math.log1p(-d) :
        -phi * xi + Math.log1p(Math.exp(-2 * a)) - Math.log1p(Math.exp(-2 * phi));
      shape = { sign: 1, log: logShape };
    }
    if (xi !== 1) {
      const logRatio = phi < 1 ? logSinh(logPhi + Math.log(1 - xi)) - denominator :
        -phi * xi + logOneMinusExpNegative(logPhi + Math.log(2 * (1 - xi))) - Math.log1p(Math.exp(-2 * phi));
      slope = { sign: -1, log: logPhi - Math.log(L) + logRatio };
    }
    if (phi < 1e-3) {
      const daLinear = Math.exp(da.log);
      effectivenessDeficit = shift(da, -Math.log(3) + Math.log1p(-2 * daLinear / 5 + 17 * daLinear ** 2 / 105));
      effectiveness = { sign: 1, log: Math.log1p(-(observe(effectivenessDeficit).value ?? 0)) };
    } else {
      effectiveness = { sign: 1, log: Math.log(Math.tanh(phi)) - logPhi };
      effectivenessDeficit = sum(product(1), { sign: -1, log: effectiveness.log });
    }
  }
  const concentration = multiply(product(c0), shape), gradient = multiply(product(c0), slope);
  const flux = multiply(product(-D), gradient), consumption = multiply(product(k), concentration);
  const mass = multiply(product(A, c0, L), effectiveness);
  const totalConsumption = multiply(product(A, c0, L, k), effectiveness);
  const inletFlux = shift(totalConsumption, -Math.log(A));
  // Integral over [0, xi*L]. A positive hyperbolic-product expression avoids
  // subtracting two nearly equal inlet/interior fluxes at a tiny observation window.
  const partialMass = xi === 0 ? zero : k === 0 ? product(A, c0, L, xi) :
    shift(product(A, c0, L), ln2 + logCosh(phi * (1 - xi / 2)) +
      logSinh(logPhi + Math.log(xi) - ln2) - logPhi - logCosh(phi));
  const partialConsumption = multiply(product(k), partialMass);
  return {
    xi, position: observe(product(L, xi)), damkohler: observe(da),
    thiele: observe(k === 0 ? zero : { sign: 1, log: logPhi }),
    basisShape: observe(shape), stableDepletion: observe(depletion),
    stableConcentrationDepletion: observe(multiply(product(c0), depletion)),
    observedNormalizedShape: c0 === 0 ? null : observe(shape),
    normalizationStatus: c0 === 0 ? 'zero-surface-concentration' as const : 'defined' as const,
    concentration: observe(concentration), gradient: observe(gradient),
    curvature: observe(shift(consumption, -Math.log(D))),
    localConsumption: observe(consumption), diffusiveFlux: observe(flux),
    storedMass: observe(mass), totalConsumption: observe(totalConsumption), inletFlux: observe(inletFlux),
    partialStoredMass: observe(partialMass), partialConsumption: observe(partialConsumption),
    analyticEffectiveness: observe(effectiveness), stableEffectivenessDeficit: observe(effectivenessDeficit),
    observedEffectiveness: k === 0 || c0 === 0 ? null : observe(effectiveness),
    effectivenessStatus: k === 0 || c0 === 0 ? 'zero-ideal-consumption' as const : 'defined' as const,
    diffusionTime: observe(shift(product(L, L), -Math.log(D))),
    reactionTime: k === 0 ? null : observe({ sign: 1, log: -Math.log(k) }),
    penetrationLength: k === 0 ? null : observe({ sign: 1, log: (Math.log(D) - Math.log(k)) / 2 }),
    reactionScaleStatus: k === 0 ? 'absent-zero-reaction' as const : 'defined' as const,
    residualMeaning: 'exact-family-observations-not-independent-validation' as const,
  };
}
