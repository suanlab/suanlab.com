/** Periodic two-mode forced Burgers family with an exact Gaussian spatial filter. */
import type { TurbulenceObservation } from './learning-continuum-turbulence';
interface Q { sign: number; log: number }
const zero: Q = { sign: 0, log: 0 }, ln2 = Math.log(2);
function product(...v: number[]): Q { return v.includes(0) ? zero : { sign: v.reduce((s, x) => s * Math.sign(x), 1), log: v.reduce((s, x) => s + Math.log(Math.abs(x)), 0) }; }
const neg = (a: Q): Q => a.sign === 0 ? zero : { sign: -a.sign, log: a.log };
const mul = (a: Q, b: Q): Q => a.sign === 0 || b.sign === 0 ? zero : { sign: a.sign * b.sign, log: a.log + b.log };
const scale = (a: Q, ...v: number[]): Q => mul(a, product(...v));
const square = (a: Q) => mul(a, a);
const shift = (a: Q, log: number): Q => a.sign === 0 ? zero : { sign: a.sign, log: a.log + log };
function sum(...terms: Q[]): Q {
  let result = zero;
  for (const a of terms) {
    if (!a.sign) continue;
    if (!result.sign) { result = a; continue; }
    const hi = result.log >= a.log ? result : a, lo = result.log >= a.log ? a : result;
    result = hi.sign === lo.sign ? { sign: hi.sign, log: hi.log + Math.log1p(Math.exp(lo.log - hi.log)) }
      : hi.log === lo.log ? zero : { sign: hi.sign, log: hi.log + Math.log(-Math.expm1(lo.log - hi.log)) };
  }
  return result;
}
function obs(a: Q): TurbulenceObservation {
  if (!a.sign) return { value: 0, sign: 0, logMagnitude: null };
  const n = a.log <= Math.log(Number.MIN_VALUE) - ln2 ? 0 : a.sign * Math.exp(a.log);
  return { value: n === 0 || !Number.isFinite(n) ? null : n, sign: a.sign, logMagnitude: a.log };
}
function bound(n: number, low: number, high: number, name: string) { if (!Number.isFinite(n) || n < low || n > high) throw new RangeError(`Invalid ${name}`); }
/** 1-exp(-z), with z represented before products can underflow. */
function loss(z: Q): Q {
  if (!z.sign) return zero;
  return { sign: 1, log: z.log < -36 ? z.log : Math.log(-Math.expm1(-Math.exp(z.log))) };
}
function attenuation(z: Q) { return z.sign ? -Math.exp(z.log) : 0; }
function cycleTrig(cycles: number) {
  const f = cycles % 1;
  if (f === .125 || f === .375 || f === .625 || f === .875) { const q = product(Math.SQRT1_2); return { sin: f < .5 ? q : neg(q), cos: f < .25 || f > .75 ? q : neg(q) }; }
  const quadrant = Math.round(f * 4), delta = f - quadrant / 4, angle = 2 * Math.PI * delta;
  const sin = Math.abs(delta) < 1e-8 ? product(delta, 2 * Math.PI, 1 - angle * angle / 6) : product(Math.sin(angle)), cos = product(Math.cos(angle));
  switch ((quadrant + 4) % 4) { case 0: return { sin, cos }; case 1: return { sin: cos, cos: neg(sin) }; case 2: return { sin: neg(sin), cos: neg(cos) }; default: return { sin: neg(cos), cos: sin }; }
}
function angleTrig(angle: Q) {
  if (!angle.sign) return { sin: zero, cos: product(1) };
  const x = angle.sign * Math.exp(angle.log);
  return angle.log < Math.log(1e-8) ? { sin: scale(angle, 1 - x * x / 6), cos: product(1) } : { sin: product(Math.sin(x)), cos: product(Math.cos(x)) };
}
function compose(a: { sin: Q; cos: Q }, b: { sin: Q; cos: Q }) {
  return { sin: sum(mul(a.sin, b.cos), mul(a.cos, b.sin)), cos: sum(mul(a.cos, b.cos), neg(mul(a.sin, b.sin))) };
}
export interface GaussianBurgersParameters {
  wavenumber: number; filterWidth: number; kinematicViscosity: number;
  initialFirstAmplitude: number; initialSecondAmplitude: number;
  density: number; area: number; frameVelocity: number;
}
function validate(p: GaussianBurgersParameters, time: number) {
  bound(p.wavenumber, .25, 4, 'positive wavenumber'); bound(p.filterWidth, 0, 4, 'Gaussian standard deviation'); bound(p.kinematicViscosity, 0, 2, 'molecular kinematic viscosity');
  bound(p.initialFirstAmplitude, -2, 2, 'first velocity amplitude'); bound(p.initialSecondAmplitude, -2, 2, 'second velocity amplitude');
  bound(p.density, .25, 4, 'density'); bound(p.area, .25, 4, 'area'); bound(p.frameVelocity, -2, 2, 'Galilean frame velocity'); bound(time, 0, 4, 'time');
}
function modalState(p: GaussianBurgersParameters, time: number) {
  const k = p.wavenumber, d = product(p.kinematicViscosity, k, k, time), t = product(k, p.filterWidth, k, p.filterWidth);
  const A = shift(product(p.initialFirstAmplitude), attenuation(d)), B = shift(product(p.initialSecondAmplitude), attenuation(scale(d, 4)));
  const g = (n: number): Q => ({ sign: 1, log: attenuation(scale(t, n * n / 2)) });
  /** A ca+B cb after 0, 1 or 2 filter applications, retaining differential damping. */
  function linear(ca: Q, cb: Q, filters = 0) {
    const common = sum(d, scale(t, filters / 2)), difference = sum(scale(d, 3), scale(t, 1.5 * filters));
    if (difference.sign && difference.log > Math.log(.25)) return sum(shift(scale(ca, p.initialFirstAmplitude), attenuation(common)), shift(scale(cb, p.initialSecondAmplitude), attenuation(common) + attenuation(difference)));
    // Exact binary power-of-two initial combinations must cancel before taking logarithms.
    const power = cb.sign ? Math.round(cb.log / ln2) : -1;
    const exactInitial = ca.sign === 1 && ca.log === 0 && power >= 0 && power <= 3 && cb.log === power * ln2;
    const initial = exactInitial ? product(p.initialFirstAmplitude + cb.sign * 2 ** power * p.initialSecondAmplitude)
      : sum(scale(ca, p.initialFirstAmplitude), scale(cb, p.initialSecondAmplitude));
    return shift(sum(initial, neg(mul(scale(cb, p.initialSecondAmplitude), loss(difference)))), attenuation(common));
  }
  return { k, d, t, A, B, g, linear };
}
/** xCycles=x/L and phaseCycles are exact declared cycle fractions; moving coordinate is x-Ct. */
export function gaussianBurgersAt(p: GaussianBurgersParameters, time: number, xCycles: number, phaseCycles = 0) {
  validate(p, time); bound(xCycles, 0, 1, 'position cycle fraction'); bound(phaseCycles, 0, 1, 'initial phase cycle fraction');
  const s = modalState(p, time), { k, A, B, t, d, g, linear } = s, C = p.frameVelocity, nu = p.kinematicViscosity;
  const scaledCycle = (n: number, x: number) => x !== 0 && Math.abs(n * x) < 1e-8 ? angleTrig(product(2 * Math.PI, n, x)) : cycleTrig(n * x);
  const harmonic = (n: number) => compose(compose(scaledCycle(n, xCycles), scaledCycle(n, phaseCycles)), angleTrig(product(-n, k, C, time)));
  const h = [harmonic(1), harmonic(2), harmonic(3), harmonic(4)], sin = h[0].sin, cos = h[0].cos, sin2 = h[1].sin, cos2 = h[1].cos;
  const half = harmonic(.5), threeHalf = harmonic(1.5), reference = (obs(cos).value ?? 0) >= 0 ? 1 : -1;
  const cosineChange = reference === 1 ? scale(square(half.sin), -2) : scale(square(half.cos), 2);
  const differenceCosines = reference === 1 ? scale(mul(threeHalf.sin, half.sin), -2) : scale(mul(threeHalf.cos, half.cos), 2);
  function sinePair(second: number, filters = 0) {
    if (Math.abs(obs(cos).value ?? 0) < .5) return mul(sin, linear(product(1), scale(cos, 2, second), filters));
    return mul(sin, sum(linear(product(1), product(2, second, reference), filters), scale(mul(mul(B, filters ? { sign: 1, log: attenuation(scale(t, 2 * filters)) } : product(1)), cosineChange), 2, second)));
  }
  function cosinePair(second: number, filters = 0) {
    if (!cos.sign || !cos2.sign) return linear(cos, scale(cos2, second), filters);
    const filteredB = filters ? mul(B, { sign: 1, log: attenuation(scale(t, 2 * filters)) }) : B;
    return sum(mul(cos, linear(product(1), product(second, reference), filters)), scale(mul(filteredB, differenceCosines), second));
  }
  const v = sinePair(1), vx = scale(cosinePair(2), k), vxx = scale(sinePair(4), -k, k);
  const w = sinePair(1, 1), wx = scale(cosinePair(2, 1), k), wxx = scale(sinePair(4, 1), -k, k);
  const u = sum(product(C), v), filtered = sum(product(C), w), ut = sum(scale(vxx, nu), scale(vx, -C)), wt = sum(scale(wxx, nu), scale(wx, -C));
  const one = loss(t), z = Math.exp(attenuation(t));
  const Ve = scale(square(one), .5), Vo = scale(loss(scale(t, 2)), .5);
  const De = scale(mul(square(square(one)), product((1 + z) ** 2, 1 + z + z * z)), .5);
  const Do = scale(mul(mul(square(one), one), product(1 + z, 1 + z * z, 1 + z + z * z)), .5);
  const reChange = neg(sum(loss(scale(t, .5)), scale(loss(scale(t, 1.5)), 2), loss(scale(t, 2.5))));
  const roChange = neg(sum(loss(scale(t, .5)), loss(scale(t, 2.5))));
  const e = sum(sinePair(4), mul(mul(B, sin2), reChange));
  const o = sum(cosinePair(2), mul(mul(B, cos2), roChange));
  const ex = scale(sum(cosinePair(8), scale(mul(mul(B, cos2), reChange), 2)), k);
  const ox = scale(sum(sinePair(4), scale(mul(mul(B, sin2), roChange), 2)), -k);
  const bs = mul(B, sin2), bc = mul(B, cos2);
  const even = mul(Ve, square(e)), evenRemainder = mul(De, square(bs)), odd = mul(Vo, square(o)), oddRemainder = mul(Do, square(bc));
  const tau = sum(even, evenRemainder, odd, oddRemainder);
  const gramTauX = scale(sum(mul(Ve, mul(e, ex)), scale(mul(De, mul(bs, bc)), 2, k), mul(Vo, mul(o, ox)), scale(mul(Do, mul(bc, bs)), -2, k)), 2);
  const fourierCoefficients = [
    mul(mul(A, B), mul(g(1), loss(scale(t, 2)))),
    scale(mul(square(A), mul({ sign: 1, log: attenuation(t) }, loss(t))), .5),
    mul(mul(A, B), mul({ sign: 1, log: attenuation(scale(t, 2.5)) }, loss(scale(t, 2)))),
    scale(mul(square(B), mul({ sign: 1, log: attenuation(scale(t, 4)) }, loss(scale(t, 4)))), .5),
  ];
  const fourierTauX = scale(sum(...fourierCoefficients.map((c, i) => scale(mul(c, h[i].sin), i + 1))), -k);
  // Positive-variance derivatives retain small-width cancellations; Fourier coefficients retain broad-filter tails.
  const broadFilter = t.sign !== 0 && t.log > Math.log(.25);
  const tauX = broadFilter ? fourierTauX : gramTauX;
  // A direct Fourier-force diagnostic is retained; signed-log moment form below is the stable observable.
  const forcing = mul(v, vx);
  const directFourierForce = scale(sum(scale(mul(square(A), mul(g(2), sin2)), .5), scale(mul(mul(A, B), sum(scale(mul(g(3), h[2].sin), 3), neg(mul(g(1), sin)))), .5), mul(square(B), mul(g(4), h[3].sin))), k);
  // This exact moment identity stabilizes small-width differences; it is checked against independent force convolution.
  const filteredForcing = broadFilter ? directFourierForce : sum(mul(w, wx), scale(tauX, .5));
  const transfer = scale(mul(tau, wx), -.5), lossDensity = scale(square(wx), nu), forceWork = mul(filtered, filteredForcing);
  const flux = sum(scale(mul(square(filtered), filtered), 1 / 3), scale(mul(filtered, wx), -nu), scale(mul(filtered, tau), .5));
  const fluxX = sum(mul(square(filtered), wx), scale(sum(square(wx), mul(filtered, wxx)), -nu), scale(sum(mul(wx, tau), mul(filtered, tauX)), .5));
  const residualRatioChange = neg(sum(loss(scale(t, .5)), loss(t), loss(scale(t, 1.5))));
  const smallWidthResidual = mul(loss(scale(t, .5)), sum(sinePair(4), mul(mul(B, sin2), residualRatioChange)));
  const repeatedRatioChange = neg(sum(loss(scale(t, 1.5)), loss(scale(t, 2)), loss(scale(t, 2.5)), loss(scale(t, 3))));
  const smallWidthFilteredResidual = mul(mul(g(1), loss(scale(t, .5))), sum(sinePair(4), mul(mul(B, sin2), repeatedRatioChange)));
  const residual = broadFilter ? sum(v, neg(w)) : smallWidthResidual;
  const filteredResidual = broadFilter ? sum(w, neg(sinePair(1, 2))) : smallWidthFilteredResidual;
  const filteredSquare = sum(square(filtered), tau), first = obs(filteredSquare).value, second = obs(filtered).value;
  return { time, xCycles, phaseCycles, cellLength: 2 * Math.PI / k,
    firstAmplitude: obs(A), secondAmplitude: obs(B), firstAmplitudeChange: obs(neg(mul(product(p.initialFirstAmplitude), loss(d)))), secondAmplitudeChange: obs(neg(mul(product(p.initialSecondAmplitude), loss(scale(d, 4))))),
    firstTransferFunction: obs(g(1)), secondTransferFunction: obs(g(2)),
    velocity: obs(u), velocityDeparture: obs(v), velocityGradient: obs(vx), velocityCurvature: obs(vxx), velocityTimeRate: obs(ut),
    filteredVelocity: obs(filtered), filteredDeparture: obs(w), filteredGradient: obs(wx), filteredCurvature: obs(wxx), filteredTimeRate: obs(wt), twiceFilteredVelocity: obs(sum(product(C), sinePair(1, 2))),
    residualVelocity: obs(residual), filteredResidualVelocity: obs(filteredResidual), variance: obs(tau), varianceGradient: obs(tauX), filteredVelocitySquare: obs(filteredSquare),
    positiveVarianceTerms: [even, evenRemainder, odd, oddRemainder].map(obs), ordinarySubtractionVariance: first === null || second === null ? null : first - second * second,
    force: obs(forcing), filteredForce: obs(filteredForcing), directFourierFilteredForce: obs(directFourierForce), resolvedNonlinearFluxGradient: obs(mul(filtered, wx)), filteredNonlinearFluxGradient: obs(sum(filteredForcing, scale(wx, C))), subgridForce: obs(scale(tauX, -.5)),
    transferPerMass: obs(transfer), resolvedMolecularLossPerMass: obs(lossDensity), filteredForcingWorkPerMass: obs(forceWork), resolvedEnergyPerMass: obs(scale(square(filtered), .5)), residualEnergyPerMass: obs(scale(tau, .5)),
    resolvedEnergyRatePerMass: obs(mul(filtered, wt)), resolvedEnergyFlux: obs(flux), resolvedEnergyFluxGradient: obs(fluxX) };
}
/** Full-cell J and W, including density and cross-sectional area. */
export function gaussianBurgersBudgets(p: GaussianBurgersParameters, time: number) {
  validate(p, time); const { k, t, A, B, g } = modalState(p, time), L = 2 * Math.PI / k, factor = p.density * p.area * L / 4;
  const a2 = square(A), b2 = square(B), fa2 = mul(square(g(1)), a2), fb2 = mul(square(g(2)), b2);
  const ra2 = mul(loss(t), a2), rb2 = mul(loss(scale(t, 4)), b2), background = product(2, factor, p.frameVelocity, p.frameVelocity);
  const total = sum(background, scale(sum(a2, b2), factor)), resolved = sum(background, scale(sum(fa2, fb2), factor)), residual = scale(sum(ra2, rb2), factor);
  const D = scale(sum(a2, scale(b2, 4)), 2, factor, p.kinematicViscosity, k, k), Dr = scale(sum(fa2, scale(fb2, 4)), 2, factor, p.kinematicViscosity, k, k), Ds = scale(sum(ra2, scale(rb2, 4)), 2, factor, p.kinematicViscosity, k, k);
  const P = scale(mul(mul(a2, B), mul(product(Math.exp(attenuation(t))), loss(scale(t, 3)))), -factor, k);
  return { time, cellLength: L, cellVolume: L * p.area, totalKineticEnergy: obs(total), resolvedKineticEnergy: obs(resolved), residualKineticEnergy: obs(residual),
    totalMolecularLoss: obs(D), resolvedMolecularLoss: obs(Dr), residualMolecularLoss: obs(Ds), totalEnergyRate: obs(neg(D)), resolvedEnergyRate: obs(neg(Dr)), residualEnergyRate: obs(neg(Ds)),
    totalForcingWork: obs(zero), resolvedForcingWork: obs(P), residualForcingWork: obs(neg(P)), resolvedToResidualTransfer: obs(P) };
}
