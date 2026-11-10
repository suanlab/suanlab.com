/** Manufactured, steady, forced incompressible channel ensemble; not a turbulence DNS. */
export interface TurbulenceObservation { value: number | null; sign: number; logMagnitude: number | null }
interface Q { sign: number; log: number }
const zero: Q = { sign: 0, log: 0 };
const ln2 = Math.log(2);
function product(...factors: number[]): Q {
  return factors.includes(0) ? zero : { sign: factors.reduce((s, v) => s * Math.sign(v), 1), log: factors.reduce((s, v) => s + Math.log(Math.abs(v)), 0) };
}
const negate = (a: Q): Q => a.sign === 0 ? zero : { sign: -a.sign, log: a.log };
const multiply = (a: Q, b: Q): Q => a.sign === 0 || b.sign === 0 ? zero : { sign: a.sign * b.sign, log: a.log + b.log };
const scale = (a: Q, ...factors: number[]): Q => multiply(a, product(...factors));
const square = (a: Q): Q => multiply(a, a);
function sum(...terms: Q[]): Q {
  let result = zero;
  for (const term of terms) {
    if (term.sign === 0) continue;
    if (result.sign === 0) { result = term; continue; }
    const high = result.log >= term.log ? result : term, low = result.log >= term.log ? term : result;
    if (high.sign === low.sign) result = { sign: high.sign, log: high.log + Math.log1p(Math.exp(low.log - high.log)) };
    else if (high.log === low.log) result = zero;
    else result = { sign: high.sign, log: high.log + Math.log(-Math.expm1(low.log - high.log)) };
  }
  return result;
}
function observe(a: Q): TurbulenceObservation {
  if (a.sign === 0) return { value: 0, sign: 0, logMagnitude: null };
  const value = a.log <= Math.log(Number.MIN_VALUE) - ln2 ? 0 : a.sign * Math.exp(a.log);
  return { value: value === 0 || !Number.isFinite(value) ? null : value, sign: a.sign, logMagnitude: a.log };
}
function bound(value: number, low: number, high: number, name: string) {
  if (!Number.isFinite(value) || value < low || value > high) throw new RangeError(`Invalid ${name}`);
}
export interface ReynoldsEnsembleParameters {
  height: number; wavenumber: number; spanWidth: number; density: number; dynamicViscosity: number;
  bulkVelocity: number; meanAsymmetry: number; firstStreamfunctionAmplitude: number; secondStreamfunctionAmplitude: number;
  candidateEddyViscosity: number;
}
function validate(p: ReynoldsEnsembleParameters) {
  bound(p.height, .25, 4, 'channel height'); bound(p.wavenumber, .25, 4, 'streamwise wavenumber'); bound(p.spanWidth, .25, 4, 'span width');
  bound(p.density, .25, 4, 'density'); bound(p.dynamicViscosity, 0, 2, 'dynamic viscosity'); bound(p.bulkVelocity, -2, 2, 'bulk velocity');
  bound(p.meanAsymmetry, -1, 1, 'mean asymmetry'); bound(p.firstStreamfunctionAmplitude, -2, 2, 'first streamfunction amplitude');
  bound(p.secondStreamfunctionAmplitude, -2, 2, 'second streamfunction amplitude'); bound(p.candidateEddyViscosity, 0, 4, 'candidate kinematic eddy viscosity');
}
/** Factored polynomial jets keep nonzero near-wall fields available as signed logs. */
function jets(p: ReynoldsEnsembleParameters, xi: number) {
  const H = p.height, c = p.meanAsymmetry, Ub = p.bulkVelocity, d = 2 * xi - 1, wall = 1 - xi;
  const phi = product(xi, xi, wall, wall), phiY = product(2, xi, wall, 1 - 2 * xi, 1 / H);
  const phiYY = product(2 - 12 * xi + 12 * xi * xi, 1 / H, 1 / H), phiYYY = product(24 * xi - 12, 1 / H, 1 / H, 1 / H);
  const chi = scale(phi, d), chiY = sum(scale(phiY, d), scale(phi, 2 / H));
  const chiYY = sum(scale(phiYY, d), scale(phiY, 4 / H)), chiYYY = sum(scale(phiYYY, d), scale(phiYY, 6 / H));
  const meanFactor = sum(product(1 - c, wall), product(1 + c, xi));
  const mean = scale(meanFactor, 6, Ub, xi, wall);
  const shear = sum(scale(meanFactor, 6, Ub, 1 - 2 * xi, 1 / H), product(12, Ub, c, xi, wall, 1 / H));
  const curvature = product(-12, Ub, 1 + 3 * c * d, 1 / H, 1 / H);
  return { phi, phiY, phiYY, phiYYY, chi, chiY, chiYY, chiYYY, mean, shear, curvature };
}
/** Fractions of a cycle: exact quarter-cycle inputs have exact sine/cosine zeros. */
function trig(cycles: number) {
  const f = cycles % 1;
  if (f === .125 || f === .375 || f === .625 || f === .875) {
    const t = product(Math.SQRT1_2);
    return { sin: f < .5 ? t : negate(t), cos: f < .25 || f > .75 ? t : negate(t) };
  }
  const quadrant = Math.round(4 * f), delta = f - quadrant / 4;
  const angle = 2 * Math.PI * delta;
  const sin = Math.abs(delta) < 1e-8 ? product(delta, 2 * Math.PI, 1 - angle * angle / 6) : product(Math.sin(angle));
  const cos = product(Math.cos(angle));
  switch ((quadrant + 4) % 4) {
    case 0: return { sin, cos };
    case 1: return { sin: cos, cos: negate(sin) };
    case 2: return { sin: negate(sin), cos: negate(cos) };
    default: return { sin: negate(cos), cos: sin };
  }
}
function instantaneous(p: ReynoldsEnsembleParameters, xi: number, xCycles: number, phaseCycles: number) {
  const r = jets(p, xi), tx = trig(xCycles), tp = trig(phaseCycles);
  const sin = sum(multiply(tx.sin, tp.cos), multiply(tx.cos, tp.sin));
  const cos = sum(multiply(tx.cos, tp.cos), negate(multiply(tx.sin, tp.sin)));
  const a = p.firstStreamfunctionAmplitude, b = p.secondStreamfunctionAmplitude;
  const k = p.wavenumber, rho = p.density, mu = p.dynamicViscosity;
  // Combine the polynomial coefficients before near-wall leading terms can cancel.
  const eta = Math.min(xi, 1 - xi), side = xi <= .5 ? 1 : -1;
  function combination(A: Q, B: Q, order: 0 | 1 | 2 | 3): Q {
    if (xi === .5) {
      const first = [r.phi, r.phiY, r.phiYY, r.phiYYY][order], second = [r.chi, r.chiY, r.chiYY, r.chiYYY][order];
      return sum(multiply(A, first), multiply(B, second));
    }
    if (order === 0) return multiply(r.phi, sum(A, scale(B, -side), scale(B, 2, side, eta)));
    if (order === 1) return scale(sum(sum(scale(A, side), negate(B)), scale(sum(scale(A, -2, side), scale(B, 5)), eta), scale(B, -5, eta, eta)), 2, xi, 1 - xi, 1 / p.height);
    if (order === 2) return scale(sum(sum(scale(A, 2), scale(B, -2, side)), scale(sum(scale(A, -12), scale(B, 24, side)), eta), scale(sum(scale(A, 12), scale(B, -60, side)), eta, eta), scale(B, 40, side, eta, eta, eta)), 1 / p.height, 1 / p.height);
    return scale(sum(sum(scale(A, -12, side), scale(B, 24)), scale(sum(scale(A, 24, side), scale(B, -120)), eta), scale(B, 120, eta, eta)), 1 / p.height, 1 / p.height, 1 / p.height);
  }
  const as = scale(sin, a), bc = scale(cos, b), ac = scale(cos, a), bs = scale(sin, b);
  const uPrime = combination(as, bc, 1), v = combination(scale(ac, -k), scale(bs, k), 0), u = sum(r.mean, uPrime);
  const ux = combination(scale(ac, k), scale(bs, -k), 1), uy = sum(r.shear, combination(as, bc, 2));
  const vx = combination(scale(as, k, k), scale(bc, k, k), 0), vy = negate(ux);
  const lapU = sum(r.curvature, combination(as, bc, 3), scale(uPrime, -k, k));
  const lapV = sum(combination(scale(ac, -k), scale(bs, k), 2), scale(v, -k, k));
  const forceX = sum(scale(sum(multiply(u, ux), multiply(v, uy)), rho), scale(lapU, -mu));
  const forceY = sum(scale(sum(multiply(u, vx), multiply(v, vy)), rho), scale(lapV, -mu));
  return { u, v, uPrime, ux, uy, vx, vy, lapU, lapV, forceX, forceY };
}
/** SI: velocity m/s, force per volume N/m³. xCycles and phaseCycles are declared coordinates, not rounded radian multiples. */
export function reynoldsInstantaneousAt(p: ReynoldsEnsembleParameters, xi: number, xCycles = 0, phaseCycles = 0) {
  validate(p); bound(xi, 0, 1, 'wall fraction'); bound(xCycles, 0, 1, 'streamwise cycle fraction'); bound(phaseCycles, 0, 1, 'ensemble phase fraction');
  const r = instantaneous(p, xi, xCycles, phaseCycles);
  return { xi, xCycles, phaseCycles, u: observe(r.u), v: observe(r.v), uPrime: observe(r.uPrime),
    derivativeUX: observe(r.ux), derivativeUY: observe(r.uy), derivativeVX: observe(r.vx), derivativeVY: observe(r.vy),
    laplacianU: observe(r.lapU), laplacianV: observe(r.lapV), forceX: observe(r.forceX), forceY: observe(r.forceY) };
}
export function reynoldsEnsembleAt(p: ReynoldsEnsembleParameters, xi: number) {
  validate(p); bound(xi, 0, 1, 'wall fraction');
  const r = jets(p, xi), a = p.firstStreamfunctionAmplitude, b = p.secondStreamfunctionAmplitude, k = p.wavenumber, H = p.height;
  const rho = p.density, mu = p.dynamicViscosity, nu = p.candidateEddyViscosity;
  const Ruu = scale(sum(square(scale(r.phiY, a)), square(scale(r.chiY, b))), .5);
  const Rvv = scale(sum(square(scale(r.phi, a, k)), square(scale(r.chi, b, k))), .5);
  const Ruv = scale(square(r.phi), -a, b, k, 1 / H);
  const RuvY = scale(multiply(r.phi, r.phiY), -2, a, b, k, 1 / H);
  const RvvY = scale(sum(scale(multiply(r.phi, r.phiY), a, a), scale(multiply(r.chi, r.chiY), b, b)), k, k);
  const gram = sum(scale(multiply(r.phi, r.phiY), a, a), scale(multiply(r.chi, r.chiY), b, b));
  const determinant = scale(square(gram), .25, k, k);
  const forceX = sum(scale(RuvY, rho), scale(r.curvature, -mu)), forceY = scale(RvvY, rho);
  const modeledRuv = scale(r.shear, -nu), covarianceDefect = sum(Ruv, negate(modeledRuv));
  const modeledForceX = sum(scale(r.curvature, -mu), scale(r.curvature, -rho, nu));
  const forceDefect = sum(scale(RuvY, rho), scale(r.curvature, rho, nu));
  const molecularStress = scale(r.shear, mu), reynoldsShearStress = scale(Ruv, -rho);
  const production = scale(multiply(Ruv, r.shear), -rho), molecularLoss = scale(square(r.shear), mu);
  const modeledProduction = scale(square(r.shear), rho, nu);
  const inferredStatus = r.shear.sign === 0 ? Ruv.sign === 0 ? 'indeterminate' : 'nonzero-stress-at-zero-shear' : 'defined';
  const inferred = inferredStatus === 'defined' ? observe(Ruv.sign === 0 ? zero : { sign: -Ruv.sign * r.shear.sign, log: Ruv.log - r.shear.log }) : null;
  const reynoldsStatus = mu === 0 ? p.bulkVelocity === 0 ? 'indeterminate' : 'inviscid-nonzero-bulk' : 'defined';
  const reynoldsNumber = reynoldsStatus === 'defined' ? observe(p.bulkVelocity === 0 ? zero : { sign: 1, log: Math.log(rho) + Math.log(Math.abs(p.bulkVelocity)) + Math.log(H) - Math.log(mu) }) : null;
  return { xi, position: observe(product(xi, H)), meanU: observe(r.mean), meanV: observe(zero), meanShear: observe(r.shear), meanCurvature: observe(r.curvature),
    covarianceUU: observe(Ruu), covarianceVV: observe(Rvv), covarianceUV: observe(Ruv), covarianceUVGradient: observe(RuvY), covarianceVVGradient: observe(RvvY),
    covarianceDeterminant: observe(determinant), fluctuationEnergyPerMass: observe(scale(sum(Ruu, Rvv), .5)),
    meanForceX: observe(forceX), meanForceY: observe(forceY), molecularShearStress: observe(molecularStress), reynoldsShearStress: observe(reynoldsShearStress),
    totalMeanShearStress: observe(sum(molecularStress, reynoldsShearStress)), modeledCovarianceUV: observe(modeledRuv), covarianceDefect: observe(covarianceDefect),
    modeledForceX: observe(modeledForceX), forceDefect: observe(forceDefect), modeledTotalShearStress: observe(sum(molecularStress, scale(r.shear, rho, nu))),
    meanForcingWorkDensity: observe(multiply(forceX, r.mean)), molecularMeanLossDensity: observe(molecularLoss), meanToFluctuationTransferDensity: observe(production),
    modeledTransferDensity: observe(modeledProduction), forcingWorkDefectDensity: observe(multiply(forceDefect, r.mean)),
    inferredEddyViscosity: inferred, inferredEddyViscosityStatus: inferredStatus, reynoldsNumber, reynoldsNumberStatus: reynoldsStatus };
}
/** Integrals over the full streamwise period, channel height and declared span width. Powers are W, energies J. */
export function reynoldsEnsembleBudgets(p: ReynoldsEnsembleParameters) {
  validate(p);
  const H = p.height, k = p.wavenumber, W = p.spanWidth, rho = p.density, mu = p.dynamicViscosity;
  const Ub = p.bulkVelocity, c = p.meanAsymmetry, a = p.firstStreamfunctionAmplitude, b = p.secondStreamfunctionAmplitude, nu = p.candidateEddyViscosity;
  const L = 2 * Math.PI / k, area = L * W;
  const shearIntegral = product(Ub, Ub, 1 / H, 12 + 36 * c * c / 5, area);
  const viscous = scale(shearIntegral, mu), modeledTransfer = scale(shearIntegral, rho, nu);
  const production = product(rho, a, b, k, Ub, c, 4 / 1155, 1 / H, area);
  const meanEnergy = product(.5, rho, Ub, Ub, H, 6 / 5 + 6 * c * c / 35, area);
  const fluctuationEnergy = product(.25, rho, area);
  const fluctuationIntegral = sum(product(a, a, 2 / (105 * H) + k * k * H / 630), product(b, b, 2 / (315 * H) + k * k * H / 6930));
  return { cellLength: L, cellVolume: area * H, ensemblePhaseCount: 8,
    meanKineticEnergy: observe(meanEnergy), fluctuationKineticEnergy: observe(multiply(fluctuationEnergy, fluctuationIntegral)),
    molecularMeanLoss: observe(viscous), meanToFluctuationTransfer: observe(production), meanForcingWork: observe(sum(viscous, production)),
    modeledTransfer: observe(modeledTransfer), modeledForcingWork: observe(sum(viscous, modeledTransfer)), forcingWorkDefect: observe(sum(production, negate(modeledTransfer))) };
}
