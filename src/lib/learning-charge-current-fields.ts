/** Introductory electrostatics and prescribed charge transport. Browser-safe SI
 * mathematics. A planar display uses the THREE-dimensional Coulomb kernel.
 * The transported Gaussian is not a self-consistent solution of Maxwell's equations.
 */
export type ChargeVector = [number, number];
export interface PointCharge { position: ChargeVector; charge: number; }
/** Deliberately rounded vacuum teaching value, not an exact SI constant. */
export const teachingCoulombConstant = 9e9;
function bound(x: number, lo: number, hi: number, name: string) {
  if (!Number.isFinite(x) || x < lo || x > hi) throw new RangeError(`${name} outside [${lo}, ${hi}].`);
}
function point(p: ChargeVector) {
  if (!Array.isArray(p) || p.length !== 2) throw new RangeError('A planar point needs two coordinates.');
  bound(p[0], -16, 16, 'Position m'); bound(p[1], -16, 16, 'Position m');
}
function sum(values: number[]) {
  let total = 0, correction = 0;
  for (const value of values) { const next = total + value; correction += Math.abs(total) >= Math.abs(value) ? total - next + value : value - next + total; total = next; }
  return total + correction;
}
function finite(x: number) { return Number.isFinite(x) ? x : null; }
function vector(x: number, y: number): ChargeVector | null {
  return Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(Math.hypot(x, y)) ? [x, y] : null;
}
function scaledScalar(value: number, scale: number) {
  const product = value * scale;
  return !Number.isFinite(product) || value !== 0 && scale !== 0 && product === 0 ? null : product;
}
function scaledVector(value: ChargeVector, scale: number): ChargeVector | null {
  const x = scaledScalar(value[0], scale), y = scaledScalar(value[1], scale);
  return x === null || y === null ? null : vector(x, y);
}
export function pointChargeField(sources: PointCharge[], observation: ChargeVector, testCharge = 1e-9, constant = teachingCoulombConstant) {
  if (!Array.isArray(sources) || sources.length > 16) throw new RangeError('Use at most 16 declared point sources.');
  point(observation); bound(testCharge, -4, 4, 'Test charge C'); bound(constant, 1e-12, 1e12, 'Coulomb factor');
  const contributions = sources.map(source => {
    point(source.position); bound(source.charge, -4, 4, 'Source charge C');
    const displacement: ChargeVector = [observation[0] - source.position[0], observation[1] - source.position[1]];
    const distance = Math.hypot(...displacement);
    if (source.charge === 0) return { distance, displacement, status: 'zero-source' as const, fieldOverK: [0, 0] as ChargeVector | null, potentialOverK: 0 as number | null };
    if (distance === 0) return { distance, displacement, status: 'source-singularity' as const, fieldOverK: null as ChargeVector | null, potentialOverK: null as number | null };
    const radial = source.charge / distance / distance;
    const fieldOverK = Number.isFinite(radial) && radial !== 0 ? scaledVector([displacement[0] / distance, displacement[1] / distance], radial) : null;
    const ratio = source.charge / distance;
    const potentialOverK = ratio === 0 ? null : finite(ratio);
    return { distance, displacement, status: fieldOverK && potentialOverK !== null ? 'finite' as const : 'numerical-range' as const, fieldOverK, potentialOverK };
  });
  const singular = contributions.some(c => c.status === 'source-singularity');
  const normalizedField = singular || contributions.some(c => c.fieldOverK === null) ? null
    : vector(sum(contributions.map(c => c.fieldOverK![0])), sum(contributions.map(c => c.fieldOverK![1])));
  const normalizedPotential = singular || contributions.some(c => c.potentialOverK === null) ? null : finite(sum(contributions.map(c => c.potentialOverK!)));
  const field = normalizedField && scaledVector(normalizedField, constant);
  const potential = normalizedPotential === null ? null : scaledScalar(normalizedPotential, constant);
  const force = field && scaledVector(field, testCharge);
  const fieldMagnitude = field ? Math.hypot(...field) : null;
  const fieldDirection: ChargeVector | null = field && fieldMagnitude! > 0 ? [field[0] / fieldMagnitude!, field[1] / fieldMagnitude!] : null;
  const unavailable = singular ? 'source-singularity' as const : 'numerical-range' as const;
  return { observation: [...observation] as ChargeVector, testCharge, constant, contributions,
    fieldOverK: normalizedField, potentialOverK: normalizedPotential, field, potential, force, fieldMagnitude, fieldDirection,
    fieldStatus: field ? 'finite' as const : unavailable,
    potentialStatus: potential === null ? unavailable : 'finite' as const,
    forceStatus: force ? 'finite' as const : unavailable,
    status: singular ? 'source-singularity' as const : field && potential !== null && force ? 'finite' as const : 'numerical-range' as const,
    kernel: 'three-dimensional-coulomb-planar-observation' as const };
}

/** Uniform volume-current flux through a directed patch, not sheet density K
 * or a constitutive law J=sigma E. */
export function directedSurfaceCurrent(currentDensity: ChargeVector, area: number, normalAngle: number) {
  if (!Array.isArray(currentDensity) || currentDensity.length !== 2 || ![currentDensity[0], currentDensity[1]].every(Number.isFinite)) throw new RangeError('Finite current-density vector required.');
  bound(area, 0, 4, 'Surface area m²'); bound(normalAngle, -4 * Math.PI, 4 * Math.PI, 'Normal angle rad');
  const normal: ChargeVector = [Math.cos(normalAngle), Math.sin(normalAngle)];
  const products = [scaledScalar(currentDensity[0], normal[0]), scaledScalar(currentDensity[1], normal[1])];
  const densityThroughNormal = products.includes(null) ? null : finite(sum(products as number[]));
  const current = area === 0 ? 0 : densityThroughNormal === null ? null : scaledScalar(densityThroughNormal, area);
  return { normal, densityThroughNormal, current, area,
    status: current === null || densityThroughNormal === null ? 'numerical-range' as const : 'finite' as const };
}

export interface TransportedGaussianCharge {
  totalCharge: number; sigma: number; initialCenter: number; velocity: number; area: number;
}
function transport(s: TransportedGaussianCharge) {
  bound(s.totalCharge, -4, 4, 'Total charge C'); bound(s.sigma, .1, 2, 'Gaussian width m');
  bound(s.initialCenter, -4, 4, 'Initial center m'); bound(s.velocity, -2, 2, 'Velocity m/s');
  bound(s.area, .1, 4, 'Transport cross-section m²');
}
type TransportValue = number | null;
const transportStatus = (value: TransportValue) => value === null ? 'numerical-range' as const : 'finite' as const;
function transportScale(value: TransportValue, factor: number): TransportValue {
  return factor === 0 || value === 0 ? 0 : value === null ? null : scaledScalar(value, factor);
}
function transportSum(values: TransportValue[]): TransportValue {
  return values.includes(null) ? null : finite(sum(values as number[]));
}
export function transportedChargeObservation(s: TransportedGaussianCharge, x: number, time: number) {
  transport(s); bound(x, -16, 16, 'Observation m'); bound(time, -4, 4, 'Time s');
  const center = s.initialCenter + s.velocity * time, z = (x - center) / s.sigma;
  // Combine the normalization and Gaussian in log space: an underflowed bare
  // exponential need not imply an unavailable normalized density.
  const rawDensity = s.totalCharge === 0 ? 0 : Math.sign(s.totalCharge) * Math.exp(Math.log(Math.abs(s.totalCharge)) - Math.log(s.area) - Math.log(s.sigma) - .5 * Math.log(2 * Math.PI) - z * z / 2);
  const density = rawDensity === 0 && s.totalCharge !== 0 ? null : finite(rawDensity);
  const currentDensity = transportScale(density, s.velocity);
  const spatialDerivative = transportScale(density, -z / s.sigma);
  const timeDerivative = transportScale(spatialDerivative, -s.velocity), currentDivergence = transportScale(spatialDerivative, s.velocity);
  const positiveXCurrent = transportScale(currentDensity, s.area);
  return { x, time, center, z, density, densityStatus: transportStatus(density), currentDensity, currentDensityStatus: transportStatus(currentDensity), positiveXCurrent,
    spatialDerivative, timeDerivative, currentDivergence, localContinuityResidual: transportSum([timeDerivative, currentDivergence]),
    residualMeaning: 'analytic-identity-not-an-independent-numerical-check' as const };
}
// Positive abscissas and weights for Gauss-Legendre integration, orders 16 and 8.
const x16 = [.09501250983763744, .2816035507792589, .4580167776572274, .6178762444026438, .755404408355003, .8656312023878318, .9445750230732326, .9894009349916499];
const w16 = [.1894506104550685, .1826034150449236, .16915651939500254, .14959598881657673, .12462897125553387, .09515851168249278, .06225352393864789, .027152459411754095];
const x8 = [.1834346424956498, .525532409916329, .7966664774136267, .9602898564975363];
const w8 = [.362683783378362, .3137066458778873, .2223810344533745, .1012285362903763];
function quadrature(fn: (x: number) => TransportValue, start: number, width: number, panels: number, nodes: number[], weights: number[]): TransportValue {
  if (width === 0) return 0;
  const step = width / panels, half = step / 2, values: TransportValue[] = [];
  for (let j = 0; j < panels; j++) {
    const center = start + (j + .5) * step;
    for (let i = 0; i < nodes.length; i++) values.push(transportScale(transportScale(transportSum([fn(center - half * nodes[i]), fn(center + half * nodes[i])]), weights[i]), half));
  }
  return transportSum(values);
}
/** Direct density integral avoids subtracting two CDF values close to one.
 * Original physical interval width is retained even if standardized endpoints
 * round to the same value. No tail is clipped or probability renormalized.
 */
export function gaussianChargeInterval(s: TransportedGaussianCharge, left: number, right: number, time: number) {
  transport(s); bound(left, -16, 16, 'Left boundary m'); bound(right, -16, 16, 'Right boundary m'); bound(time, -4, 4, 'Time s');
  if (!(right > left)) throw new RangeError('A fixed control interval requires left < right.');
  const center = s.initialCenter + s.velocity * time, start = (left - center) / s.sigma, width = (right - left) / s.sigma;
  const panels = Math.max(1, Math.ceil(width));
  const pdf = (z: number) => Math.exp(-z * z / 2) / Math.sqrt(2 * Math.PI);
  const rawFraction = quadrature(pdf, start, width, panels, x16, w16);
  const fraction = rawFraction === 0 ? null : rawFraction;
  const coarse = quadrature(pdf, start, width, panels, x8, w8);
  const atLeft = transportedChargeObservation(s, left, time), atRight = transportedChargeObservation(s, right, time);
  const chargeInside = transportScale(fraction, s.totalCharge);
  const leftOutwardCurrent = transportScale(atLeft.positiveXCurrent, -1), rightOutwardCurrent = atRight.positiveXCurrent;
  const netOutflow = transportSum([leftOutwardCurrent, rightOutwardCurrent]);
  return { time, center, left, right, fraction, chargeInside, leftOutwardCurrent, rightOutwardCurrent,
    netOutflow, netInflow: transportScale(netOutflow, -1),
    fractionStatus: fraction === null ? 'numerical-underflow' as const : 'finite' as const,
    chargeInsideStatus: transportStatus(chargeInside),
    quadratureDifference: fraction === null || coarse === null || coarse === 0 ? null : Math.abs(fraction - coarse), quadraturePanels: panels,
    quadratureMeaning: 'order-comparison-diagnostic-not-certified-error-bound' as const };
}
export function transportedChargeBalance(s: TransportedGaussianCharge, left: number, right: number, startTime: number, endTime: number) {
  const initial = gaussianChargeInterval(s, left, right, startTime), final = gaussianChargeInterval(s, left, right, endTime);
  const duration = endTime - startTime, panels = Math.max(1, Math.ceil(Math.abs(duration * s.velocity / s.sigma)));
  const inflow = (t: number) => transportScale(transportSum([transportedChargeObservation(s, left, t).currentDensity, transportScale(transportedChargeObservation(s, right, t).currentDensity, -1)]), s.area);
  const integratedNetInflow = quadrature(inflow, startTime, duration, panels, x16, w16);
  const coarse = quadrature(inflow, startTime, duration, panels, x8, w8);
  const chargeChange = duration === 0 || s.velocity === 0 || s.totalCharge === 0 ? 0 : transportSum([final.chargeInside, transportScale(initial.chargeInside, -1)]);
  return { initial, final, startTime, endTime, chargeChange, integratedNetInflow,
    balanceResidual: transportSum([chargeChange, transportScale(integratedNetInflow, -1)]), flowQuadratureDifference: integratedNetInflow === null || coarse === null ? null : Math.abs(integratedNetInflow - coarse),
    model: 'prescribed-one-dimensional-charge-transport' as const };
}
