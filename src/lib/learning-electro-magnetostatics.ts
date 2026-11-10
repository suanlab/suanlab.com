/** Static prescribed sources in vacuum. Full 3D vector observations; no force
 * equilibrium, conductivity law, time evolution, or finite return circuit.
 * Integral identities are analytic diagnostics, not independent certificates.
 */
import {teachingCoulombConstant} from './learning-charge-current-fields';
export type StaticVector = [number, number, number];
/** Rounded teaching approximation, not an exact constant of modern SI. */
export const teachingVacuumPermeability = 4 * Math.PI * 1e-7;
export interface UniformChargedSphere {
  radius: number; totalCharge: number; coulombConstant?: number; backgroundField?: StaticVector;
}
export interface UniformSteadyCylinder {
  radius: number; totalCurrent: number; permeability?: number; backgroundField?: StaticVector;
}
type ScalarState = 'finite' | 'numerical-range';
function bound(value: number, lo: number, hi: number, name: string) {
  if (!Number.isFinite(value) || value < lo || value > hi) throw new RangeError(`${name} outside [${lo}, ${hi}].`);
}
function vec(v: StaticVector, limit: number, name: string) {
  if (!Array.isArray(v) || v.length !== 3) throw new RangeError(`${name} requires three components.`);
  for (let i = 0; i < 3; i++) bound(v[i], -limit, limit, name);
}
function scalar(raw: number, exactZero: boolean): {value: number | null; status: ScalarState} {
  const available = Number.isFinite(raw) && (raw !== 0 || exactZero);
  return {value: available ? (raw === 0 ? 0 : raw) : null, status: available ? 'finite' : 'numerical-range'};
}
function sourceVector(raw: StaticVector, exactZero: boolean): StaticVector | null {
  if (exactZero) return [0, 0, 0];
  const magnitude = Math.hypot(...raw);
  return Number.isFinite(magnitude) && (magnitude !== 0 || exactZero) ? raw : null;
}
function add(a: StaticVector | null, b: StaticVector): StaticVector | null {
  return a === null ? null : a.map((x, i) => x + b[i]) as StaticVector;
}
function direction(v: StaticVector | null): StaticVector | null {
  const length = v === null ? 0 : Math.hypot(...v);
  return v === null || length === 0 ? null : v.map(x => x / length) as StaticVector;
}
const region = (r: number, a: number) => r < a ? 'interior' as const : r > a ? 'exterior' as const : 'boundary' as const;

export function electrostaticSphere(s: UniformChargedSphere, observation: StaticVector) {
  bound(s.radius, .25, 4, 'Source radius m'); bound(s.totalCharge, -4, 4, 'Total source charge C');
  const k = s.coulombConstant === undefined ? teachingCoulombConstant : s.coulombConstant;
  bound(k, 1e-12, 1e12, 'Coulomb coefficient'); vec(observation, 16, 'Observation m');
  const background: StaticVector = s.backgroundField === undefined ? [0, 0, 0] : [...s.backgroundField];
  vec(background, 1e6, 'Uniform electric background N/C');
  const a = s.radius, charge = s.totalCharge, r = Math.hypot(...observation), location = region(r, a);
  const inside = r <= a, zeroField = charge === 0 || r === 0, ratio = Math.min(r / a, 1);
  const coefficient = inside ? k / a / a / a : k / r / r / r;
  // Evaluate the regular Cartesian interior field directly: no r/r cancellation
  // at the origin, and no division by an underflowed enclosed charge or area.
  const rawField = observation.map(x => charge * (coefficient * x)) as StaticVector;
  const field = sourceVector(rawField, zeroField);
  const radial = scalar(charge * (inside ? k / a / a * (r / a) : k / r / r), zeroField);
  const potential = scalar(charge * (inside ? k / a * (1.5 - .5 * (r / a) ** 2) : k / r), charge === 0);
  const totalField = add(field, background);
  const backgroundPotential = -background.reduce((sum, value, i) => sum + value * observation[i], 0);
  const totalPotential = potential.value === null ? null : potential.value + backgroundPotential;
  const enclosed = scalar(((charge * ratio) * ratio) * ratio, charge === 0 || r === 0);
  const area = scalar((4 * Math.PI * r) * r, r === 0);
  const flux = scalar(radial.value === null ? NaN : ((4 * Math.PI * r) * radial.value) * r, zeroField);
  const expectedFlux = scalar((((4 * Math.PI * k) * charge * ratio) * ratio) * ratio, zeroField);
  const meanRadialFromFlux = r === 0 || area.value === null || flux.value === null ? null : flux.value / area.value;
  const density = scalar(charge * (3 / (4 * Math.PI) / a / a / a), charge === 0);
  const slopeInside = scalar(charge * (k / a / a / a), charge === 0);
  const slopeOutsideAtInterface = scalar(charge * (-2 * k / a / a / a), charge === 0);
  const divergenceInside = scalar(charge * (3 * k / a / a / a), charge === 0);
  const atInterface = location === 'boundary' && charge !== 0;
  const derivative = atInterface ? null : inside ? slopeInside.value : scalar(charge * (-2 * k / r / r / r), charge === 0).value;
  const divergence = atInterface ? null : inside ? divergenceInside.value : 0;
  return {
    radius: a, totalCharge: charge, coulombConstant: k, observation: [...observation] as StaticVector,
    radialDistance: r, region: location, radialUnit: r === 0 ? null : observation.map(x => x / r) as StaticVector,
    sourceField: field, sourceFieldStatus: field === null ? 'numerical-range' as const : 'finite' as const,
    sourceFieldMagnitude: field === null ? null : Math.hypot(...field), sourceRadialField: radial.value,
    sourceRadialFieldStatus: radial.status, sourceFieldDirection: direction(field),
    backgroundField: background, totalField, totalFieldDirection: direction(totalField),
    sourcePotential: potential.value, sourcePotentialStatus: potential.status, backgroundPotential, totalPotential,
    potentialReference: 'isolated-source-zero-at-infinity-plus-background-zero-at-origin' as const,
    sourceVolumeDensity: density.value, sourceVolumeDensityStatus: density.status,
    densityAtObservation: atInterface ? null : inside ? density.value : 0,
    densityAtObservationStatus: atInterface ? 'interface-value-not-assigned' as const : inside ? density.status : 'finite' as const,
    enclosedCharge: enclosed.value, enclosedChargeStatus: enclosed.status,
    gaussianSurfaceArea: area.value, gaussianSurfaceAreaStatus: area.status,
    closedElectricFlux: flux.value, closedElectricFluxStatus: flux.status,
    expectedGaussFlux: expectedFlux.value, expectedGaussFluxStatus: expectedFlux.status,
    gaussResidual: flux.value === null || expectedFlux.value === null ? null : flux.value - expectedFlux.value,
    meanRadialFromFlux, normalizationStatus: r === 0 ? 'zero-surface-measure' as const : meanRadialFromFlux === null ? 'numerical-range' as const : 'finite' as const,
    sourceRadialDerivative: derivative, sourceDivergence: divergence,
    derivativeStatus: atInterface ? 'boundary-one-sided-only' as const : derivative === null || divergence === null ? 'numerical-range' as const : 'finite' as const,
    interfaceLimits: {field: scalar(charge * (k / a / a), charge === 0).value,
      radialDerivativeInside: slopeInside.value, radialDerivativeOutside: slopeOutsideAtInterface.value,
      divergenceInside: divergenceInside.value, divergenceOutside: 0, densityInside: density.value, densityOutside: 0},
    model: 'prescribed-uniform-nonconducting-sphere-with-uniform-background' as const,
    residualMeaning: 'analytic-identity-not-independent-verification' as const,
  };
}

export function magnetostaticCylinder(s: UniformSteadyCylinder, observation: StaticVector) {
  bound(s.radius, .25, 4, 'Source radius m'); bound(s.totalCurrent, -4, 4, 'Total source current A');
  const mu = s.permeability === undefined ? teachingVacuumPermeability : s.permeability;
  bound(mu, 1e-12, 2, 'Permeability T m/A'); vec(observation, 16, 'Observation m');
  const background: StaticVector = s.backgroundField === undefined ? [0, 0, 0] : [...s.backgroundField];
  vec(background, 1, 'Uniform magnetic background T');
  const a = s.radius, current = s.totalCurrent, [x, y] = observation, r = Math.hypot(x, y), location = region(r, a);
  const inside = r <= a, zeroField = current === 0 || r === 0, ratio = Math.min(r / a, 1);
  const coefficient = inside ? mu / (2 * Math.PI) / a / a : mu / (2 * Math.PI) / r / r;
  const field = sourceVector([-current * (coefficient * y), current * (coefficient * x), 0], zeroField);
  const azimuthal = scalar(current * (inside ? mu / (2 * Math.PI) / a * (r / a) : mu / (2 * Math.PI) / r), zeroField);
  const totalField = add(field, background);
  const enclosed = scalar((current * ratio) * ratio, zeroField);
  const length = scalar(2 * Math.PI * r, r === 0);
  const circulation = scalar(azimuthal.value === null ? NaN : (2 * Math.PI * r) * azimuthal.value, zeroField);
  const expected = scalar(((mu * current) * ratio) * ratio, zeroField);
  const meanAzimuthalFromCirculation = r === 0 || length.value === null || circulation.value === null ? null : circulation.value / length.value;
  const density = scalar(current * (1 / Math.PI / a / a), current === 0);
  const slopeInside = scalar(current * (mu / (2 * Math.PI) / a / a), current === 0);
  const curlInside = scalar(current * (mu / Math.PI / a / a), current === 0);
  const atInterface = location === 'boundary' && current !== 0;
  const derivative = atInterface ? null : inside ? slopeInside.value : scalar(current * (-mu / (2 * Math.PI) / r / r), current === 0).value;
  const curlZ = atInterface ? null : inside ? curlInside.value : 0;
  return {
    radius: a, totalCurrent: current, permeability: mu, observation: [...observation] as StaticVector,
    radialDistance: r, region: location, azimuthalUnit: r === 0 ? null : [-y / r, x / r, 0] as StaticVector,
    sourceField: field, sourceFieldStatus: field === null ? 'numerical-range' as const : 'finite' as const,
    sourceFieldMagnitude: field === null ? null : Math.hypot(...field), sourceAzimuthalField: azimuthal.value,
    sourceAzimuthalFieldStatus: azimuthal.status, sourceFieldDirection: direction(field),
    backgroundField: background, totalField, totalFieldDirection: direction(totalField),
    sourceCurrentDensity: density.value, sourceCurrentDensityStatus: density.status,
    currentDensityAtObservation: atInterface ? null : inside ? density.value : 0,
    currentDensityAtObservationStatus: atInterface ? 'interface-value-not-assigned' as const : inside ? density.status : 'finite' as const,
    enclosedCurrent: enclosed.value, enclosedCurrentStatus: enclosed.status,
    amperianLoopLength: length.value, amperianLoopLengthStatus: length.status,
    circulation: circulation.value, circulationStatus: circulation.status,
    expectedAmpereCirculation: expected.value, expectedAmpereCirculationStatus: expected.status,
    ampereResidual: circulation.value === null || expected.value === null ? null : circulation.value - expected.value,
    meanAzimuthalFromCirculation, normalizationStatus: r === 0 ? 'zero-loop-length' as const : meanAzimuthalFromCirculation === null ? 'numerical-range' as const : 'finite' as const,
    sourceRadialDerivative: derivative, sourceCurlZ: curlZ,
    sourceDivergence: atInterface ? null : 0,
    derivativeStatus: atInterface ? 'boundary-one-sided-only' as const : derivative === null || curlZ === null ? 'numerical-range' as const : 'finite' as const,
    interfaceLimits: {field: scalar(current * (mu / (2 * Math.PI) / a), current === 0).value,
      radialDerivativeInside: slopeInside.value, radialDerivativeOutside: slopeInside.value === null ? null : -slopeInside.value,
      curlInside: curlInside.value, curlOutside: 0, currentDensityInside: density.value, currentDensityOutside: 0},
    model: 'prescribed-infinite-steady-current-cylinder-with-uniform-background' as const,
    residualMeaning: 'analytic-identity-not-independent-verification' as const,
  };
}
