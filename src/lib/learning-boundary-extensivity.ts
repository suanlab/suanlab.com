/** Finite two-block microstates on open lattices and complete graphs.
 * These energy calculations are not an equilibrium free-energy theorem.
 */
import { exactMarkovObservation as observe } from './learning-finite-markov-exact';
import type { PartitionObservation } from './learning-partition-exact-core';
export interface BoundaryExtensivityParameters {
 dimension: 1 | 2 | 3;
 lengthOne: number;
 lengthTwo: number;
 width: number;
 growthMode: 'uniform-dilation' | 'fixed-thickness';
 scale: number;
 spinOne: -1 | 1;
 spinTwo: -1 | 1;
 couplingNumerator: number;
 couplingDenominator: number;
}
export type BoundaryEnergyMode = 'nearest-neighbor' | 'complete-unscaled' | 'complete-kac';
export type BoundaryLimit = { kind: 'finite'; value: PartitionObservation } | { kind: 'positive-infinity' | 'negative-infinity' };
const Z = BigInt(0), O = BigInt(1), T = BigInt(2);
const integer = (v: unknown, min: number): v is number => typeof v === 'number' && Number.isInteger(v) && v >= min && v <= 1000000;
export function isBoundaryExtensivityParameters(value: unknown): value is BoundaryExtensivityParameters {
 if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
 const p = value as Record<string, unknown>, keys = ['dimension', 'lengthOne', 'lengthTwo', 'width', 'growthMode', 'scale', 'spinOne', 'spinTwo', 'couplingNumerator', 'couplingDenominator'];
 return Object.keys(p).length === keys.length && keys.every(k => Object.prototype.hasOwnProperty.call(p, k)) && [1, 2, 3].includes(p.dimension as number)
  && ['lengthOne', 'lengthTwo', 'width', 'scale', 'couplingDenominator'].every(k => integer(p[k], 1)) && integer(p.couplingNumerator, 0)
  && [1, -1].includes(p.spinOne as number) && [1, -1].includes(p.spinTwo as number)
  && (p.growthMode === 'uniform-dilation' || p.growthMode === 'fixed-thickness')
  && (p.dimension !== 1 || p.width === 1 && p.growthMode === 'uniform-dilation');
}
export function finiteBoundaryExtensivityModel(input: BoundaryExtensivityParameters) {
 if (!isBoundaryExtensivityParameters(input)) throw new RangeError('Expected dimension 1..3, positive integer lengths/scale/denominator up to 1000000, nonnegative coupling numerator, spins ±1; one-dimensional geometry has width one and uniform dilation');
 const parameters = Object.freeze({ dimension: input.dimension, lengthOne: input.lengthOne, lengthTwo: input.lengthTwo, width: input.width,
  growthMode: input.growthMode, scale: input.scale, spinOne: input.spinOne, spinTwo: input.spinTwo,
  couplingNumerator: input.couplingNumerator + 0, couplingDenominator: input.couplingDenominator });
 const d = BigInt(input.dimension), a0 = BigInt(input.lengthOne), b0 = BigInt(input.lengthTwo), w0 = BigInt(input.width), L = a0 + b0;
 const J = BigInt(input.couplingNumerator), D = BigInt(input.couplingDenominator), sA = BigInt(input.spinOne), sB = BigInt(input.spinTwo), sign = sA * sB;
 const uniform = input.growthMode === 'uniform-dilation';
 function sample(scale: number) {
  const k = BigInt(scale), a = a0 * (uniform ? k : O), b = b0 * (uniform ? k : O), w = input.dimension === 1 ? O : w0 * k;
  const section = w ** (d - O), A = a * section, B = b * section, N = A + B;
  const transverse = input.dimension === 1 ? Z : (d - O) * (w - O) * w ** (d - T);
  const nearA = (a - O) * section + a * transverse, nearB = (b - O) * section + b * transverse, nearCross = section;
  const completeA = A * (A - O) / T, completeB = B * (B - O) / T, completeCross = A * B;
  const externalFaces = T * section + (input.dimension === 1 ? Z : T * (d - O) * (a + b) * w ** (d - T));
  function mode(internalA: bigint, internalB: bigint, cross: bigint, normalization: bigint) {
   const denominator = D * normalization, energyA = -J * internalA, energyB = -J * internalB, cut = -J * sign * cross, combined = energyA + energyB + cut;
   return { internalPairCountOne: internalA.toString(), internalPairCountTwo: internalB.toString(), crossPairCount: cross.toString(), normalization: normalization.toString(),
    restrictedEnergyOne: observe(energyA, denominator), restrictedEnergyTwo: observe(energyB, denominator), crossEnergy: observe(cut, denominator), combinedEnergy: observe(combined, denominator),
    cutDifference: observe(combined - energyA - energyB, denominator), reconstructionResidual: observe(combined - energyA - energyB - cut, denominator),
    energyPerSite: observe(combined, denominator * N), crossEnergyPerSite: observe(cut, denominator * N), crossMagnitudePerSite: observe(J * cross, denominator * N),
    maximumAbsoluteEnergy: observe(J * (internalA + internalB + cross), denominator), maximumAbsoluteEnergyPerSite: observe(J * (internalA + internalB + cross), denominator * N) };
  }
  const modes: Record<BoundaryEnergyMode, ReturnType<typeof mode>> = {
   'nearest-neighbor': mode(nearA, nearB, nearCross, O),
   'complete-unscaled': mode(completeA, completeB, completeCross, O),
   'complete-kac': mode(completeA, completeB, completeCross, N),
  };
  // Separately normalizing each cut block changes its Hamiltonian. Include the
  // exact counterterm, rather than interpreting this comparison as a cut alone.
  const separateOne = -J * (A - O), separateTwo = -J * (B - O), separateSum = separateOne + separateTwo;
  const fullKacTwice = -T * J * (completeA + completeB + sign * completeCross);
  const restrictionTwice = -T * J * (completeA + completeB);
  const counterTwice = restrictionTwice - separateSum * N, cutTwice = -T * J * sign * completeCross;
  return { scale, geometry: { dimension: input.dimension, lengthOne: a.toString(), lengthTwo: b.toString(), width: w.toString(), siteCountOne: A.toString(), siteCountTwo: B.toString(), siteCount: N.toString(), externalFaceCount: externalFaces.toString(), interfaceFaceCount: section.toString(),
    externalFacesPerSite: observe(externalFaces, N), interfaceFacesPerSite: observe(section, N), blockFractionOne: observe(A, N), magnetizationDensity: observe(sA * A + sB * B, N) }, modes,
   separatelyNormalizedKac: { energyOne: observe(separateOne, T * D), energyTwo: observe(separateTwo, T * D), energySum: observe(separateSum, T * D),
    combinedMinusSeparate: observe(fullKacTwice - separateSum * N, T * D * N), combinedMinusSeparatePerSite: observe(fullKacTwice - separateSum * N, T * D * N * N),
    normalizationCounterterm: observe(counterTwice, T * D * N), reconstructionResidual: observe(fullKacTwice - separateSum * N - counterTwice - cutTwice, T * D * N) } };
 }
 const scales: number[] = []; for (let k = 1; k <= input.scale; k *= 2) scales.push(k); if (scales[scales.length - 1] !== input.scale) scales.push(input.scale);
 const history = scales.map(sample), current = history[history.length - 1], finite = (n: bigint, den = O): BoundaryLimit => ({ kind: 'finite', value: observe(n, den) });
 const signedDivergence = (positive: boolean): BoundaryLimit => ({ kind: positive ? 'positive-infinity' : 'negative-infinity' });
 const rawCross = J === Z ? finite(Z) : signedDivergence(sign < Z), rawMaximum = J === Z ? finite(Z) : signedDivergence(true);
 const magnetizationNumerator = sA * a0 + sB * b0;
 const limits = {
  externalFacesPerSite: uniform ? finite(Z) : finite(T, L),
  'nearest-neighbor': { energyPerSite: uniform ? finite(-J * d, D) : finite(-J * (d * L - T + sign), D * L),
   crossEnergyPerSite: uniform ? finite(Z) : finite(-J * sign, D * L), crossMagnitudePerSite: uniform ? finite(Z) : finite(J, D * L),
   maximumAbsoluteEnergyPerSite: uniform ? finite(J * d, D) : finite(J * (d * L - O), D * L) },
  'complete-unscaled': { energyPerSite: J === Z ? finite(Z) : magnetizationNumerator === Z ? finite(J, T * D) : signedDivergence(false), crossEnergyPerSite: rawCross, crossMagnitudePerSite: rawMaximum, maximumAbsoluteEnergyPerSite: rawMaximum },
  'complete-kac': { energyPerSite: finite(-J * magnetizationNumerator * magnetizationNumerator, T * D * L * L), crossEnergyPerSite: finite(-J * sign * a0 * b0, D * L * L), crossMagnitudePerSite: finite(J * a0 * b0, D * L * L), maximumAbsoluteEnergyPerSite: finite(J, T * D) },
  separatelyNormalizedKacDifferencePerSite: finite(J * a0 * b0 * (O - sign), D * L * L),
 };
 return { parameters, arithmetic: 'exact-bigint-rational' as const, fullyResolved: true, history, current, limits,
  classification: { vanHoveGeometryAlongPath: uniform, zeroCoupling: J === Z, balancedMagnetization: magnetizationNumerator === Z,
   localCutNegligiblePerSiteAlongPath: J === Z || uniform, kacCutNegligiblePerSiteAlongPath: J === Z,
   maximumEnergyGrowth: { 'nearest-neighbor': J === Z ? 'zero' as const : 'linear-in-site-count' as const, 'complete-unscaled': J === Z ? 'zero' as const : 'quadratic-in-site-count' as const, 'complete-kac': J === Z ? 'zero' as const : 'linear-in-site-count' as const } },
  approximationContract: 'Two adjacent open rectangular lattice blocks with uniformly fixed classical spins ±1 inside each block, dimension 1..3. Positive integer base lengths/width and scale are at most 1000000; the first dimension is the joining direction. One-dimensional width is one and only uniform dilation is allowed. Uniform dilation grows both longitudinal lengths and transverse widths; fixed-thickness growth in dimension 2 or 3 grows only transverse widths. Site volume is fixed; geometry counts are exact decimal integer strings, including counts above Number safe integers. The coupling is a nonnegative exact rational in a chosen common energy unit, with numerator 0..1000000 and denominator 1..1000000. Nearest-neighbor edges and unordered complete-graph pairs exclude all self pairs. The complete Kac kernel uses J/N globally, including on restricted blocks; separately normalizing each block changes its Hamiltonian and is recorded with an explicit counterterm. All finite-size observations and finite analytic path limits are exact rational points; divergent analytic limits have signed symbolic tags, never Number infinities. History uses powers of two through the requested scale and includes the requested endpoint; its last entry is current. Energies concern named deterministic microstates, not Gibbs expectations. The energy norm bounds and cut comparisons do not by themselves prove existence, ensemble equivalence or analyticity of an equilibrium free-energy density, a phase diagram or a general thermodynamic-limit theorem. Growth scale is physical spatial dilation, not stochastic time, a sample count or an animation-frame duration.' };
}
