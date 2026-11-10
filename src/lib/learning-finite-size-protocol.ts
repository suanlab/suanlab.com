/** Main-thread transport validation only. Type-only numerical imports are erased.
 * Structure and preparation identity are checked; a sender is not certified as
 * mathematically truthful by these checks. Independent numerical tests do that.
 */
import type { PartitionRational } from './learning-partition-exact-core';
import type { finiteBoundaryExtensivityModel, BoundaryExtensivityParameters } from './learning-boundary-extensivity';
import type { spatialDensitySelfAveragingModel, SpatialSelfAveragingParameters } from './learning-spatial-self-averaging';
import type { twoPhaseRoundingModel, TwoPhaseRoundingParameters } from './learning-two-phase-rounding';
export const finiteSizeKinds = ['boundary-growth-extensivity-additivity', 'spatial-density-self-averaging-correlations', 'two-phase-partition-zeros-rounding'] as const;
export type FiniteSizeKind = typeof finiteSizeKinds[number];
export type FiniteSizeInput = { kind: typeof finiteSizeKinds[0]; parameters: BoundaryExtensivityParameters } | { kind: typeof finiteSizeKinds[1]; parameters: SpatialSelfAveragingParameters } | { kind: typeof finiteSizeKinds[2]; parameters: TwoPhaseRoundingParameters };
type Results = { 'boundary-growth-extensivity-additivity': ReturnType<typeof finiteBoundaryExtensivityModel>; 'spatial-density-self-averaging-correlations': ReturnType<typeof spatialDensitySelfAveragingModel>; 'two-phase-partition-zeros-rounding': ReturnType<typeof twoPhaseRoundingModel> };
export type FiniteSizeSuccess = { [K in FiniteSizeKind]: { version: 1; requestId: number; ok: true; kind: K; input: Extract<FiniteSizeInput, { kind: K }>; inputKey: string; result: Results[K] } }[FiniteSizeKind];
export type FiniteSizeResponse = FiniteSizeSuccess | { version: 1; requestId: number | null; ok: false; error: 'invalid-request' | 'invalid-input' | 'calculation-failed' };
export const finiteSizeRecord = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v);
type Check = (v: unknown) => boolean;
const shape = (v: unknown, spec: Record<string, Check>): v is Record<string, unknown> => finiteSizeRecord(v) && Object.keys(v).length === Object.keys(spec).length && Object.entries(spec).every(([k, check]) => Object.prototype.hasOwnProperty.call(v, k) && check(v[k]));
const integer = (v: unknown, low = 0, high = 1000000): v is number => typeof v === 'number' && Number.isInteger(v) && v >= low && v <= high;
const positive: Check = v => integer(v, 1);
const literal = (x: unknown): Check => v => v === x;
const choice = (...xs: unknown[]): Check => v => xs.includes(v);
const bool: Check = v => typeof v === 'boolean';
const finite: Check = v => typeof v === 'number' && Number.isFinite(v);
const nullable = (check: Check): Check => v => v === null || check(v);
const array = (size: number, check: Check): Check => v => Array.isArray(v) && v.length === size && Array.from(v).every(check);
const fields = (names: string, check: Check) => Object.fromEntries(names.split(' ').map(name => [name, check]));
const specs: Record<string, Check>[] = [
 { dimension: choice(1, 2, 3), ...fields('lengthOne lengthTwo width scale couplingDenominator', positive), growthMode: choice('uniform-dilation', 'fixed-thickness'), spinOne: choice(-1, 1), spinTwo: choice(-1, 1), couplingNumerator: integer },
 { siteCount: (v: unknown) => integer(v, 1, 512), occupiedWeight: integer, emptyWeight: integer, commonNumerator: integer, commonDenominator: positive, commonScaling: choice('fixed', 'inverse-size', 'inverse-size-squared'), epsilonNumerator: positive, epsilonDenominator: positive },
 { siteCount: (v: unknown) => integer(v, 1, 512), activityNumerator: positive, activityDenominator: positive, sizePath: choice('fixed', 'inverse-size', 'inverse-size-squared'), approachSide: choice(-1, 1), offsetNumerator: integer, offsetDenominator: positive },
];
export function isFiniteSizeInput(v: unknown): v is FiniteSizeInput {
 if (!shape(v, { kind: choice(...finiteSizeKinds), parameters: finiteSizeRecord })) return false;
 const p = v.parameters as Record<string, unknown>, index = finiteSizeKinds.indexOf(v.kind as FiniteSizeKind);
 if (!shape(p, specs[index])) return false;
 if (index === 0) return p.dimension !== 1 || p.width === 1 && p.growthMode === 'uniform-dilation';
 if (index === 1) return Number(p.occupiedWeight) + Number(p.emptyWeight) > 0 && Number(p.commonNumerator) <= Number(p.commonDenominator);
 return Number(p.offsetNumerator) <= Number(p.offsetDenominator);
}
export function snapshotFiniteSizeInput(input: FiniteSizeInput): FiniteSizeInput {
 if (!isFiniteSizeInput(input)) throw new RangeError('Invalid finite-size preparation');
 const index = finiteSizeKinds.indexOf(input.kind), source = input.parameters as unknown as Record<string, number | string>;
 const keys = index === 0 ? ['dimension', 'lengthOne', 'lengthTwo', 'width', 'growthMode', 'scale', 'spinOne', 'spinTwo', 'couplingNumerator', 'couplingDenominator'] : Object.keys(specs[index]);
 const parameters = Object.freeze(Object.fromEntries(keys.map(k => [k, typeof source[k] === 'number' ? (source[k] as number) + 0 : source[k]])));
 return Object.freeze({ kind: input.kind, parameters }) as unknown as FiniteSizeInput;
}
export const finiteSizeInputKey = (input: FiniteSizeInput) => JSON.stringify(snapshotFiniteSizeInput(input));
export const isFiniteSizeRequestEnvelope = (v: unknown): v is { version: 1; requestId: number; input: unknown } => shape(v, { version: literal(1), requestId: v => Number.isSafeInteger(v) && (v as number) > 0, input: () => true });
const same = (a: unknown, b: unknown) => { if (a === b) return true; try { return JSON.stringify(a) === JSON.stringify(b); } catch { return false; } };
/** The bound covers the largest squared probability denominators and interval
 * endpoints in N<=512 calculations without parsing BigInts on the UI thread.
 */
export function isFiniteSizeObservation(v: unknown, allowInterval = true): boolean {
 if (!shape(v, { value: nullable(finite), sign: choice(-1, 0, 1), logMagnitude: nullable(finite), logResolution: choice('finite', 'not-applicable'), resolution: choice('exact-rational', 'structural-zero', ...(allowInterval ? ['interval-resolved'] : [])), bounds: finiteSizeRecord })) return false;
 const coefficient = (x: unknown, positiveOnly = false) => typeof x === 'string' && x.length <= 16384 && (positiveOnly ? /^[1-9]\d*$/ : /^-?(?:0|[1-9]\d*)$/).test(x) && x !== '-0';
 const term = (x: unknown, pos = false) => shape(x, { coefficient: v => coefficient(v, pos), exponent: literal(0) });
 const rational: Check = x => shape(x, { numerator: v => term(v), denominator: v => term(v, true) });
 if (!shape(v.bounds, { lower: rational, upper: rational })) return false;
 const bounds = v.bounds as { lower: PartitionRational; upper: PartitionRational }, a = bounds.lower.numerator.coefficient as string, b = bounds.upper.numerator.coefficient as string;
 if (v.resolution === 'structural-zero') return v.value === 0 && v.sign === 0 && v.logMagnitude === null && v.logResolution === 'not-applicable' && same(bounds.lower, bounds.upper) && a === '0' && bounds.lower.denominator.coefficient === '1';
 if (a === '0' || b === '0' || v.sign !== (a[0] === '-' ? -1 : 1) || v.sign !== (b[0] === '-' ? -1 : 1) || !finite(v.logMagnitude) || v.logResolution !== 'finite') return false;
 if (v.value !== null && (!finite(v.value) || v.value === 0 || Math.sign(v.value as number) !== v.sign)) return false;
 return v.resolution === 'exact-rational' ? same(bounds.lower, bounds.upper) : !same(bounds.lower, bounds.upper);
}
export function isFiniteSizeSuccessForInput(v: unknown, input: FiniteSizeInput, requestId: number): v is FiniteSizeSuccess {
 if (!Number.isSafeInteger(requestId) || requestId <= 0 || !isFiniteSizeInput(input) || !shape(v, { version: literal(1), requestId: literal(requestId), ok: literal(true), kind: literal(input.kind), input: isFiniteSizeInput, inputKey: literal(finiteSizeInputKey(input)), result: finiteSizeRecord }) || finiteSizeInputKey(v.input as FiniteSizeInput) !== finiteSizeInputKey(input)) return false;
 const result = v.result as Record<string, unknown>, snapshot = snapshotFiniteSizeInput(input), cache = new WeakMap<object, boolean>();
 const q: Check = x => { if (!finiteSizeRecord(x)) return false; if (cache.has(x)) return cache.get(x)!; const valid = isFiniteSizeObservation(x, input.kind === finiteSizeKinds[2]); cache.set(x, valid); return valid; };
 const exact: Check = x => isFiniteSizeObservation(x, false);
 const contract: Check = x => typeof x === 'string' && x.length > 0 && x.length < 10000;
 const base = { parameters: (x: unknown) => same(x, snapshot.parameters), fullyResolved: literal(true), approximationContract: contract };
 const limit = (...kinds: string[]): Check => x => shape(x, { kind: literal('finite'), value: q }) || shape(x, { kind: choice(...kinds) });
 const sizes: number[] = [], endpoint = input.kind === finiteSizeKinds[0] ? input.parameters.scale : input.parameters.siteCount;
 for (let n = 1; n <= endpoint; n *= 2) sizes.push(n); if (sizes.at(-1) !== endpoint) sizes.push(endpoint);
 const orderedHistory = (row: Check, key: string): Check => x => array(sizes.length, row)(x) && (x as Record<string, unknown>[]).every((r, i) => r[key] === sizes[i]);
 const count: Check = x => typeof x === 'string' && x.length <= 100 && /^(?:0|[1-9]\d*)$/.test(x);
 const modes = ['nearest-neighbor', 'complete-unscaled', 'complete-kac'];
 if (input.kind === finiteSizeKinds[0]) {
  const mode: Check = x => shape(x, { ...fields('internalPairCountOne internalPairCountTwo crossPairCount normalization', count), ...fields('restrictedEnergyOne restrictedEnergyTwo crossEnergy combinedEnergy cutDifference reconstructionResidual energyPerSite crossEnergyPerSite crossMagnitudePerSite maximumAbsoluteEnergy maximumAbsoluteEnergyPerSite', q) });
  const row: Check = x => shape(x, { scale: positive, geometry: y => shape(y, { dimension: literal(input.parameters.dimension), ...fields('lengthOne lengthTwo width siteCountOne siteCountTwo siteCount externalFaceCount interfaceFaceCount', count), ...fields('externalFacesPerSite interfaceFacesPerSite blockFractionOne magnetizationDensity', q) }), modes: y => shape(y, Object.fromEntries(modes.map(k => [k, mode]))), separatelyNormalizedKac: y => shape(y, fields('energyOne energyTwo energySum combinedMinusSeparate combinedMinusSeparatePerSite normalizationCounterterm reconstructionResidual', q)) });
  const energyLimits: Check = x => shape(x, fields('energyPerSite crossEnergyPerSite crossMagnitudePerSite maximumAbsoluteEnergyPerSite', limit('positive-infinity', 'negative-infinity')));
  return shape(result, { ...base, arithmetic: literal('exact-bigint-rational'), current: row, history: orderedHistory(row, 'scale'), limits: x => shape(x, { externalFacesPerSite: limit(), ...Object.fromEntries(modes.map(k => [k, energyLimits])), separatelyNormalizedKacDifferencePerSite: limit() }), classification: x => shape(x, { vanHoveGeometryAlongPath: literal(input.parameters.growthMode === 'uniform-dilation'), zeroCoupling: literal(input.parameters.couplingNumerator === 0), balancedMagnetization: bool, localCutNegligiblePerSiteAlongPath: bool, kacCutNegligiblePerSiteAlongPath: literal(input.parameters.couplingNumerator === 0), maximumEnergyGrowth: y => shape(y, Object.fromEntries(modes.map(k => [k, choice('zero', 'linear-in-site-count', 'quadratic-in-site-count')])) ) }) }) && same(result.current, (result.history as unknown[]).at(-1));
 }
 if (input.kind === finiteSizeKinds[1]) {
  const p = input.parameters, n = p.siteCount, degenerate = p.occupiedWeight === 0 || p.emptyWeight === 0;
  const summary: Check = x => shape(x, { siteCount: v => integer(v, 1, 512), ...fields('commonWeight meanDensity densityVariance scaledDensityVariance independentDensityVariance eventProbability independentEventProbability collectiveEventProbability chebyshevBound mixtureAwareBound independentReferenceBound', q), varianceEquivalentCount: degenerate ? literal(null) : q });
  const row: Check = x => shape(x, { count: v => integer(v, 0, n), ...fields('density deviation independentProbability collectiveProbability probability', q), inDeviationEvent: bool });
  const currentResult = result.current as Results['spatial-density-self-averaging-correlations']['current'];
  const current: Check = x => shape(x, { summary: y => summary(y) && (y as Record<string, unknown>).siteCount === n, distribution: y => array(n + 1, row)(y) && (y as Record<string, unknown>[]).every((r, i) => r.count === i),
   ...fields('siteVariance meanCount countVariance densitySecondMoment diagonalCovarianceSum orderedCrossCovarianceSum covarianceSumDensityVariance designFactor distributionNormalization densityMeanFromDistribution densityVarianceFromDistribution varianceIdentityResidual independentEventContribution collectiveEventContribution eventMixtureIdentityResidual chebyshevSlack mixtureAwareSlack eventProbabilityMinusIndependent', q),
   distinctPairCovariance: n === 1 ? literal(null) : q, distinctPairCorrelation: n === 1 || degenerate ? literal(null) : q,
   varianceRatioToIndependent: degenerate ? literal(null) : q, eventTailRatioToIndependent: nullable(q), diagonalCovarianceMultiplicity: literal(n), orderedCrossCovarianceMultiplicity: literal(n * (n - 1)), distributionMomentsMatch: literal(true), independentReferenceBoundFailsForMixture: bool,
   eventCounts: y => Array.isArray(y) && y.length <= n + 1 && Array.from(y).every(k => integer(k, 0, n)) && new Set(y).size === y.length, eventContainsPositiveMass: bool, eventIsWholeGrid: bool, positiveMassCount: v => integer(v, 1, n + 1), distributionIsProductForThisSize: literal(n === 1 || p.commonNumerator === 0 || degenerate), approximationContract: contract });
  return shape(result, { ...base, arithmetic: literal('exact-bigint-rational'), current, history: orderedHistory(summary, 'siteCount'), limits: x => shape(x, { ...fields('commonWeight densityVariance deviationEventProbability', q), scaledDensityVariance: limit('positive-infinity'), varianceEquivalentCount: degenerate ? (y => shape(y, { kind: literal('not-applicable') })) : limit('positive-infinity'), varianceEquivalentFraction: degenerate ? (y => shape(y, { kind: literal('not-applicable') })) : limit(), distinctPairCorrelation: degenerate ? literal(null) : q, densityLawAtoms: y => Array.isArray(y) && y.length >= 1 && y.length <= 3 && Array.from(y).every(a => shape(a, { density: q, probability: q })) }), classification: x => shape(x, { deterministicOccupancy: literal(degenerate), distinctPairExistsAtCurrentSize: literal(n >= 2), selfAveragesInProbabilityAlongPath: literal(degenerate || p.commonScaling !== 'fixed' || p.commonNumerator === 0), commonModeVanishesAlongPath: literal(p.commonScaling !== 'fixed' || p.commonNumerator === 0), countVarianceGrowth: choice('zero', 'linear-in-site-count', 'quadratic-in-site-count') }) })
   && same(currentResult.summary, (result.history as unknown[]).at(-1)) && same(currentResult.eventCounts, currentResult.distribution.filter(r => r.inDeviationEvent).map(r => r.count));
 }
 const n = input.parameters.siteCount, p = input.parameters, fixedOffCoexistence = p.sizePath === 'fixed' && p.activityNumerator !== p.activityDenominator;
 const row: Check = x => shape(x, { siteCount: v => integer(v, 1, 512), ...fields('activity activityPower partitionFunction occupiedProbability emptyProbability meanDensity densityVariance meanCount countVariance densityResponseToField countResponseToField minorityProbability minorityToMajorityWeight exponentialPressureCorrection densityVarianceFromNamedStates', exact), ...fields('field scaledField limitingPressureAtCurrentField pressure pressureCorrection pressureCorrectionUpperBound tenToNinetyFieldWidth', q), allowedConfigurationCount: literal(2), distribution: y => Array.isArray(y) && y.length === 2 && y.every((r, i) => shape(r, { count: literal(i === 0 ? 0 : (x as Record<string, unknown>).siteCount), density: exact, probability: exact })), approximationContract: contract });
 const fractionMatches = (x: unknown, a: number, b: number) => { if (!exact(x)) return false; const q = (x as { bounds: { lower: PartitionRational } }).bounds.lower; return q.numerator.coefficient.length <= 6 && q.denominator.coefficient.length <= 6 && Number(q.numerator.coefficient) * b === a * Number(q.denominator.coefficient); };
 return shape(result, { ...base, current: x => row(x) && (x as Record<string, unknown>).siteCount === n, history: orderedHistory(row, 'siteCount'), roots: x => array(n, y => finiteSizeRecord(y))(x) && (x as Record<string, unknown>[]).every((r, i) => shape(r, { index: literal(i), turns: y => fractionMatches(y, 2 * i + 1, 2 * n), poweredTurnsModuloOne: y => fractionMatches(y, 1, 2), conjugateIndex: literal(n - 1 - i), isNegativeRealRoot: literal(2 * i + 1 === n), plotReal: finite, plotImaginary: finite })),
  limits: x => shape(x, { ...fields('pressure meanDensity densityVariance', q), densityResponseToField: fixedOffCoexistence ? limit() : (y => shape(y, { kind: literal('positive-infinity') })), scaledField: limit('positive-infinity', 'negative-infinity'), densityLawAtoms: y => Array.isArray(y) && y.length >= 1 && y.length <= 2 && Array.from(y).every(a => shape(a, { density: exact, probability: q })) }),
  rootGeometry: x => shape(x, { polynomial: literal('1+z^N'), degree: literal(n), nearestPositiveAxisTurns: y => fractionMatches(y, 1, 2 * n), nearestLogFieldImaginaryPiFactor: y => fractionMatches(y, 1, n), negativeRealRootExists: literal(n % 2 === 1), positiveRealRootExists: literal(false), closestRootDistanceToOnePlot: finite, coordinatesAreApproximate: literal(true) }), classification: x => shape(x, { selfAveragesAlongPath: literal(fixedOffCoexistence), nonzeroLimitingDensityVariance: literal(!fixedOffCoexistence), responseDivergesAlongPath: literal(!fixedOffCoexistence), finitePressureSmoothForPositiveActivity: literal(true) }) }) && same(result.current, (result.history as unknown[]).at(-1));
}
