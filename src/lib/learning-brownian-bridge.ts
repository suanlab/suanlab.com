/** Exact finite Gaussian bridge. This numerical module must not enter the reader bundle. */
import { readBrownianBridgeInput } from './learning-brownian-bridge-input';
import { f, zero, one, two, add, sub, mul, div, sum, eq, type F } from './learning-quantum-hilbert-core';
const integer = (n: number) => f(BigInt(n));
const read = (value: string) => { const [n, d = '1'] = value.split('/'); return f(BigInt(n), BigInt(d)); };
const text = (x: F) => x.d === BigInt(1) ? String(x.n) : `${x.n}/${x.d}`;
const square = (x: F) => mul(x, x);
const wire = (matrix: F[][]) => matrix.map(row => row.map(text));

export function calculateBrownianBridge(value: unknown) {
  const input = readBrownianBridgeInput(value);
  if (!input) throw RangeError('Invalid finite Brownian bridge input');
  const n = input.intervals, dimension = n - 1, N = integer(n), time = read(input.time), diffusion = read(input.diffusion), start = read(input.start), end = read(input.end), source = read(input.source);
  const delta = div(time, N), scale = mul(mul(two, diffusion), delta), displacement = sub(end, start);
  const times = Array.from({ length: n + 1 }, (_, i) => mul(delta, integer(i)));
  const baselineMean = times.map(t => add(start, mul(displacement, div(t, time))));
  const covariance = times.map((_, i) => times.map((__, j) => mul(scale, sub(integer(Math.min(i, j)), div(integer(i * j), N)))));
  const precision = Array.from({ length: dimension }, (_, i) => Array.from({ length: dimension }, (__, j) => div(integer(i === j ? 2 : Math.abs(i - j) === 1 ? -1 : 0), scale)));
  const mean = baselineMean.map((m, i) => add(m, mul(source, covariance[i][input.probe])));
  const ldltDiagonal = Array.from({ length: dimension }, (_, i) => div(div(integer(i + 2), integer(i + 1)), scale));
  const ldltSubdiagonal = Array.from({ length: Math.max(0, dimension - 1) }, (_, i) => div(integer(-(i + 1)), integer(i + 2)));
  const determinant = ldltDiagonal.reduce(mul, one);
  const endpointForcing = Array.from({ length: dimension }, () => zero);
  endpointForcing[0] = add(endpointForcing[0], div(start, scale));
  endpointForcing[dimension - 1] = add(endpointForcing[dimension - 1], div(end, scale));
  const meanResidual = precision.map((row, i) => sub(sum(row.map((x, j) => mul(x, mean[j + 1]))), add(endpointForcing[i], i + 1 === input.probe ? source : zero)));
  const inverseResidual = precision.map((row, i) => row.map((_, j) => sub(sum(row.map((x, k) => mul(x, covariance[k + 1][j + 1]))), i === j ? one : zero)));
  if (meanResidual.some(x => !eq(x, zero)) || inverseResidual.some(row => row.some(x => !eq(x, zero)))) throw Error('Finite bridge identities failed');
  const baselineAction = div(square(displacement), mul(mul(integer(4), diffusion), time));
  const sourceVariance = covariance[input.probe][input.probe];
  const logGenerating = add(mul(source, baselineMean[input.probe]), div(mul(square(source), sourceVariance), two));
  const areaMean = mul(delta, add(div(add(start, end), two), sum(mean.slice(1, n))));
  const areaVariance = mul(square(delta), sum(covariance.slice(1, n).map(row => sum(row.slice(1, n)))));
  const limitingAreaVariance = div(mul(diffusion, mul(square(time), time)), integer(6));
  const incrementMean = mean.slice(1).map((m, i) => sub(m, mean[i]));
  const incrementCovariance = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (__, j) => mul(scale, sub(i === j ? one : zero, div(one, N)))));
  const deterministicVariation = sum(incrementMean.map(square));
  const expectedQuadraticVariation = add(deterministicVariation, sum(incrementCovariance.map((row, i) => row[i])));
  const expectedAction = div(expectedQuadraticVariation, mul(two, scale));
  const refinementRows = [1, 2, 4, 8].map(factor => {
    const intervals = n * factor, step = div(time, integer(intervals));
    const variance = mul(limitingAreaVariance, sub(one, div(one, integer(intervals * intervals))));
    return {
      intervals, integrationDimension: intervals - 1, matchedProbeIndex: input.probe * factor,
      timeStep: text(step), probeTime: text(times[input.probe]), meanAtProbe: text(mean[input.probe]), varianceAtProbe: text(sourceVariance),
      areaMean: text(areaMean), areaVariance: text(variance), areaVarianceGap: text(sub(limitingAreaVariance, variance)),
      expectedQuadraticVariation: text(add(div(deterministicVariation, integer(factor)), mul(mul(two, mul(diffusion, time)), sub(one, div(one, integer(intervals)))))),
      expectedAction: text(add(add(baselineAction, div(integer(intervals - 1), two)), div(mul(square(source), sourceVariance), two))),
      fullMatrixCalculated: factor === 1,
    };
  });
  return {
    input, integrationDimension: dimension, timeStep: text(delta), times: times.map(text), baselineMean: baselineMean.map(text), mean: mean.map(text),
    precision: wire(precision), covariance: wire(covariance),
    factorization: { diagonal: ldltDiagonal.map(text), subdiagonal: ldltSubdiagonal.map(text), determinant: text(determinant) },
    kernel: { prefactorSquaredTimesPi: text(div(one, mul(integer(4), mul(diffusion, time)))), baselineNegativeLogExponential: text(baselineAction), tiltedLogExponential: text(sub(logGenerating, baselineAction)), formula: 'K_tau(b,a)=exp[-(b-a)^2/(4 D tau)]/sqrt(4 pi D tau)' },
    source: { convention: 'dimensionless-exponent-point-source', probeTime: text(times[input.probe]), baselineMeanAtProbe: text(baselineMean[input.probe]), varianceAtProbe: text(sourceVariance), logGenerating: text(logGenerating), meanResponse: covariance.map(row => text(row[input.probe])) },
    area: { baselineMean: text(div(mul(time, add(start, end)), two)), mean: text(areaMean), variance: text(areaVariance), analyticLimitingVariance: text(limitingAreaVariance), varianceGap: text(sub(limitingAreaVariance, areaVariance)) },
    increments: { mean: incrementMean.map(text), covariance: wire(incrementCovariance), deterministicQuadraticVariation: text(deterministicVariation), expectedQuadraticVariation: text(expectedQuadraticVariation), expectedAction: text(expectedAction), baselineExpectedAction: text(add(baselineAction, div(integer(dimension), two))), fluctuationAction: text(div(integer(dimension), two)) },
    refinements: refinementRows, certificates: { meanResidual: meanResidual.map(text), inverseResidual: wire(inverseResidual) },
    scope: { fullMatrixIntervals: n, largestAnalyticRefinementIntervals: n * 8, continuousMeasureExistenceProved: false, randomPathsGenerated: false, transcendentalDecimalEnclosuresComputed: false },
  };
}
export type BrownianBridgeResult = ReturnType<typeof calculateBrownianBridge>;
