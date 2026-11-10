/** Exact finite trapezoidal harmonic kernel. Calculation Worker only. */
import { readEuclideanKernelInput } from './learning-euclidean-kernel-input';
import { f, zero, one, two, add, sub, mul, div, sum, eq, type F } from './learning-quantum-hilbert-core';
const integer = (n: number) => f(BigInt(n));
const read = (value: string) => { const [n, d = '1'] = value.split('/'); return f(BigInt(n), BigInt(d)); };
const text = (value: F) => value.d === BigInt(1) ? String(value.n) : `${value.n}/${value.d}`;
const square = (value: F) => mul(value, value);
const wire = (matrix: F[][]) => matrix.map(row => row.map(text));

function recurrence(n: number, v: F) {
  const values = [one, add(two, v)];
  for (let k = 2; k < n; k++) values.push(sub(mul(add(two, v), values[k - 1]), values[k - 2]));
  return values;
}
function coefficients(n: number, time: F, mass: F, hbar: F, omegaSquared: F) {
  const delta = div(time, integer(n)), scale = div(mass, mul(hbar, delta)), v = mul(omegaSquared, square(delta)), ds = recurrence(n, v), d = ds[n - 1];
  const p = div(scale, mul(two, d)), g = div(mul(scale, sub(add(one, div(v, two)), div(ds[n - 2], d))), two), h = div(scale, d);
  return { delta, scale, v, ds, p, g, h };
}
function packed(matrix: F[][]) {
  const values: string[] = [], index = new Map<string, number>();
  const indices = matrix.map(row => row.map(value => {
    const key = text(value); let id = index.get(key);
    if (id === undefined) { id = values.length; values.push(key); index.set(key, id); }
    return id;
  }));
  return { values: Array.from({ length: Math.ceil(values.length / 128) }, (_, i) => values.slice(i * 128, (i + 1) * 128)), indices };
}

export function calculateEuclideanKernel(value: unknown) {
  const input = readEuclideanKernelInput(value);
  if (!input) throw RangeError('Invalid finite Euclidean kernel input');
  const n = input.intervals, size = n - 1, time = read(input.time), mass = read(input.mass), hbar = read(input.hbar), omegaSquared = read(input.omegaSquared), start = read(input.start), end = read(input.end);
  const { delta, scale, v, ds, p, g, h } = coefficients(n, time, mass, hbar, omegaSquared);
  const precision = Array.from({ length: size }, (_, i) => Array.from({ length: size }, (__, j) => mul(scale, i === j ? add(two, v) : integer(Math.abs(i - j) === 1 ? -1 : 0))));
  const mean = Array.from({ length: n + 1 }, (_, i) => i === 0 ? start : i === n ? end : div(add(mul(start, ds[size - i]), mul(end, ds[i - 1])), ds[size]));
  const covariance = Array.from({ length: n + 1 }, (_, i) => Array.from({ length: n + 1 }, (__, j) => i === 0 || j === 0 || i === n || j === n ? zero : div(mul(ds[Math.min(i, j) - 1], ds[size - Math.max(i, j)]), mul(scale, ds[size]))));
  const forcing = Array.from({ length: size }, () => zero);
  forcing[0] = add(forcing[0], mul(scale, start)); forcing[size - 1] = add(forcing[size - 1], mul(scale, end));
  const meanResidual = precision.map((row, i) => sub(sum(row.map((q, j) => mul(q, mean[j + 1]))), forcing[i]));
  // Tridiagonal multiplication retains the complete inverse certificate without cubic work.
  const inverseResidual = precision.map((_, i) => precision.map((__, j) => sub(sum([i - 1, i, i + 1].filter(k => k >= 0 && k < size).map(k => mul(precision[i][k], covariance[k + 1][j + 1]))), i === j ? one : zero)));
  if (meanResidual.some(x => !eq(x, zero)) || inverseResidual.some(row => row.some(x => !eq(x, zero)))) throw Error('Finite oscillator identities failed');
  const constant = add(div(mul(scale, add(square(start), square(end))), two), div(mul(mul(scale, v), add(square(start), square(end))), integer(4)));
  const minimumAction = sub(mul(g, add(square(start), square(end))), mul(h, mul(start, end)));
  const endpointShift = div(mul(mul(scale, v), sub(square(start), square(end))), integer(4));
  const diagonal = Array.from({ length: size }, (_, i) => mul(scale, div(ds[i + 1], ds[i])));
  const subdiagonal = Array.from({ length: size - 1 }, (_, i) => div(mul(integer(-1), ds[i]), ds[i + 1]));
  const determinant = diagonal.reduce(mul, one);
  const meanIncrement = mean.slice(1).map((x, i) => sub(x, mean[i]));
  const incrementVariance = meanIncrement.map((_, i) => sub(add(covariance[i + 1][i + 1], covariance[i][i]), mul(two, covariance[i][i + 1])));
  const kinetic = div(mul(scale, sum(meanIncrement.map((x, i) => add(square(x), incrementVariance[i])))), two);
  const potential = div(mul(mul(scale, v), add(div(add(square(start), square(end)), two), sum(mean.slice(1, n).map((x, i) => add(square(x), covariance[i + 1][i + 1]))))), two);
  const traceCurvature = sub(mul(two, g), h), isFree = omegaSquared.n === BigInt(0);
  const refinements = [1, 2, 4, 8].map(factor => {
    const intervals = n * factor, probe = input.probe * factor, c = coefficients(intervals, time, mass, hbar, omegaSquared), count = intervals - 1;
    const mu = probe === 0 ? start : probe === intervals ? end : div(add(mul(start, c.ds[count - probe]), mul(end, c.ds[probe - 1])), c.ds[count]);
    const variance = probe === 0 || probe === intervals ? zero : div(mul(c.ds[probe - 1], c.ds[count - probe]), mul(c.scale, c.ds[count]));
    return { intervals, integrationDimension: count, matchedProbeIndex: probe, timeStep: text(c.delta), probeTime: text(mul(c.delta, integer(probe))), meanAtProbe: text(mu), varianceAtProbe: text(variance), prefactorSquaredTimesPi: text(c.p), diagonalExponentCoefficient: text(c.g), crossExponentCoefficient: text(c.h), minimumAction: text(sub(mul(c.g, add(square(start), square(end))), mul(c.h, mul(start, end)))), rightLogCorrection: text(div(endpointShift, integer(factor))), traceSquared: isFree ? null : text(div(c.p, sub(mul(two, c.g), c.h))), fullMatrixCalculated: factor === 1 };
  });
  return {
    input, integrationDimension: size, timeStep: text(delta), inverseTemperature: text(div(time, hbar)), times: Array.from({ length: n + 1 }, (_, i) => text(mul(delta, integer(i)))), mean: mean.map(text), precision: wire(precision), covariance: packed(covariance), endpointForcing: forcing.map(text),
    factorization: { diagonal: diagonal.map(text), subdiagonal: subdiagonal.map(text), determinant: text(determinant), recurrence: ds.map(text) },
    kernel: { prefactorSquaredTimesPi: text(p), diagonalExponentCoefficient: text(g), crossExponentCoefficient: text(h), actionConstant: text(constant), minimumAction: text(minimumAction), rightLogCorrection: text(endpointShift), rightLogExponential: text(sub(endpointShift, minimumAction)), symmetricUnderEndpointReversal: true },
    trace: { status: isFree ? 'divergent-unbounded-line' as const : 'finite' as const, diagonalCurvature: text(traceCurvature), squared: isFree ? null : text(div(p, traceCurvature)) },
    endpointIntegral: { prefactorSquared: text(div(p, g)), negativeLogQuadraticCoefficient: text(sub(g, div(square(h), mul(integer(4), g)))), conservesMass: isFree },
    expectedAction: { kinetic: text(kinetic), potential: text(potential), total: text(add(kinetic, potential)), minimum: text(minimumAction), fluctuation: text(div(integer(size), two)) },
    refinements, certificates: { meanResidual: meanResidual.map(text), inverseResidual: wire(inverseResidual) },
    scope: { fullMatrixIntervals: n, largestAnalyticRefinementIntervals: n * 8, covarianceRepresentation: 'indexed-exact-fractions' as const, exactFiniteRule: 'trapezoidal-potential' as const, continuousKernelEvaluated: false, transcendentalDecimalEnclosuresComputed: false, generalOperatorConvergenceProved: false, randomPathsGenerated: false },
  };
}
export type EuclideanKernelResult = ReturnType<typeof calculateEuclideanKernel>;
