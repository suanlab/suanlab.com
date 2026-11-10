/** Exact static/frequency preparation for a protocol-dependent three-state CTMC.
 * This is not yet a registered lesson or a time-domain/finite-input solver. */
import { markovFraction as fraction, exactMarkovObservation, type MarkovFraction as F } from './learning-finite-markov-exact';

export interface FiniteStateResponseParameters {
  a: number; b: number; c: number; d: number; e: number; f: number;
  rateDenominator: number;
  kineticNumerator: number; kineticDenominator: number;
  frequencyNumerator: number; frequencyDenominator: number;
}

const keys = ['a', 'b', 'c', 'd', 'e', 'f', 'rateDenominator', 'kineticNumerator', 'kineticDenominator', 'frequencyNumerator', 'frequencyDenominator'] as const;
const zero = fraction(BigInt(0)), one = fraction(BigInt(1)), half = fraction(BigInt(1), BigInt(2));
const add = (x: F, y: F) => fraction(x.n * y.d + y.n * x.d, x.d * y.d);
const neg = (x: F) => fraction(-x.n, x.d);
const sub = (x: F, y: F) => add(x, neg(y));
const mul = (x: F, y: F) => fraction(x.n * y.n, x.d * y.d);
const div = (x: F, y: F) => fraction(x.n * y.d, x.d * y.n);
const sum = (xs: readonly F[]) => xs.reduce(add, zero);
const dot = (xs: readonly F[], ys: readonly F[]) => sum(xs.map((x, i) => mul(x, ys[i])));
const output = (x: F) => exactMarkovObservation(x.n, x.d);

export function isFiniteStateResponseParameters(value: unknown): value is FiniteStateResponseParameters {
  if (!value || typeof value !== 'object' || Object.getPrototypeOf(value) !== Object.prototype || Reflect.ownKeys(value).length !== keys.length) return false;
  const p = value as Record<string, unknown>;
  return keys.every(k => {
    const x = p[k], min = k === 'kineticNumerator' ? -1000000 : k === 'frequencyNumerator' ? 0 : 1;
    return Object.prototype.hasOwnProperty.call(p, k) && typeof x === 'number' && Number.isSafeInteger(x) && x >= min && x <= 1000000;
  });
}

/** Pivoting is exact: no tolerance can discard a small nonzero rational pivot. */
function solve(matrix: readonly (readonly F[])[], rhs: readonly F[]): F[] {
  const n = rhs.length, rows = matrix.map((row, i) => [...row, rhs[i]]);
  for (let column = 0; column < n; column++) {
    const pivot = rows.findIndex((row, i) => i >= column && row[column].n !== BigInt(0));
    if (pivot < 0) throw new RangeError('Singular response system');
    [rows[column], rows[pivot]] = [rows[pivot], rows[column]];
    const scale = rows[column][column];
    rows[column] = rows[column].map(x => div(x, scale));
    for (let i = 0; i < n; i++) if (i !== column) {
      const coefficient = rows[i][column];
      rows[i] = rows[i].map((x, j) => sub(x, mul(coefficient, rows[column][j])));
    }
  }
  return rows.map(row => row[n]);
}

export function finiteStateLinearResponseModel(input: FiniteStateResponseParameters) {
  if (!isFiniteStateResponseParameters(input)) throw new RangeError('Eleven owned integer fields; positive rates and denominators, signed kinetic coefficient, nonnegative angular frequency; magnitudes at most1000000');
  const parameters = Object.freeze(Object.fromEntries(keys.map(k => [k, input[k] + 0]))) as unknown as Readonly<FiniteStateResponseParameters>;
  const p = parameters, rate = (x: number) => fraction(BigInt(x), BigInt(p.rateDenominator));
  const [a, b, c, d, e, f] = [p.a, p.b, p.c, p.d, p.e, p.f].map(rate);
  const eta = fraction(BigInt(p.kineticNumerator), BigInt(p.kineticDenominator));
  const omega = fraction(BigInt(p.frequencyNumerator), BigInt(p.frequencyDenominator));
  const Q = [[neg(add(a, f)), a, f], [d, neg(add(d, b)), b], [c, e, neg(add(c, e))]];
  const tau = [sum([mul(c, d), mul(d, e), mul(b, c)]), sum([mul(a, e), mul(a, c), mul(e, f)]), sum([mul(b, f), mul(a, b), mul(d, f)])];
  const Z = sum(tau), pi = tau.map(t => div(t, Z));
  const forwardDerivative = mul(a, add(eta, half)), reverseDerivative = mul(d, sub(eta, half));
  const Q1 = [[neg(forwardDerivative), forwardDerivative, zero], [reverseDerivative, neg(reverseDerivative), zero], [zero, zero, zero]];
  const source = [0, 1, 2].map(j => sum(pi.map((x, i) => mul(x, Q1[i][j]))));
  const current = sub(mul(pi[0], a), mul(pi[1], d));
  const prime = solve([[Q[0][0], Q[1][0], Q[2][0]], [Q[0][1], Q[1][1], Q[2][1]], [one, one, one]], [neg(source[0]), neg(source[1]), zero]);
  const observables = [0, 1, 2].map(index => {
    const observable = [0, 1, 2].map(i => i === index ? one : zero);
    const centered = observable.map(x => sub(x, pi[index]));
    const poisson = solve([Q[0], Q[1], pi], [neg(centered[0]), neg(centered[1]), zero]);
    const staticResponse = dot(source, poisson);
    let real: F, imaginary: F;
    if (omega.n === BigInt(0)) {
      // The full generator is singular at DC; never pass it to the resolvent solver.
      real = staticResponse; imaginary = zero;
    } else {
      const system = Array.from({ length: 6 }, (_, i) => Array.from({ length: 6 }, (_, j) => {
        if (i < 3 && j < 3 || i >= 3 && j >= 3) return neg(Q[i % 3][j % 3]);
        return i % 3 === j % 3 ? (i < 3 ? neg(omega) : omega) : zero;
      }));
      const z = solve(system, [...observable, zero, zero, zero]);
      real = dot(source, z.slice(0, 3)); imaginary = dot(source, z.slice(3));
    }
    return {
      state: index, stationaryOccupancy: output(pi[index]), initialKernel: output(source[index]),
      poissonSolution: poisson.map(output), stationaryDerivative: output(prime[index]), staticSusceptibility: output(staticResponse),
      staticIdentityResidual: output(sub(staticResponse, prime[index])),
      frequencySusceptibility: { real: output(real), imaginary: output(imaginary), magnitudeSquared: output(add(mul(real, real), mul(imaginary, imaginary))), phaseDefined: real.n !== BigInt(0) || imaginary.n !== BigInt(0) },
    };
  });
  return {
    parameters, stage: 'static-and-frequency-preparation' as const, generator: Q.map(row => row.map(output)),
    generatorDerivative: Q1.map(row => row.map(output)), stationaryLaw: pi.map(output), stationaryCurrent: output(current),
    signedSource: source.map(output), conjugateObservable: source.map((x, i) => output(div(x, pi[i]))),
    sourceMassResidual: output(sum(source)), stationaryDerivativeMassResidual: output(sum(prime)),
    sourceProtocolContribution: [neg(mul(eta, current)), mul(eta, current), zero].map(output), observables,
    classification: { strictlyPositiveRates: true, detailedBalance: current.n === BigInt(0), stationaryInitialization: true, frequencyIsAngular: true, frequencyUsesStaticLimit: omega.n === BigInt(0), sampledPaths: false, thermalBathSpecified: false, timeDomainImplemented: false, finiteAmplitudeImplemented: false },
    approximationContract: 'Exact rational row-law three-state CTMC. Clockwise rates a,b,c and reverse d,e,f in s^-1. Dimensionless h changes only q01=a exp((eta+1/2)h),q10=d exp((eta-1/2)h); diagonals conserve mass. Initial law is unperturbed pi. All occupancy sources, initial right-limit kernels, normalized Poisson solutions, stationary derivatives and complex susceptibilities are exact. Nonzero angular frequency uses exp(i Omega t) input and exp(-i Omega lag) retarded transform; zero frequency uses normalized static response, not an inverse of singular Q. Magnitude squared is exact; phase is undefined for zero complex response and no rounded phase is computed here. This preparation does not implement time curves, finite-amplitude exponentials, a worker, manuscript loading or an interactive lesson. No heat or temperature is inferred.',
  };
}
