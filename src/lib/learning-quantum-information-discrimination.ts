/** Exact rational-qubit discrimination with directed algebraic observations.
 * Worker/CLI foundation only; no public input ownership or Worker protocol yet. */
import { f, zero, one, two, add, sub, mul, div, neg, eq, le, sum, c, cq, q,
  sqrtRange, observe, type F, type R } from './learning-quantum-hilbert-core';
type Bloch = readonly [F, F, F];
const abs = (x: F) => x.n < BigInt(0) ? neg(x) : x;
const half = (x: F) => div(x, two);
const dot = (a: readonly F[], b: readonly F[]) => sum(a.map((x, i) => mul(x, b[i])));
const point = (x: F): R => ({ lower: x, upper: x });
const plus = (a: R, b: R): R => ({ lower: add(a.lower, b.lower), upper: add(a.upper, b.upper) });
const minus = (a: R, b: R): R => ({ lower: sub(a.lower, b.upper), upper: sub(a.upper, b.lower) });
const scale = (a: R, x: F): R => x.n < BigInt(0)
  ? { lower: mul(a.upper, x), upper: mul(a.lower, x) }
  : { lower: mul(a.lower, x), upper: mul(a.upper, x) };
const dividePositive = (x: F, denominator: R): R => {
  if (!le(zero, denominator.lower) || eq(denominator.lower, zero)) throw new RangeError('Unresolved positive denominator');
  return x.n < BigInt(0) ? { lower: div(x, denominator.lower), upper: div(x, denominator.upper) }
    : { lower: div(x, denominator.upper), upper: div(x, denominator.lower) };
};
const unit = (x: R): R => {
  const lower = le(x.lower, zero) ? zero : x.lower, upper = le(one, x.upper) ? one : x.upper;
  if (!le(lower, upper)) throw new RangeError('Probability enclosure misses the unit interval');
  return { lower, upper };
};
function requireFraction(x: F) {
  if (!x || typeof x.n !== 'bigint' || typeof x.d !== 'bigint' || x.d <= BigInt(0)
    || abs(x).n.toString().length > 32 || x.d.toString().length > 32)
    throw new RangeError('Rational coefficients require at most32 digits and a positive denominator');
}
function requireBloch(r: Bloch) {
  if (!Array.isArray(r) || r.length !== 3) throw new RangeError('A qubit Bloch vector has three components');
  Array.from(r).forEach(requireFraction);
  if (!le(dot(r, r), one)) throw new RangeError('Bloch vector must lie inside the unit ball');
}
const density = (r: readonly F[]) => [
  [cq(c(half(add(one, r[2])))), cq(c(half(r[0]), neg(half(r[1]))))],
  [cq(c(half(r[0]), half(r[1]))), cq(c(half(sub(one, r[2]))))],
];

export function binaryQubitDiscrimination(r0: Bloch, r1: Bloch, prior: F) {
  requireBloch(r0); requireBloch(r1); requireFraction(prior);
  if (!le(zero, prior) || !le(prior, one)) throw new RangeError('Prior must lie in[0,1]');
  const other = sub(one, prior), t = sub(prior, other), absoluteTrace = abs(t);
  const vector = r0.map((x, i) => sub(mul(prior, x), mul(other, r1[i])));
  const d = dot(vector, vector), t2 = mul(t, t), determinant = div(sub(t2, d), f(BigInt(4)));
  const indefinite = !le(d, t2), zeroDifference = eq(d, zero) && eq(t, zero);
  const strategy = zeroDifference ? 'uninformative' : indefinite ? 'spectral-projector' : t.n >= BigInt(0) ? 'always-zero' : 'always-one';
  const root = sqrtRange(d), traceNorm = indefinite ? root : point(absoluteTrace);
  const success = unit(scale(plus(point(one), traceNorm), half(one)));
  const baseline = le(prior, other) ? other : prior;
  // Rationalization retains a tiny positive gain above the prior-only strategy.
  const denominator = plus(root, point(absoluteTrace));
  const gain = !indefinite ? point(zero) : denominator.lower.n === BigInt(0)
    ? scale(root, half(one)) : scale(dividePositive(sub(d, t2), denominator), half(one));
  const axis = indefinite && root.lower.n > BigInt(0) ? vector.map(x => dividePositive(x, root)) : null;
  const firstOutcome = (r: Bloch): R => strategy === 'spectral-projector'
    ? axis === null ? { lower: zero, upper: one } : unit(scale(plus(point(one), axis.reduce((s, x, i) => plus(s, scale(x, r[i])), point(zero))), half(one)))
    : point(strategy === 'always-one' ? zero : one);
  const conditional = [firstOutcome(r0), firstOutcome(r1)].map(p => [p, minus(point(one), p)]);
  const achieved = plus(scale(conditional[0][0], prior), scale(conditional[1][1], other));
  const delta = [
    [cq(c(half(add(t, vector[2])))), cq(c(half(vector[0]), neg(half(vector[1]))))],
    [cq(c(half(vector[0]), half(vector[1]))), cq(c(half(sub(t, vector[2]))))],
  ];
  const norm0 = dot(r0, r0), norm1 = dot(r1, r1), difference = r0.map((x, i) => sub(x, r1[i]));
  const distanceSquared = div(dot(difference, difference), f(BigInt(4)));
  const fidelityRadicand = mul(sub(one, norm0), sub(one, norm1));
  const fidelity = unit(scale(plus(point(add(one, dot(r0, r1))), sqrtRange(fidelityRadicand)), half(one)));
  const realComplex = (re: R, im = point(zero)) => ({ real: observe(re), imaginary: observe(im) });
  const constant = strategy === 'always-one' ? zero : one;
  const effects = axis === null ? strategy === 'spectral-projector' ? null : [
    [realComplex(point(constant)), realComplex(point(zero))],
    [realComplex(point(zero)), realComplex(point(constant))],
  ] : [
    [realComplex(scale(plus(point(one), axis[2]), half(one))), realComplex(scale(axis[0], half(one)), scale(axis[1], neg(half(one))))],
    [realComplex(scale(axis[0], half(one)), scale(axis[1], half(one))), realComplex(scale(minus(point(one), axis[2]), half(one)))],
  ];
  return {
    schemaVersion: 1, dimension: 2, prior: q(prior), otherPrior: q(other),
    density0: density(r0), density1: density(r1), bloch0: r0.map(q), bloch1: r1.map(q),
    weightedDifference: delta, weightedTrace: q(t), weightedBloch: vector.map(q),
    weightedRadiusSquared: q(d), weightedDeterminant: q(determinant),
    eigenvalues: [observe(scale(minus(point(t), root), half(one))), observe(scale(plus(point(t), root), half(one)))],
    traceNorm: observe(traceNorm), optimalSuccess: observe(success), priorOnlySuccess: q(baseline),
    gainAbovePriorOnly: observe(gain), strictlyBeatsPriorOnly: indefinite,
    strategy, uniqueOptimalEffect: determinant.n !== BigInt(0),
    optimalGuessZeroEffect: effects, measurementAxis: axis?.map(observe) ?? null,
    conditionalOutcomeProbabilities: conditional.map(row => row.map(x => observe(unit(x)))),
    achievedSuccess: observe(unit(achieved)), successResidual: observe(minus(achieved, success)),
    traceDistanceSquared: q(distanceSquared), traceDistance: observe(sqrtRange(distanceSquared)),
    squaredUhlmannFidelity: observe(fidelity), fidelityConvention: 'squared-uhlmann-fidelity',
    exactStateEquality: r0.every((x, i) => eq(x, r1[i])),
    exactOrthogonalSupports: eq(distanceSquared, one),
    approximationContract: 'Exact rational Bloch inputs and physicality/classification. General square roots use directed512-bit rational brackets; matrix effects may contain algebraic intervals. A zero eigenvalue makes the optimal effect nonunique. The uninformative case chooses the identity effect canonically. Interval bounds and positive-gain classifications are retained separately from rounded success. No entropy, accessible-information optimization, channel discrimination, public input reader, Worker or UI is supplied.',
  };
}
