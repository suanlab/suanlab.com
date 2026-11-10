/** Rational coefficient matching and monotonicity certificates for a convex quartic. */
import { readRegularPerturbationInput } from './learning-regular-perturbation-input';
import { f, zero, one, two, add, sub, mul, div, neg, eq, le, sum, polyMul, polyValue, type F } from './learning-quantum-hilbert-core';

const text = (x: F) => x.d === BigInt(1) ? String(x.n) : `${x.n}/${x.d}`;
const read = (s: string) => { const [n, d = '1'] = s.split('/'); return f(BigInt(n), BigInt(d)); };
const abs = (x: F) => x.n < BigInt(0) ? neg(x) : x;
const square = (x: F) => mul(x, x);
const cube = (x: F) => mul(square(x), x);
const integer = (x: number) => f(BigInt(x));

export function calculateRegularPerturbation(value: unknown) {
  const input = readRegularPerturbationInput(value);
  if (!input) throw new RangeError('Invalid regular perturbation input');
  const k = read(input.stiffness), lambda = read(input.coupling), h = read(input.source);
  const residual = (x: F) => sub(add(mul(k, x), mul(lambda, cube(x))), h);
  const scale = div(h, k), g = div(mul(lambda, square(h)), cube(k));
  // x(lambda)=sum a_n lambda^n; coefficient matching, not a closed-form inverse.
  const coefficients: F[] = [scale];
  for (let n = 1; n <= input.order; n++) {
    const contributions: F[] = [];
    for (let i = 0; i < n; i++) for (let j = 0; j < n - i; j++) {
      const l = n - 1 - i - j;
      contributions.push(mul(mul(coefficients[i], coefficients[j]), coefficients[l]));
    }
    coefficients.push(neg(div(sum(contributions), k)));
  }
  let lower = le(scale, zero) ? scale : zero;
  let upper = le(scale, zero) ? zero : scale;
  if (eq(residual(lower), zero)) upper = lower;
  else if (eq(residual(upper), zero)) lower = upper;
  // Rational endpoints certify the unique real root; no rounded Cardano formula.
  for (let i = 0; i < 48 && !eq(lower, upper); i++) {
    const midpoint = div(add(lower, upper), two), r = residual(midpoint);
    if (eq(r, zero)) { lower = midpoint; upper = midpoint; }
    else if (le(r, zero)) lower = midpoint;
    else upper = midpoint;
  }
  const rootAbsLower = le(zero, lower) ? lower : le(upper, zero) ? neg(upper) : zero;
  const rootAbsUpper = le(abs(lower), abs(upper)) ? abs(upper) : abs(lower);
  const susceptibilityLower = div(one, add(k, mul(mul(integer(3), lambda), square(rootAbsUpper))));
  const susceptibilityUpper = div(one, add(k, mul(mul(integer(3), lambda), square(rootAbsLower))));
  let power = one, partial = zero;
  const approximations = coefficients.map((coefficient, order) => {
    const term = mul(coefficient, power); partial = add(partial, term); power = mul(power, lambda);
    const r = residual(partial), distanceLower = le(partial, lower) ? sub(lower, partial) : le(upper, partial) ? sub(partial, upper) : zero;
    const distanceUpper = le(abs(sub(partial, lower)), abs(sub(partial, upper))) ? abs(sub(partial, upper)) : abs(sub(partial, lower));
    return {
      order, coefficient: text(coefficient), term: text(term), approximation: text(partial), residual: text(r),
      errorBound: text(div(abs(r), k)), energyGapBound: text(div(square(r), mul(two, k))),
      rootDistance: { lower: text(distanceLower), upper: text(distanceUpper) },
    };
  });
  const cubic = polyMul(polyMul(coefficients, coefficients), coefficients);
  const residualPolynomial = Array.from({ length: cubic.length + 1 }, (_, i) =>
    sub(add(mul(k, coefficients[i] ?? zero), i ? cubic[i - 1] : zero), i ? zero : h));
  const criticalComparison = sub(mul(integer(27), g), integer(4));
  return {
    input, dimensionlessCoupling: text(g), linearScale: text(scale),
    convergenceRegion: eq(h, zero) ? 'identically-zero' as const : eq(criticalComparison, zero) ? 'radius-boundary' as const : le(criticalComparison, zero) ? 'inside-radius' as const : 'outside-radius' as const,
    coefficients: coefficients.map(text), approximations,
    residualPolynomial: residualPolynomial.map(text),
    cancelledThroughOrder: input.order,
    evaluatedResidual: text(polyValue(residualPolynomial, lambda)),
    root: { lower: text(lower), upper: text(upper), width: text(sub(upper, lower)), lowerResidual: text(residual(lower)), upperResidual: text(residual(upper)), bisectionLimit: 48 },
    susceptibility: { lower: text(susceptibilityLower), upper: text(susceptibilityUpper) },
  };
}
