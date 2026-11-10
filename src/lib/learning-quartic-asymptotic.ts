import { quarticAsymptoticContract } from './learning-quartic-asymptotic-contract';
/** Zero-dimensional normalized quartic integral; exact finite asymptotic algebra. */
import { readQuarticAsymptoticInput } from './learning-quartic-asymptotic-input';
import { encloseQuarticHalfIntegral } from './learning-quartic-asymptotic-integral';
import { f, zero, one, add, sub, mul, div, neg, eq, le, type F } from './learning-quantum-hilbert-core';
const B = BigInt;
const read = (s: string) => { const [n, d = '1'] = s.split('/'); return f(B(n), B(d)); };
const text = (x: F) => x.d === B(1) ? String(x.n) : `${x.n}/${x.d}`;
const abs = (x: F) => x.n < B(0) ? neg(x) : x;
const min = (a: F, b: F) => le(a, b) ? a : b, max = (a: F, b: F) => le(a, b) ? b : a;
let gaussian: ReturnType<typeof encloseQuarticHalfIntegral> | undefined;
function globallyBestBoundOrder(g: F) {
  if (g.n === B(0)) return 0;
  const rises = (order: bigint) => { const m = order + B(1); return g.n * (B(4) * m + B(1)) * (B(4) * m + B(3)) >= B(4) * (m + B(1)) * g.d; };
  let lower = B(0), upper = g.d / (B(4) * g.n) + B(1);
  while (lower < upper) { const middle = (lower + upper) / B(2); if (rises(middle)) upper = middle; else lower = middle + B(1); }
  return Number(lower); // <= 250000000000 for the declared bounded fraction input.
}
export function calculateQuarticAsymptotic(value: unknown) {
  const input = readQuarticAsymptoticInput(value); if (!input) throw new RangeError('Invalid quartic asymptotic input');
  const g = read(input.coupling); gaussian ??= encloseQuarticHalfIntegral(zero);
  const integral = eq(g, zero) ? gaussian : encloseQuarticHalfIntegral(g);
  const referenceLower = eq(g, zero) ? one : div(integral.lower, gaussian.upper), referenceUpper = eq(g, zero) ? one : div(integral.upper, gaussian.lower);
  const coefficients: F[] = [one], terms: F[] = [one], partials: F[] = [one];
  for (let n = 0; n <= input.order; n++) {
    const coefficient = neg(mul(coefficients[n], f(B((4 * n + 1) * (4 * n + 3)), B(4 * (n + 1)))));
    coefficients.push(coefficient); terms.push(mul(terms[n], neg(mul(g, f(B((4 * n + 1) * (4 * n + 3)), B(4 * (n + 1)))))));
    partials.push(add(partials[n], terms[n + 1]));
  }
  let envelopeLower = zero, envelopeUpper = one;
  partials.forEach((p, n) => { if (n % 2) envelopeLower = max(envelopeLower, p); else envelopeUpper = min(envelopeUpper, p); });
  const combinedLower = max(referenceLower, envelopeLower), combinedUpper = min(referenceUpper, envelopeUpper);
  if (!le(combinedLower, combinedUpper)) throw Error('Independent integral and asymptotic certificates disagree');
  const approximations = partials.slice(0, input.order + 1).map((p, order) => {
    const omitted = abs(terms[order + 1]), dl = le(p, referenceLower) ? sub(referenceLower, p) : le(referenceUpper, p) ? sub(p, referenceUpper) : zero;
    const du = max(abs(sub(p, referenceLower)), abs(sub(p, referenceUpper)));
    if (!le(dl, min(du, omitted))) throw Error('Empty certified error interval');
    return { order, coefficient: text(coefficients[order]), term: text(terms[order]), partial: text(p), firstOmittedMagnitude: text(omitted),
      position: eq(g, zero) ? 'exact' as const : order % 2 ? 'below' as const : 'above' as const,
      quadratureDistance: { lower: text(dl), upper: text(du) }, errorInterval: { lower: text(dl), upper: text(min(du, omitted)) } };
  });
  let best = 0; for (let n = 1; n <= input.order; n++) if (!eq(abs(terms[n + 1]), abs(terms[best + 1])) && le(abs(terms[n + 1]), abs(terms[best + 1]))) best = n;
  const globalBest = globallyBestBoundOrder(g);
  return { input, series: eq(g, zero) ? 'exact-gaussian' as const : 'divergent-asymptotic' as const,
    coefficients: coefficients.map(text), approximations,
    quadratureInterval: { lower: text(referenceLower), upper: text(referenceUpper), width: text(sub(referenceUpper, referenceLower)) },
    asymptoticEnvelope: { lower: text(envelopeLower), upper: text(envelopeUpper) }, combinedInterval: { lower: text(combinedLower), upper: text(combinedUpper) },
    bestBoundOrderWithinBudget: best, globallyBestBoundOrder: globalBest, budgetReachesBestBound: globalBest <= input.order,
    quadrature: { ...integral.certificate }, gaussianNormalization: { ...gaussian.certificate },
    contract: quarticAsymptoticContract };
}
