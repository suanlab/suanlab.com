/** Independent dyadic quadrature verification. Uses an alternating negative
 * exponential, binomial expansion of the action, and diagonal polynomial
 * products. No producer quadrature or transcendental engine is imported.
 */
import { Rational as R } from './learning-graph-gauge-validation-math';
const B = BigInt, precision = 192, S = B(1) << B(precision), zero = new R(B(0)), one = new R(B(1));
type I = readonly [bigint, bigint];
const ceiling = (n: bigint, d: bigint) => (n + d - B(1)) / d;
const plus = (a: I, b: I): I => [a[0] + b[0], a[1] + b[1]];
const negate = (a: I): I => [-a[1], -a[0]];
const integer = (n: number) => new R(B(n));
function power(a: R, n: number) { let value = one; for (let i = 0; i < n; i++) value = value.mul(a); return value; }
function choose(n: number, k: number) { let value = 1; for (let j = 0; j < k; j++) value = value * (n - j) / (j + 1); return integer(value); }

export function independentQuarticExponential(numerator: bigint, denominator: bigint): I {
  if (numerator === B(0)) return [S, S];
  if (numerator < B(0) || denominator <= B(0)) throw Error('Positive exponent argument required');
  let squarings = 0; while (B(8) * numerator > (denominator << B(squarings))) squarings++;
  for (let bits = 256; bits <= 2048; bits += 64) {
    const scale = B(1) << B(bits), d = denominator << B(squarings), x: I = [numerator * scale / d, ceiling(numerator * scale, d)];
    const multiply = (a: I, b: I): I => [a[0] * b[0] / scale, ceiling(a[1] * b[1], scale)];
    const evenDegree = 2 * Math.ceil((bits + 12) / 6);
    let term: I = [scale, scale], sum: I = term, evenUpper = scale;
    for (let j = 1; j <= evenDegree + 1; j++) {
      const product = multiply(term, x); term = [product[0] / B(j), ceiling(product[1], B(j))];
      sum = plus(sum, j % 2 ? negate(term) : term);
      if (j === evenDegree) evenUpper = sum[1];
    }
    let result: I = [sum[0], evenUpper];
    if (result[0] < B(0)) throw Error('Negative reduced exponential');
    for (let j = 0; j < squarings; j++) result = multiply(result, result);
    const lo = result[0] >> B(bits - precision), hi = result[1] >> B(bits - precision);
    if (lo === hi) return [lo, lo + B(1)];
  }
  throw Error('Independent exponential cell unresolved');
}

export function independentlyEncloseQuarticIntegral(g: R) {
  const step = new R(B(1), B(16));
  const exponent = (x: R) => power(x, 2).div(integer(2)).add(g.mul(power(x, 4)).div(integer(4)));
  const dyadic = (x: R): I => [x.n * S / x.d, ceiling(x.n * S, x.d)];
  const product = (a: I, b: I): I => [a[0] * b[0] / S, ceiling(a[1] * b[1], S)];
  let lo = B(0), hi = B(0), tiny = 0;
  for (let j = 0; j < 256; j++) {
    const at = step.mul(integer(j)), action = exponent(at), factor = independentQuarticExponential(action.n, action.d);
    if (factor[1] <= B(1)) { hi += factor[1]; tiny++; continue; }
    // Expand (a+δu)^2/2+g(a+δu)^4/4, subtract its constant.
    const delta: I[] = Array.from({ length: 5 }, (_, m) => {
      if (!m) return [B(0), B(0)];
      const quadratic = m <= 2 ? choose(2, m).mul(power(at, 2 - m)).div(integer(2)) : zero;
      const quartic = choose(4, m).mul(g).mul(power(at, 4 - m)).div(integer(4));
      return dyadic(quadratic.add(quartic).mul(power(step, m)));
    });
    let coefficients: I[] = [[S, S]], integral: I = [S, S], boundAbove = S, factorial = B(1);
    for (let n = 1; n <= 33; n++) {
      const previous = coefficients;
      coefficients = Array.from({ length: 4 * n + 1 }, (_, degree) => {
        let coefficient: I = [B(0), B(0)];
        for (let d = 1; d <= 4; d++) if (degree >= d && degree - d < previous.length) coefficient = plus(coefficient, product(previous[degree - d], delta[d]));
        return coefficient;
      });
      factorial *= B(n);
      const term = coefficients.reduce<I>((total, coefficient, degree) => {
        const denominator = B(degree + 1) * factorial;
        return plus(total, [coefficient[0] / denominator, ceiling(coefficient[1], denominator)]);
      }, [B(0), B(0)]);
      integral = plus(integral, n % 2 ? negate(term) : term);
      if (n === 32) boundAbove = integral[1];
    }
    const lower = factor[0] * (integral[0] < B(0) ? B(0) : integral[0]) / S;
    const upper = ceiling(factor[1] * boundAbove, S);
    lo += lower; hi += upper < factor[1] ? upper : factor[1];
  }
  const L = integer(16), action = exponent(L), decay = independentQuarticExponential(action.n, action.d);
  const tail = new R(decay[1], S).div(L.add(g.mul(power(L, 3))));
  const lower = new R(lo, S).mul(step), upper = new R(hi, S).mul(step).add(tail);
  return { lower, upper, certificate: { cutoff: 16, cells: 256, precisionBits: 192, localTaylorEvenDegree: 32, localTaylorOddDegree: 33,
    omittedTinyPanels: tiny, halfLineLower: lower.text(), halfLineUpper: upper.text(), tailUpper: tail.text() } };
}
