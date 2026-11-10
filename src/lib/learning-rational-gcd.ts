/** Exact gcd with Lehmer-style high-limb Euclidean batches (HAC 14.57).
 * Approximate leading limbs propose only integer unimodular transformations.
 * All accepted transformations are applied as BigInts and checked for positive
 * decreasing remainders; otherwise one exact Euclidean step is used.
 * https://cacr.uwaterloo.ca/hac/about/chap14.pdf, section 14.4.2.
 */
export function rationalGreatestCommonDivisor(a: bigint, b: bigint): bigint {
  const zero = BigInt(0);
  if (a < zero) a = -a; if (b < zero) b = -b;
  if (b > a) [a, b] = [b, a];
  const limb = BigInt(1) << BigInt(52), limit = 2 ** 26;
  while (b >= limb) {
    const hex = a.toString(16), first = parseInt(hex[0], 16);
    const length = (hex.length - 1) * 4 + (first >= 8 ? 4 : first >= 4 ? 3 : first >= 2 ? 2 : 1);
    const shift = BigInt(Math.max(0, length - 52));
    let x = Number(a >> shift), y = Number(b >> shift);
    let A = 1, B = 0, C = 0, D = 1;
    while (y + C > 0 && y + D > 0) {
      const q = Math.floor((x + A) / (y + C)), other = Math.floor((x + B) / (y + D));
      if (q !== other || q < 1 || Math.abs(q * C) > 2 ** 50 || Math.abs(q * D) > 2 ** 50) break;
      const nextC = A - q * C, nextD = B - q * D, nextY = x - q * y;
      if (![nextC, nextD].every(v => Number.isSafeInteger(v) && Math.abs(v) <= limit) || !Number.isSafeInteger(nextY) || nextY < 0) break;
      [A, B, C, D] = [C, D, nextC, nextD]; [x, y] = [y, nextY];
    }
    if (B !== 0) {
      const nextA = BigInt(A) * a + BigInt(B) * b, nextB = BigInt(C) * a + BigInt(D) * b;
      if (nextA >= nextB && nextB >= zero && nextA <= a && nextB < b) {
        [a, b] = [nextA, nextB]; continue;
      }
    }
    [a, b] = [b, a % b];
  }
  while (b !== zero) [a, b] = [b, a % b];
  return a;
}
