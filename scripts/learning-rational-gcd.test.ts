import test from 'node:test';
import assert from 'node:assert/strict';
import { rationalGreatestCommonDivisor as gcd } from '../src/lib/learning-rational-gcd';
const Z = BigInt(0), O = BigInt(1);
function reference(a: bigint, b: bigint) {
  if (a < Z) a = -a; if (b < Z) b = -b;
  while (b !== Z) [a, b] = [b, a % b]; return a;
}
test('small signed integers, zero and reversed inputs match exact Euclidean gcd', () => {
  for (let a = -64; a <= 64; a++) for (let b = -64; b <= 64; b++) assert.equal(gcd(BigInt(a), BigInt(b)), reference(BigInt(a), BigInt(b)));
});
test('large shared factors, powers of two and adjacent integers preserve exact divisibility', () => {
  for (const bits of [32, 53, 128, 1024, 4096, 30000]) {
    const a = (O << BigInt(bits)) - O, factor = BigInt(7919);
    assert.equal(gcd(a * factor, (a + O) * factor), factor);
    assert.equal(gcd(a, O << BigInt(bits)), O); assert.equal(gcd(a, a), a);
    assert.equal(gcd(-a * factor, a * BigInt(3) * factor), a * factor);
  }
});
test('deterministically generated full-width operands match a separate Euclidean implementation', () => {
  let state = BigInt(26110631);
  const random = (bits: number) => {
    let n = Z;
    for (let i = 0; i < bits; i += 31) { state = (state * BigInt(48271)) % BigInt(2147483647); n = (n << BigInt(31)) | state; }
    return n;
  };
  for (const bits of [33, 64, 127, 256, 1024, 2048, 4096]) for (let i = 0; i < 24; i++) {
    const a = random(bits), b = random(bits), common = random(64) + O;
    assert.equal(gcd(a * common, b * common), reference(a, b) * common);
    assert.equal(gcd(b, a), reference(a, b));
  }
});
test('long Fibonacci neighbors exercise quotient batches and still return exact one', () => {
  let a = O, b = O;
  for (let i = 0; i < 12000; i++) [a, b] = [b, a + b];
  assert.equal(gcd(a, b), O); const factor = (O << BigInt(8192)) + BigInt(3);
  assert.equal(gcd(a * factor, b * factor), factor);
});
