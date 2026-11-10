import test from 'node:test';
import assert from 'node:assert/strict';
import { roundInformationTerm as round, informationArithmeticBits } from '../src/lib/learning-information-arithmetic';
const Z = BigInt(0), O = BigInt(1), T = BigInt(2), bits = BigInt(informationArithmeticBits);
function bound(n: bigint, d: bigint) {
  const r = round(n, d), gap = r.n * d - n * r.d, absolute = gap < Z ? -gap : gap;
  assert.ok((absolute << bits) <= n * r.d); // Exact relative error bound; no floating conversion.
  assert.ok(r.n > Z && r.d > Z); assert.equal(r.d & (r.d - O), Z);
  assert.ok(r.n.toString(2).replace(/0+$/, '').length <= informationArithmeticBits);
}
test('term rounding has the exact relative-error bound over wide exponents and unreduced inputs', () => {
  for (const shift of [-12000, -4096, -1074, -1, 0, 53, 1024, 4096]) for (let j = 1; j <= 30; j++) {
    const n = (O << BigInt(j * 11)) + BigInt(j), d = (O << BigInt(j * 7)) + BigInt(j + 3);
    bound(shift < 0 ? n : n << BigInt(shift), shift < 0 ? d << BigInt(-shift) : d);
    bound(n * BigInt(7919), d * BigInt(7919));
  }
});
test('halfway cases choose even mantissas at several exponent scales', () => {
  const even = O << (bits - O);
  for (const shift of [0, 100, 4096]) {
    const scale = O << BigInt(shift);
    assert.deepEqual(round((T * even + O) * scale, T), { n: even * scale, d: O });
    assert.deepEqual(round((T * even + BigInt(3)) * scale, T), { n: (even + T) * scale, d: O });
  }
});
test('zero stays exact and arbitrarily tiny positive terms stay positive', () => {
  assert.deepEqual(round(Z, BigInt(7)), { n: Z, d: O });
  assert.deepEqual(round(O, O << BigInt(12000)), { n: O, d: O << BigInt(12000) });
  assert.throws(() => round(-O, O), RangeError); assert.throws(() => round(O, Z), RangeError);
  assert.throws(() => round(O, -O), RangeError);
});
