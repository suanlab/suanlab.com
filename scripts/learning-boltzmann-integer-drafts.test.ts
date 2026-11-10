import test from 'node:test';
import assert from 'node:assert/strict';
import { parseBoltzmannNumber, parseBoltzmannNumberList } from '../src/lib/learning-boltzmann-gibbs-protocol';
test('discrete Gibbs drafts reject fractional digits before Number rounding erases them', () => {
  for (const raw of ['2.00000000000000001', '8.00000000000000001', '16.00000000000000001', '0.99999999999999999', '1.60000000000000001e1', '4294967295.00000000001']) {
    assert.equal(Number.isInteger(Number(raw)), true, raw);
    assert.equal(parseBoltzmannNumber(raw, true).ok, false, raw);
    assert.equal(parseBoltzmannNumber(raw).ok, true, 'continuous fields retain ordinary binary64 rounding');
  }
});
test('integer mode accepts exact decimal and exponent integers including legitimate zero', () => {
  for (const [raw, value] of [['+2.000', 2], ['20e-1', 2], ['.2e1', 2], ['1.6e1', 16], ['0.000e999', 0], ['-0e-999', 0], ['4294967295', 4294967295]] as const)
    assert.deepEqual(parseBoltzmannNumber(raw, true), { ok: true, value });
  for (const raw of ['1e-999', '1e999', '2.5', 'NaN', '0x10']) assert.equal(parseBoltzmannNumber(raw, true).ok, false, raw);
});
test('observation-step lists preserve each integer lexeme rather than validating rounded values', () => {
  for (const raw of ['2.00000000000000001,3,5', '1,2.00000000000000001,4', '1 2 4.00000000000000001'])
    assert.equal(parseBoltzmannNumberList(raw, 8, true).ok, false, raw);
  assert.deepEqual(parseBoltzmannNumberList('1.000, 2e0, 4.0', 8, true), { ok: true, values: [1, 2, 4] });
  assert.deepEqual(parseBoltzmannNumberList('0.25, 1.00000000000000001', 8), { ok: true, values: [.25, 1] });
});
