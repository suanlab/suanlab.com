import test from 'node:test';
import assert from 'node:assert/strict';
import { parseNumberJsonDraft as parse } from '../src/lib/learning-json-number-input';

test('JSON support probabilities reject nonzero underflow and nonfinite numbers before a model sees zero', () => {
  for (const raw of ['1e-999', '-1e-999', '1e999'])
    assert.throws(() => parse(`{"weights":[${raw},1]}`), RangeError);
  assert.deepEqual(parse('{"weights":[5e-324,0e-999,-0.0,1e-310]}'), { weights: [Number.MIN_VALUE, 0, -0, 1e-310] });
});
test('JSON counts and mapping entries are validated before decimal fractions disappear through rounding', () => {
  for (const raw of ['{"sites":4.0000000000000000001}', '{"mapping":[0,1.0000000000000000001]}'])
    assert.throws(() => parse(raw, ['sites', 'mapping']), RangeError);
  assert.deepEqual(parse('{"sites":40e-1,"mapping":[0,1.00]}', ['sites', 'mapping']), { sites: 4, mapping: [0, 1] });
  assert.deepEqual(parse('{"weights":[1.0000000000000000001]}'), { weights: [1] });
});
test('JSON numeric-looking strings, escaped quotes, nested arrays and property ordering retain their meaning', () => {
  const source = String.raw`{"1":1,"0":2,"note":"\"1e-999\" and \\5","nested":[{"weights":[0,0.25]},null,true]}`;
  assert.deepEqual(parse(source), JSON.parse(source));
  for (const raw of ['{"x":01}', '{"x":1,}', '{"x":NaN}', '{"x":.5}']) assert.throws(() => parse(raw), SyntaxError);
});
