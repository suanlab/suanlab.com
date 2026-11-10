import { mkdirSync as ensureTestArtifactDirectory } from 'node:fs';
import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { field, fieldAdd, fieldSub, fieldMul, fieldDivide, fieldSign, expectedField, expectedNormalizedField, type WireField } from '../src/lib/learning-many-body-field-wire';
import { f } from '../src/lib/learning-tensor-wire-math';
import { matchManyBodyExpected } from '../src/lib/learning-many-body-wire-match';
import { snapshotManyBodyWire } from '../src/lib/learning-many-body-owned-wire';
import { bObserve, bNormalized, bComplex } from '../src/lib/learning-boson-algebraic';
import { exact } from '../src/lib/learning-quantum-hilbert-core';
ensureTestArtifactDirectory('.runtime/learning/many-body-boson-response', { recursive: true });


const runtime = '.runtime/learning/many-body-boson-response';
const refs = JSON.parse(readFileSync('scripts/fixtures/boson-algebraic-reference.json', 'utf8'));
const parse = (s: string) => { const [n, d = '1'] = s.split('/'); return f(BigInt(n), BigInt(d)); };
const vector = (a: string[]) => a.map(parse) as unknown as WireField;
let coefficientComparisons = 0, signsCompared = 0, normalizedChecks = 0;
for (const ref of refs.cases) test(`wire field arithmetic against independent SymPy: ${ref.label}`, () => {
  const a = vector(ref.a), b = vector(ref.b);
  const values: Record<string, WireField | null> = { a, b, add: fieldAdd(a, b), subtract: fieldSub(a, b), multiply: fieldMul(a, b), divide: fieldSign(b) ? fieldDivide(a, b) : null };
  for (const [name, value] of Object.entries(values)) {
    const expected = ref.expected[name];
    if (expected === null) { assert.equal(value, null); continue; }
    assert(value); assert.equal(fieldSign(value), expected.exactSign); signsCompared++;
    value.forEach((v, i) => { const e = parse(expected.coefficients[i]); assert.equal(v.n * e.d, e.n * v.d); coefficientComparisons++; });
    assert(matchManyBodyExpected(snapshotManyBodyWire(bObserve(value)), expectedField(value)));
  }
  // Field-valued normalized coordinates may cancel strongly. Compare to the
  // exact algebraic target without importing a square-root/interval engine.
  for (const norm of [f(BigInt(1)), f(BigInt(3), BigInt(2)), f(BigInt(1), BigInt(1000000))]) {
    const actual = bNormalized(bComplex(a), norm).real;
    assert(matchManyBodyExpected(snapshotManyBodyWire(actual), expectedNormalizedField(a, norm))); normalizedChecks++;
  }
});

test('field schema rejects wrong coefficients, exact signs, observations and rounded-root substitutions', () => {
  const a = field(f(BigInt(0)), f(BigInt(1))), expected = expectedField(a), actual = bObserve(a);
  for (const edit of [(x: any) => { x.exactSign = -1; }, (x: any) => { x.coefficients[1] = exact(BigInt(2), BigInt(1)); }, (x: any) => { x.observation = exact(BigInt(141421356), BigInt(100000000)); }, (x: any) => { x.coefficients.extra = 0; }, (x: any) => { x.extra = true; }]) {
    const x = structuredClone(actual); edit(x);
    try { assert.equal(matchManyBodyExpected(snapshotManyBodyWire(x), expected), false); } catch (error) { assert(error instanceof RangeError); }
  }
  const zero = field(f(BigInt(0))); assert(matchManyBodyExpected(snapshotManyBodyWire(bObserve(zero)), expectedField(zero)));
  assert.throws(() => fieldDivide(a, zero));
});
after(() => writeFileSync(`${runtime}/field-metrics.json`, JSON.stringify({ referenceCases: refs.cases.length, coefficientComparisons, signsCompared, normalizedChecks }, null, 2) + '\n'));
