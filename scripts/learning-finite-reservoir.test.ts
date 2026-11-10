import test from 'node:test';
import assert from 'node:assert/strict';
import references from './fixtures/finite-reservoir-independent.json';
import { finiteReservoirModel as evaluate } from '../src/lib/learning-finite-reservoir';
import type { BinaryObservation } from '../src/lib/learning-binary-products';
const fraction = (s: string) => s.split('/').map(BigInt);
function near(actual: BinaryObservation, expected: number) {
  assert.notEqual(actual.value, null);
  if (expected === 0) { assert.equal(actual.value, 0); assert.equal(actual.sign, 0); }
  else assert.ok(Math.abs(actual.value! - expected) <= Math.abs(expected) * 8e-12, `${actual.value} vs ${expected}`);
}
function rationalNear(actual: BinaryObservation, expected: string) {
  const [n, d] = expected.split('/').map(Number); near(actual, n / d);
}
for (const reference of references.cases) {
  const { subsystemSites: n, reservoirSites: b, totalExcitations: k } = reference;
  test(`independent finite reservoir ${n}/${b}/${k}`, () => {
    const result = evaluate(n, b, k);
    assert.equal(result.wholeShellCount, reference.wholeShellCount);
    assert.deepEqual(result.distribution.map(row => row.compatibleMicrostates), reference.blockCounts);
    const shell = BigInt(result.wholeShellCount), canonical = BigInt(result.canonicalCommonDenominator);
    result.distribution.forEach((row, a) => {
      const [pn, pd] = fraction(reference.microcanonicalMarginal[a]);
      const [qn, qd] = fraction(reference.matchingCanonical[a]);
      assert.equal(BigInt(row.compatibleMicrostates) * pd, pn * shell);
      assert.equal(BigInt(row.canonicalNumerator) * qd, qn * canonical);
    });
    for (const key of ['meanExcitations', 'varianceExcitations', 'canonicalVariance', 'totalVariation'] as const)
      rationalNear(result[key], reference[key]);
    near(result.entropyNats, Number(reference.entropyNats));
    near(result.microToCanonicalKlNats, Number(reference.microToCanonicalKlNats));
    assert.equal(result.canonicalToMicroKlNats.status, reference.canonicalToMicroKlStatus);
    if (reference.canonicalToMicroKlNats === null) assert.equal(result.canonicalToMicroKlNats.value, null);
    else near(result.canonicalToMicroKlNats, Number(reference.canonicalToMicroKlNats));
  });
}
test('equal means do not imply matching fluctuation laws; reverse KL sees forbidden shells', () => {
  const small = evaluate(2, 2, 2);
  near(small.meanExcitations, 1); near(small.varianceExcitations, 1 / 3); near(small.canonicalVariance, .5);
  near(small.totalVariation, 1 / 6); assert.equal(small.exactlyEqual, false);
  const fixed = evaluate(2, 0, 1);
  near(fixed.varianceExcitations, 0); near(fixed.microToCanonicalKlNats, Math.LN2);
  assert.equal(fixed.canonicalToMicroKlNats.status, 'infinite');
});
test('fixed subsystem at half filling has the exact finite reservoir TV sequence', () => {
  let previous = 1;
  for (const b of [2, 6, 18, 98, 998]) {
    const result = evaluate(2, b, (b + 2) / 2);
    near(result.totalVariation, 1 / (2 * (b + 1)));
    assert.ok(result.totalVariation.value! < previous); previous = result.totalVariation.value!;
  }
});
test('invalid sizes, empty total and inadmissible excitation counts are rejected', () => {
  for (const args of [[0, 0, 0], [-1, 2, 0], [1.5, 3, 1], [4096, 1, 1], [1, 1, 3], [2, 2, -.1], [NaN, 1, 0], [1, Infinity, 0]])
    assert.throws(() => evaluate(args[0], args[1], args[2]), RangeError);
});
test('largest total keeps rare positive shell probabilities separate from true zero', () => {
  const result = evaluate(32, 4064, 1);
  assert.equal(result.distribution[2].microcanonical.sign, 0);
  assert.equal(result.distribution[32].canonical.sign, 1);
  assert.ok(result.microToCanonicalKlNats.value! > 0);
  assert.equal(result.canonicalToMicroKlNats.status, 'infinite');
});
test('4096-site deterministic shell obeys exact-point-mass TV and KL identities', () => {
  const result = evaluate(4096, 0, 2048);
  const q = result.distribution[2048].canonical.value!;
  near(result.totalVariation, 1 - q);
  near(result.microToCanonicalKlNats, -Math.log(q));
  near(result.entropyNats, 0); near(result.varianceExcitations, 0);
  assert.equal(result.distribution[4096].canonical.value, null);
  assert.equal(result.distribution[4096].canonical.sign, 1);
  assert.equal(result.distribution[4096].microcanonical.sign, 0);
  assert.ok(result.distribution[4096].canonical.logMagnitude! < -2800);
});
test('rare canonical outcomes stay positive even below the linear floating-point range', () => {
  const result = evaluate(2048, 2048, 1), tail = result.distribution[2048];
  assert.equal(tail.microcanonical.sign, 0);
  assert.equal(tail.canonical.value, null); assert.equal(tail.canonical.sign, 1);
  assert.ok(tail.canonical.logMagnitude! < -17000);
  assert.equal(result.canonicalToMicroKlNats.status, 'infinite');
  assert.ok(result.microToCanonicalKlNats.value! > 0);
});
