import test from 'node:test';
import assert from 'node:assert/strict';
import refs from './fixtures/canonical-ground-tail-independent.json';
import { finiteCanonicalHelmholtzGap as evaluate } from '../src/lib/learning-canonical-gap';
import { canonicalBoltzmannConstant as kB } from '../src/lib/learning-canonical-spectrum';
import type { BinaryObservation } from '../src/lib/learning-binary-products';
function observation(a: BinaryObservation | null, b: BinaryObservation) {
  assert.ok(a !== null); assert.equal(a.sign, b.sign);
  if (b.value === null) assert.equal(a.value, null);
  else { assert.notEqual(a.value, null); if (b.sign !== 0) assert.notEqual(a.value, 0);
    assert.ok(Math.abs(a.value! - b.value) <= Math.max(2 * Number.MIN_VALUE, 6e-12 * Math.abs(b.value)), `${a.value} vs ${b.value}`); }
  if (b.logMagnitude === null) assert.equal(a.logMagnitude, null);
  else { assert.notEqual(a.logMagnitude, null);
    assert.ok(Math.abs(a.logMagnitude! - b.logMagnitude) <= Math.max(2e-9, 6e-12 * Math.abs(b.logMagnitude))); }
}
for (const [i, row] of refs.cases.entries()) test(`independent1000-digit logarithmic ground tail ${i + 1}`, () => {
  const r = evaluate(row.energies, row.temperature, row.trialWeights, row.copies);
  assert.notEqual(r.resolution, 'unresolved-at-precision-ceiling');
  for (const key of ['divergence', 'perCopyGap', 'totalGap'] as const) observation(r[key], row[key]);
  assert.equal(r.bounds.lower.sign, 1);
  if (r.resolution === 'positive-log-tail-resolved') {
    observation(r.logDivergence, row.logDivergence);
    const shift = row.logReferenceBits + r.exactLogInterval.exponent;
    assert.ok((BigInt(r.exactLogInterval.lowerCoefficient) << BigInt(shift)) <= BigInt(row.logEnvelope.lower));
    assert.ok((BigInt(r.exactLogInterval.upperCoefficient) << BigInt(shift)) >= BigInt(row.logEnvelope.upper));
    assert.equal(r.tailMetadata.groundCount, row.groundCount);
  }
});
test('reconstructed tail is multiplied before observation for both large temperature and copy count', () => {
  const a = evaluate([0, kB * Number.MAX_VALUE * 1000], Number.MAX_VALUE, [1, 0]);
  assert.equal(a.resolution, 'positive-log-tail-resolved'); assert.equal(a.divergence!.value, null);
  assert.ok(a.perCopyGap!.value! > 0);
  const b = evaluate([0, kB * 300 * 1000], 300, [1, 0], Number.MAX_VALUE);
  assert.equal(b.perCopyGap!.value, null); assert.ok(b.totalGap!.value! > 0);
});
test('huge negative log D retains a further logarithmic observation', () => {
  const r = evaluate([0, Number.MAX_VALUE], Number.MIN_VALUE, [1, 0]);
  assert.equal(r.resolution, 'positive-log-tail-resolved');
  if (r.resolution !== 'positive-log-tail-resolved') return;
  assert.equal(r.logDivergence.value, null); assert.equal(r.logDivergence.sign, -1);
  assert.ok(r.logDivergence.logMagnitude! > 1000);
  assert.equal(r.divergence.value, null); assert.equal(r.divergence.sign, 1);
  assert.equal(r.exactInterval, null); assert.ok(r.exactLogInterval.lowerCoefficient.startsWith('-'));
});
test('ground degeneracy reduces the tail without erasing its positive gap', () => {
  const E = kB * 300 * 1000;
  const a = evaluate([0, E], 300, [1, 0]), b = evaluate([0, 0, E], 300, [1, 1, 0]);
  assert.equal(a.resolution, 'positive-log-tail-resolved'); assert.equal(b.resolution, 'positive-log-tail-resolved');
  assert.ok(Math.abs(b.divergence!.logMagnitude! - a.divergence!.logMagnitude! + Math.log(2)) < 1e-10);
});
test('hidden conditional imbalance uses the general gap calculation', () => {
  const r = evaluate([0, 0, 1], Number.MIN_VALUE, [1, 2, 0]);
  assert.equal(r.resolution, 'positive-interval-resolved');
  assert.ok(r.divergence!.value! > .05);
  const absentGround = evaluate([0, 0, 1], Number.MIN_VALUE, [1, 0, 0]);
  assert.equal(absentGround.resolution, 'positive-interval-resolved');
  assert.ok(Math.abs(absentGround.divergence!.value! - Math.log(2)) < 1e-12);
});
test('state permutation, ground-weight scale and energy gauge preserve logarithmic tail', () => {
  const a = evaluate([0, 0, 1], Number.MIN_VALUE, [1, 1, 0], 8);
  const b = evaluate([5, 4, 4], Number.MIN_VALUE, [0, Number.MAX_VALUE, Number.MAX_VALUE], 8);
  assert.deepEqual(a.divergence, b.divergence); assert.deepEqual(a.perCopyGap, b.perCopyGap);
  assert.deepEqual(a.totalGap, b.totalGap); assert.doesNotThrow(() => JSON.stringify(a));
});
test('boundary matrix retains positive gaps across spectra, weights and temperatures', () => {
  const spectra = [[0, Number.MIN_VALUE], [-Number.MAX_VALUE, Number.MAX_VALUE], [0, 0, Number.MIN_VALUE], [0, 0, 1, 2]];
  let cases = 0;
  for (const E of spectra) for (const T of [Number.MIN_VALUE, 1e-100, 300, Number.MAX_VALUE]) {
    const min = Math.min(...E);
    const weights = [E.map(() => 1), E.map(e => e === min ? 1 : 0), E.map((_, i) => i === 0 ? 1 : 0), E.map((_, i) => i % 2 === 0 ? Number.MAX_VALUE : Number.MIN_VALUE)];
    for (const w of weights) {
      const r = evaluate(E, T, w, Number.MAX_VALUE);
      assert.notEqual(r.resolution, 'unresolved-at-precision-ceiling');
      assert.equal(r.divergence!.sign, 1); assert.equal(r.perCopyGap!.sign, 1); assert.equal(r.totalGap!.sign, 1);
      assert.equal(r.bounds.lower.sign, 1); assert.doesNotThrow(() => JSON.stringify(r)); cases++;
    }
  }
  assert.equal(cases, 64);
});
test('group entropy chain rule and conditional KL distinguish hidden imbalance', () => {
  const E = [0, 0, kB * 300 * Math.log(2), 2 * kB * 300];
  for (const w of [[4, 0, 1, 2], [1, 3, 0, 2], [2, 2, 1, 1]]) {
    const r = evaluate(E, 300, w), W = w.reduce((a, b) => a + b, 0), p = w.map(x => x / W);
    const groups = [[0, 1], [2], [3]], H = (v: number[]) => -v.reduce((a, x) => a + (x === 0 ? 0 : x * Math.log(x)), 0);
    const P = groups.map(g => g.reduce((a, i) => a + p[i], 0));
    const Q = groups.map(g => g.reduce((a, i) => a + r.trial.canonical.states[i].probability.value!, 0));
    let conditionalH = 0, conditionalKL = 0;
    groups.forEach((g, a) => {
      if (P[a] === 0) return;
      const h = H(g.map(i => p[i] / P[a]));
      conditionalH += P[a] * h; conditionalKL += P[a] * (Math.log(g.length) - h);
    });
    const groupKL = P.reduce((a, x, i) => a + (x === 0 ? 0 : x * Math.log(x / Q[i])), 0);
    assert.ok(Math.abs(r.trial.perCopy.informationEntropy.value! - H(P) - conditionalH) < 1e-12);
    assert.ok(Math.abs(r.divergence!.value! - groupKL - conditionalKL) < 1e-12);
    if (w[1] === 0) assert.ok(conditionalKL > .3);
  }
});
