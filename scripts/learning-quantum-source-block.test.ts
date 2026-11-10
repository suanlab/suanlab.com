import test from 'node:test';
import assert from 'node:assert/strict';
import { binarySourceBlockWeight } from '../src/lib/learning-quantum-source-block';
import { f, q, zero, one, add, sub, mul, type F } from '../src/lib/learning-quantum-hilbert-core';
import { getLearningContent } from '../src/lib/learning-content';
import { readFileSync } from 'node:fs';
// Independent reference enumerates every bit string; production groups by type.
for (const numerator of [0, 1, 2, 4, 5]) for (let n = 1; n <= 8; n++) {
  test(`finite source enumeration: p=${numerator}/5, copies=${n}`, () => {
    const p = f(BigInt(numerator), BigInt(5)), probabilities: F[] = [];
    for (let bits = 0; bits < 2 ** n; bits++) {
      let probability = one;
      for (let j = 0; j < n; j++) probability = mul(probability, (bits >> j) & 1 ? p : sub(one, p));
      probabilities.push(probability);
    }
    probabilities.sort((a, b) => a.n * b.d > b.n * a.d ? -1 : a.n * b.d < b.n * a.d ? 1 : 0);
    let cumulative = zero;
    for (let rank = 0; rank <= probabilities.length; rank++) {
      if (rank) cumulative = add(cumulative, probabilities[rank - 1]);
      const result = binarySourceBlockWeight(p, n, BigInt(rank));
      assert.deepEqual(result.retainedWeight, q(cumulative));
      assert.deepEqual(result.discardedWeight, q(sub(one, cumulative)));
      assert.equal(result.requiredQubits, rank ? Math.ceil(Math.log2(rank)) : null);
      assert.equal(result.groups.reduce((s, g) => s + BigInt(g.retained), BigInt(0)), BigInt(rank));
    }
  });
}
test('two-copy manuscript weights and integer memory size', () => {
  for (const [rank, numerator, qubits] of [[2, 20, 1], [3, 24, 2], [4, 25, 2]]) {
    const result = binarySourceBlockWeight(f(BigInt(1), BigInt(5)), 2, BigInt(rank));
    assert.deepEqual(result.retainedWeight, q(f(BigInt(numerator), BigInt(25))));
    assert.equal(result.requiredQubits, qubits);
  }
});
test('64-copy uniform spectrum uses exact counts beyond safe integers', () => {
  const rank = (BigInt(1) << BigInt(63)) + BigInt(1), result = binarySourceBlockWeight(f(BigInt(1), BigInt(2)), 64, rank);
  assert.deepEqual(result.retainedWeight, q(f(rank, BigInt(1) << BigInt(64))));
  assert.equal(result.requiredQubits, 64);
  assert.equal(result.groups.reduce((s, g) => s + BigInt(g.multiplicity), BigInt(0)), BigInt(1) << BigInt(64));
});
test('invalid probabilities, block sizes and ranks fail explicitly', () => {
  for (const n of [0, 65, 1.5, NaN]) assert.throws(() => binarySourceBlockWeight(f(BigInt(1), BigInt(2)), n, BigInt(1)), RangeError);
  for (const p of [f(-BigInt(1), BigInt(2)), f(BigInt(3), BigInt(2)), {n: BigInt(1), d: BigInt(0)}]) assert.throws(() => binarySourceBlockWeight(p, 2, BigInt(1)), RangeError);
  for (const rank of [-BigInt(1), BigInt(5)]) assert.throws(() => binarySourceBlockWeight(f(BigInt(1), BigInt(2)), 2, rank), RangeError);
});
test('entropy chapter renders in both languages without math errors', async () => {
  const id = 'quantum-entropy-compression-conditional-information', content = await getLearningContent(id);
  for (const locale of ['ko', 'en'] as const) {
    assert.equal(content.outline[locale].length, 10);
    assert(content[locale].includes('<table'));
    assert(!content[locale].includes('katex-error'));
  }
  assert(!/[가-힣]/.test(readFileSync(`content/learning/physics-math-for-ai/${id}.en.md`, 'utf8')));
});
