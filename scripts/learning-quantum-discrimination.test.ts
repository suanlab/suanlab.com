import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import references from './fixtures/quantum-discrimination-independent.json';
import { binaryQubitDiscrimination } from '../src/lib/learning-quantum-information-discrimination';
import { f, sub, mul, type F } from '../src/lib/learning-quantum-hilbert-core';
import type { PartitionObservation, PartitionRational } from '../src/lib/learning-partition-exact-core';
import { getLearningContent } from '../src/lib/learning-content';
const parse = (s: string): F => { const [n, d = '1'] = s.split('/'); return f(BigInt(n), BigInt(d)); };
const sign = (x: F) => x.n > BigInt(0) ? 1 : x.n < BigInt(0) ? -1 : 0;
function decode(x: PartitionRational): F {
  const shift = x.numerator.exponent - x.denominator.exponent;
  const n = BigInt(x.numerator.coefficient), d = BigInt(x.denominator.coefficient);
  return shift >= 0 ? f(n << BigInt(shift), d) : f(n, d << BigInt(-shift));
}
// Compare a+b√c against a rational bound without approximating any square root.
function compare(certificate: string[], bound: F): number {
  const [a, b, c] = certificate.map(parse), A = sub(a, bound);
  const sa = sign(A), sb = sign(b);
  if (!sb || !sign(c)) return sa;
  if (!sa || sa === sb) return sb;
  const difference = sign(sub(mul(A, A), mul(mul(b, b), c)));
  return difference === 0 ? 0 : difference > 0 ? sa : sb;
}
function encloses(observation: PartitionObservation, certificate: string[], label: string) {
  assert(observation.bounds, `${label}: missing bounds`);
  assert(compare(certificate, decode(observation.bounds.lower)) >= 0, `${label}: lower bound`);
  assert(compare(certificate, decode(observation.bounds.upper)) <= 0, `${label}: upper bound`);
}
const bloch = (values: string[]) => values.map(parse) as [F, F, F];
for (const reference of references.cases) test(`independent Hermitian reference: ${reference.name}`, () => {
  const result = binaryQubitDiscrimination(bloch(reference.bloch0), bloch(reference.bloch1), parse(reference.prior));
  assert.equal(result.strategy, reference.strategy);
  assert.equal(result.uniqueOptimalEffect, reference.uniqueOptimalEffect);
  assert.equal(result.strictlyBeatsPriorOnly, reference.strictlyBeatsPriorOnly);
  for (const key of Object.keys(reference.scalarCertificates) as (keyof typeof reference.scalarCertificates)[])
    encloses(result[key], reference.scalarCertificates[key], key);
  assert(result.optimalGuessZeroEffect);
  for (let i = 0; i < 4; i++) {
    const effect = result.optimalGuessZeroEffect[Math.floor(i / 2)][i % 2];
    encloses(effect.real, reference.effect[i][0], `effect${i}.real`);
    encloses(effect.imaginary, reference.effect[i][1], `effect${i}.imaginary`);
    encloses(result.conditionalOutcomeProbabilities[Math.floor(i / 2)][i % 2], reference.conditional[Math.floor(i / 2)][i % 2], `conditional${i}`);
  }
  encloses(result.successResidual, ['0', '0', '0'], 'objective residual');
  assert.doesNotThrow(() => JSON.stringify(result));
});
test('tiny distinguishability survives a rounded success display', () => {
  const D = BigInt(10) ** BigInt(30);
  const result = binaryQubitDiscrimination([f(BigInt(0)), f(BigInt(0)), f(BigInt(1))], [f(BigInt(0)), f(BigInt(0)), f(D - BigInt(1), D)], parse('1/2'));
  assert.equal(result.optimalSuccess.value, result.priorOnlySuccess.value);
  assert.equal(result.gainAbovePriorOnly.sign, 1);
  assert.equal(result.traceDistance.sign, 1);
  assert.equal(result.exactStateEquality, false);
  encloses(result.gainAbovePriorOnly, [`1/${BigInt(4) * D}`, '0', '0'], 'tiny exact gain');
});
test('invalid inputs are rejected rather than projected into the Bloch ball', () => {
  const valid = bloch(['0', '0', '1']);
  for (const bad of [bloch(['1', '1', '0']), [parse('0'), , parse('0')], [parse('0')], [{ n: BigInt(0), d: BigInt(0) }, parse('0'), parse('0')], [f(BigInt(10) ** BigInt(33)), parse('0'), parse('0')]])
    assert.throws(() => binaryQubitDiscrimination(bad as [F, F, F], valid, parse('1/2')), RangeError);
  for (const prior of ['-1/2', '3/2']) assert.throws(() => binaryQubitDiscrimination(valid, valid, parse(prior)), RangeError);
});
test('first bilingual chapter renders its worked examples and exercises', async () => {
  const id = 'binary-qubit-state-discrimination', content = await getLearningContent(id);
  for (const locale of ['ko', 'en'] as const) {
    assert(content.outline[locale].length >= 10);
    assert(!content[locale].includes('katex-error'));
    assert(content[locale].includes('katex'));
  }
  assert(!/[가-힣]/.test(readFileSync(`content/learning/physics-math-for-ai/${id}.en.md`, 'utf8')));
});
