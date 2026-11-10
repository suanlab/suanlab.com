import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkRehype from 'remark-rehype';
import rehypeKatex from 'rehype-katex';
import rehypeStringify from 'rehype-stringify';
import { finiteStateLinearResponseModel as model, isFiniteStateResponseParameters, type FiniteStateResponseParameters } from '../src/lib/learning-finite-state-linear-response';
import type { PartitionObservation } from '../src/lib/learning-partition-exact-core';

type Case = { input: FiniteStateResponseParameters; stationary: string[]; source: string[]; static: string[]; frequency: { real: string; imaginary: string }[] };
const fixtures = JSON.parse(readFileSync('scripts/fixtures/linear-response-exact-preparation.json', 'utf8')) as { cases: Case[] };
const rational = (value: PartitionObservation) => {
  assert.equal(value.resolution, value.sign === 0 ? 'structural-zero' : 'exact-rational');
  assert(value.bounds); assert.deepEqual(value.bounds.lower, value.bounds.upper);
  const { numerator: n, denominator: d } = value.bounds.lower;
  assert.equal(n.exponent, 0); assert.equal(d.exponent, 0);
  return d.coefficient === '1' ? n.coefficient : `${n.coefficient}/${d.coefficient}`;
};

for (const c of fixtures.cases) test(`exact independent reference ${JSON.stringify(c.input)}`, () => {
  const r = model(c.input);
  assert.deepEqual(r.stationaryLaw.map(rational), c.stationary);
  assert.deepEqual(r.signedSource.map(rational), c.source);
  assert.equal(rational(r.sourceMassResidual), '0'); assert.equal(rational(r.stationaryDerivativeMassResidual), '0');
  for (const [i, o] of r.observables.entries()) {
    assert.equal(rational(o.initialKernel), c.source[i]);
    assert.equal(rational(o.staticSusceptibility), c.static[i]); assert.equal(rational(o.stationaryDerivative), c.static[i]);
    assert.equal(rational(o.staticIdentityResidual), '0');
    assert.equal(rational(o.frequencySusceptibility.real), c.frequency[i].real);
    assert.equal(rational(o.frequencySusceptibility.imaginary), c.frequency[i].imaginary);
    assert.equal(o.frequencySusceptibility.phaseDefined, c.frequency[i].real !== '0' || c.frequency[i].imaginary !== '0');
  }
  assert.equal(r.classification.timeDomainImplemented, false); assert.equal(r.classification.finiteAmplitudeImplemented, false);
});

test('strict rational preparation, equivalent encodings, extreme denominators and static mode', () => {
  const p = fixtures.cases[0].input;
  for (const candidate of [null, [], { ...p, extra: 1 }, { ...p, a: 0 }, { ...p, a: 1.5 }, { ...p, rateDenominator: 0 }, { ...p, frequencyNumerator: -1 }, { ...p, kineticNumerator: Infinity }, { ...p, a: 1000001 }]) {
    assert.equal(isFiniteStateResponseParameters(candidate), false);
    assert.throws(() => model(candidate as FiniteStateResponseParameters), RangeError);
  }
  const base = model(p), equivalent = model({ ...p, a: 2, b: 2, c: 2, d: 2, e: 2, f: 2, rateDenominator: 2 });
  assert.deepEqual(equivalent.observables, base.observables);
  const slow = model({ ...p, rateDenominator: 1000000, frequencyNumerator: 1, frequencyDenominator: 1000000 });
  const fast = model({ ...p, frequencyNumerator: 1 });
  assert.deepEqual(slow.observables.map(o => o.frequencySusceptibility), fast.observables.map(o => o.frequencySusceptibility));
  assert.equal(base.classification.frequencyUsesStaticLimit, true);
  // A driven chain may still have an exactly zero source for a specific handle.
  const zero = model({ ...p, a: 3, b: 3, c: 3, d: 1, e: 1, f: 1, kineticNumerator: -1, frequencyNumerator: 2 });
  assert.equal(zero.classification.detailedBalance, false);
  assert(zero.observables.every(o => !o.frequencySusceptibility.phaseDefined && rational(o.frequencySusceptibility.magnitudeSquared) === '0'));
});

test('bilingual draft equations render, correspond, and keep the experimental scope explicit', async () => {
  const sources = ['ko', 'en'].map(lang => readFileSync(`docs/online-learning/physics-math-for-ai/linear-response-chapter.${lang}.md`, 'utf8'));
  const equations = (source: string) => [...source.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m => m[1].replace(/\s/g, '').replace(/[.,]$/, ''));
  assert.deepEqual(equations(sources[0]), equations(sources[1]));
  for (const source of sources) {
    assert.equal((source.match(/^## /gm) || []).length, 14);
    assert.equal((source.match(/^\*\*\d\./gm) || []).length, 6);
    const file = await unified().use(remarkParse).use(remarkGfm).use(remarkMath).use(remarkRehype).use(rehypeKatex, { strict: 'error' }).use(rehypeStringify).process(source);
    assert.deepEqual(file.messages, []); assert(!String(file).includes('katex-error'));
    assert(source.includes('−1/76') && source.includes('273/1444') && source.includes('120'));
    assert(source.includes('linear-response-authoring-plan.md'));
  }
});
