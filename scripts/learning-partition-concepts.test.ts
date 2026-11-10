import test from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { PartitionConceptResult, PartitionConceptView } from '../src/components/learning/partition-concept';
import PartitionCover from '../src/components/learning/partition-cover';
import { partitionConceptInputs } from '../src/lib/learning-partition-concepts';
import { evaluatePartitionRequest as evaluate } from '../src/lib/learning-partition-evaluator';
import { partitionKinds, partitionPresetInput } from '../src/lib/learning-partition';
import { partitionInputKey, type PartitionSuccess } from '../src/lib/learning-partition-protocol';
(globalThis as typeof globalThis & { React?: typeof React }).React = React;
function responses(kind: typeof partitionKinds[number], step: number): PartitionSuccess[] {
  return partitionConceptInputs(kind, step).map(input => { const response = evaluate({ version: 1, requestId: 1, input }); assert.ok(response.ok); return response; });
}
for (const kind of partitionKinds) for (let step = 0; step < 4; step++) test(`${kind} fixed concept ${step}: independent bilingual finite observations`, () => {
  const samples = responses(kind, step);
  for (const language of ['ko', 'en'] as const) {
    const html = renderToStaticMarkup(React.createElement(PartitionConceptResult, { responses: samples, step, language }));
    assert.doesNotMatch(html, /NaN|Infinity/); if (language === 'en') assert.doesNotMatch(html, /[가-힣]/);
    assert.match(html, /role="img" aria-label="[^"]+"/); assert.match(html, /scope="row"/); assert.match(html, /role="region" tabindex="0"/);
    for (const [, v] of html.matchAll(/\b(?:width|height|x|y)="([^"]*)"/g)) assert.ok(Number.isFinite(Number(v)), v);
    const labels = [...html.matchAll(/role="region" tabindex="0" aria-label="([^"]+)"/g)].map(m => m[1]);
    assert.equal(labels.length, new Set(labels).size);
  }
});
for (const kind of partitionKinds) test(`${kind}: cover is finite decorative SVG`, () => {
  const html = renderToStaticMarkup(React.createElement(PartitionCover, { kind })); assert.match(html, /aria-hidden="true"/); assert.doesNotMatch(html, /NaN|Infinity/);
  for (const [, v] of html.matchAll(/\b(?:width|height|x|y)="([^"]*)"/g)) assert.ok(Number.isFinite(Number(v)));
});
test('fixed spectrum comparisons preserve gaps under an offset, distinguish rounded input, and allow signed cumulants', () => {
  const examples = responses(partitionKinds[0], 2); assert.equal(examples.length, 3);
  assert.ok(examples.every(r => r.kind === 'finite-spectrum-partition-cumulant-generators'));
  if (examples[0].kind !== 'finite-spectrum-partition-cumulant-generators' || examples[1].kind !== examples[0].kind || examples[2].kind !== examples[0].kind) throw Error('kind');
  assert.deepEqual(examples[0].result.states.map(s => s.probability.value), examples[1].result.states.map(s => s.probability.value));
  assert.ok(examples[2].result.states.every(s => s.probability.value === 1 / 3));
  const signed = responses(partitionKinds[0], 1); assert.ok(signed[1].kind === 'finite-spectrum-partition-cumulant-generators'); assert.equal(signed[1].result.thirdEnergyCumulant.sign, -1);
});
test('support and crossing examples distinguish dependence from source response', () => {
  const support = responses(partitionKinds[1], 1); assert.ok(support[0].kind === 'coupled-binary-sites-partition-factorization' && support[1].kind === support[0].kind);
  assert.equal(support[0].result.spinCovariance.value, 0); assert.equal(support[1].result.spinCovariance.value, -1);
  const crossing = responses(partitionKinds[1], 3); assert.ok(crossing[0].kind === 'coupled-binary-sites-partition-factorization');
  assert.equal(crossing[0].result.sourceHessian[0][1].value, 0); assert.equal(crossing[0].result.responses.alignmentToField.status, 'undefined-at-zero-temperature');
});
test('cutoff comparison holds spacing and temperature fixed, and boundary examples do not invent infinite laws', () => {
  const rows = responses(partitionKinds[2], 2); assert.ok(rows[0].kind === 'infinite-energy-ladder-normalization-cutoff' && rows[1].kind === rows[0].kind);
  assert.equal(rows[0].result.parameters.gapEnergy, rows[1].result.parameters.gapEnergy); assert.equal(rows[0].result.parameters.temperature, rows[1].result.parameters.temperature);
  assert.ok(rows[0].result.comparison!.missingMass.value! > rows[1].result.comparison!.missingMass.value!);
  const boundary = responses(partitionKinds[2], 3); assert.ok(boundary.every(r => r.kind === 'infinite-energy-ladder-normalization-cutoff'));
  for (const row of boundary.slice(0, 2)) if (row.kind === 'infinite-energy-ladder-normalization-cutoff') assert.equal(row.result.infinite.quantities, null);
});
test('fixed inputs are frozen, reject invalid steps, and never use mutable saved lab settings', () => {
  for (const kind of partitionKinds) {
    const key = partitionConceptInputs(kind, 2).map(partitionInputKey); const independent = structuredClone(partitionPresetInput(kind, 0));
    (independent.parameters as { temperature: number }).temperature = 99;
    assert.deepEqual(partitionConceptInputs(kind, 2).map(partitionInputKey), key);
    assert.ok(Object.isFrozen(partitionConceptInputs(kind, 2)));
    for (const step of [-1, 4, .5, NaN]) assert.throws(() => partitionConceptInputs(kind, step), RangeError);
  }
});
test('fixed concept SSR leaves computation to workers after hydration', () => {
  const html = renderToStaticMarkup(React.createElement(PartitionConceptView, { kind: partitionKinds[0], step: 0, language: 'en' }));
  assert.match(html, /Calculating the fixed example/); assert.doesNotMatch(html, /data-partition-concept-result/);
});
