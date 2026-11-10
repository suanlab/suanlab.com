import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { EnsembleConceptResult, EnsembleConceptView } from '../src/components/learning/ensemble-concept';
import { ensembleConceptInputs } from '../src/lib/learning-ensemble-concepts';
import { evaluateEnsembleRequest } from '../src/lib/learning-ensemble-evaluator';
import { ensemblePresets, ensembleGuides, type EnsembleKind } from '../src/lib/learning-ensembles';
import { learningUnits, learningGuides } from '../src/data/interactive-book';
(globalThis as typeof globalThis & { React?: typeof React }).React = React;
const kinds = Object.keys(ensemblePresets) as EnsembleKind[];
const close = (a: number | null, b: number) => { assert.notEqual(a, null); assert.ok(Math.abs(a! - b) < 1e-11 * Math.max(1, Math.abs(b)), `${a} versus ${b}`); };
const responses = (kind: EnsembleKind, step: number) => ensembleConceptInputs(kind, step).map(input => {
  const response = evaluateEnsembleRequest({ version: 1, requestId: 1, input }); assert.ok(response.ok); return response;
});
for (const kind of kinds) for (let step = 0; step < 4; step++) for (const language of ['ko', 'en'] as const) test(`${kind} fixed step ${step} ${language}: finite accessible independent reference`, () => {
  const html = renderToStaticMarkup(React.createElement(EnsembleConceptResult, { responses: responses(kind, step), step, language }));
  assert.doesNotMatch(html, /NaN|Infinity/); if (language === 'en') assert.doesNotMatch(html, /[가-힣]/);
  assert.match(html, /<svg[^>]*role="img"[^>]*aria-label="[^"]+"/); assert.match(html, /<caption/);
  assert.match(html, /scope="row"/); assert.match(html, /role="region" tabindex="0"/);
  for (const [, attr] of html.matchAll(/\b(?:width|height|x|y)="([^"]*)"/g)) assert.ok(Number.isFinite(Number(attr)), attr);
  const names = [...html.matchAll(/role="region" tabindex="0" aria-label="([^"]+)"/g)].map(m => m[1]);
  assert.equal(names.length, new Set(names).size); if (language === 'en') assert.ok(names.every(n => n.startsWith('Fixed concept example')));
});
test('reservoir sequence keeps n and density fixed, including the singular support case', () => {
  const rows = responses(kinds[0], 3);
  assert.equal(rows.length, 5);
  for (const response of rows) { assert.equal(response.kind, kinds[0]); if (response.kind !== 'finite-binary-reservoir-canonical-comparison') throw Error('kind');
    const r = response.result, p = r.parameters; assert.equal(p.subsystemSites, 2); assert.equal(p.totalExcitations / p.totalSites, .5);
    close(r.meanExcitations.value, 1); close(r.totalVariation.value, 1 / (2 * (p.reservoirSites + 1)));
    if (p.reservoirSites === 0) assert.equal(r.canonicalToMicroKlNats.status, 'infinite');
  }
});
test('temperature response holds the Hamiltonian fixed; size sequence obeys extensive and relative scaling', () => {
  const thermal = responses(kinds[1], 2); let gap: number | undefined;
  for (const row of thermal) { assert.equal(row.kind, kinds[1]); if (row.kind !== 'canonical-binary-energy-fluctuations') throw Error('kind');
    gap ??= row.result.parameters.energyGap; assert.equal(row.result.parameters.energyGap, gap); assert.equal(row.result.parameters.sites, 4);
    const r = row.result, p = r.parameters; close(r.heatCapacityInKB.value, (p.energyGap / (r.boltzmannConstant * p.temperature)) ** 2 * r.excitationVariance.value!);
  }
  const size = responses(kinds[1], 3); const first = size[0]; assert.ok(first.kind === 'canonical-binary-energy-fluctuations');
  for (const row of size) { assert.ok(row.kind === first.kind); close(row.result.meanExcitations.value, first.result.meanExcitations.value! * row.result.parameters.sites); close(row.result.relativeCountFluctuation!.value, first.result.relativeCountFluctuation!.value! / Math.sqrt(row.result.parameters.sites)); }
});
test('chemical scan keeps epsilon and T fixed; zero-temperature samples distinguish full, threshold and empty', () => {
  const scan = responses(kinds[2], 2); for (const row of scan) { assert.ok(row.kind === 'grand-canonical-finite-occupancy-fluctuations'); const r = row.result; assert.equal(r.parameters.temperature, 1); close(r.numberSusceptibility!.value! * r.boltzmannConstant, r.numberVariance.value!); }
  const zero = responses(kinds[2], 3); for (const row of zero) assert.ok(row.kind === 'grand-canonical-finite-occupancy-fluctuations');
  assert.ok(zero[0].kind === 'grand-canonical-finite-occupancy-fluctuations' && zero[1].kind === zero[0].kind && zero[2].kind === zero[0].kind);
  assert.equal(zero[0].result.logGrandPartitionStatus, 'infinite-zero-temperature-limit'); close(zero[0].result.grandPotential.value, -2);
  assert.equal(zero[1].result.numberSusceptibility, null); close(zero[1].result.numberVariance.value, .5); close(zero[2].result.meanParticles.value, 0);
});
test('all 512 prior units and guides retain values and iteration order', () => {
  const baseline = JSON.parse(readFileSync('scripts/fixtures/pre-ensemble-registry-hashes.json', 'utf8'));
  const hash = (v: unknown) => createHash('sha256').update(JSON.stringify(v)).digest('hex');
  assert.equal(baseline.ids.length, 512); assert.deepEqual(learningUnits.slice(0, 512).map(u => u.id), baseline.ids);
  assert.deepEqual(Object.keys(learningGuides).slice(0, 512), baseline.guideOrder);
  for (const u of learningUnits.slice(0, 512)) { assert.equal(hash(u), baseline.unitHashes[u.id], u.id); assert.equal(hash(learningGuides[u.id]), baseline.guideHashes[u.id], u.id); }
});
test('three registered units provide twelve substantive bilingual concepts, nine matching guides and six explained quizzes', () => {
  assert.equal(learningUnits.slice(0, 515).length, 515); assert.equal(Object.values(learningGuides).slice(0, 515).reduce((s, g) => s + g.experiments.length, 0), 1545);
  for (const kind of kinds) {
    const unit = learningUnits.find(u => u.id === kind); assert.ok(unit);
    const guide = learningGuides[kind]; assert.equal(guide.steps.length, 4); assert.deepEqual(guide.experiments, ensembleGuides[kind]);
    assert.equal(unit.questions.length, 2); assert.ok(unit.prerequisiteIds.every(id => learningUnits.some(u => u.id === id)));
    for (const s of guide.steps) { assert.ok(s.explanation.ko.length > 90); assert.ok(s.explanation.en.length > 100); assert.doesNotMatch(s.explanation.en, /[가-힣]/); }
    for (const q of unit.questions) { assert.equal(q.options.length, 3); assert.ok(q.answer >= 0 && q.answer < 3); assert.ok(q.explanation.ko.length > 60); assert.ok(q.explanation.en.length > 80); }
  }
});
test('fixed reference SSR never constructs a worker; invalid steps are rejected', () => {
  const html = renderToStaticMarkup(React.createElement(EnsembleConceptView, { kind: kinds[0], step: 0, language: 'en' }));
  assert.match(html, /Calculating the fixed example/); assert.doesNotMatch(html, /data-ensemble-concept-result/);
  for (const s of [-1, .5, 4, NaN]) assert.throws(() => ensembleConceptInputs(kinds[0], s), RangeError);
});

test('early concept plots focus on normalized site and count probabilities rather than response diagnostics', () => {
  for (const [kind, step, axis, count] of [[kinds[1], 0, 'p(site)', 2], [kinds[1], 1, 'P(K)', 3], [kinds[2], 0, 'P(N)', 3], [kinds[2], 1, 'P(N)', 3]] as const) {
    const html = renderToStaticMarkup(React.createElement(EnsembleConceptResult, { responses: responses(kind, step), step, language: 'en' }));
    assert.ok(html.includes(axis)); assert.equal((html.match(/<rect\b/g) ?? []).length, count);
    assert.match(html, /same fixed preparation/);
  }
});
