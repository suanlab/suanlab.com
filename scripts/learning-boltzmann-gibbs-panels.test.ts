import test from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
(globalThis as typeof globalThis & { React?: typeof React }).React = React;
import { renderToStaticMarkup } from 'react-dom/server';
import { parse } from 'parse5';
import { boltzmannGibbsKinds as kinds, boltzmannGibbsPresets as presets, boltzmannGibbsPresetInput as preset } from '../src/lib/learning-boltzmann-gibbs';
import { boltzmannConceptInputs, boltzmannConceptSteps, boltzmannConceptQuestions } from '../src/lib/learning-boltzmann-gibbs-concepts';
import { boltzmannGibbsInputKey, isBoltzmannGibbsSuccessForInput, type BoltzmannGibbsInput, type BoltzmannGibbsSuccess } from '../src/lib/learning-boltzmann-gibbs-protocol';
import { evaluateBoltzmannGibbsRequest as evaluate } from '../src/lib/learning-boltzmann-gibbs-evaluator';
import { BoltzmannGibbsResultView } from '../src/components/learning/boltzmann-gibbs-view';
import { BoltzmannConceptResult } from '../src/components/learning/boltzmann-gibbs-concept';
import BoltzmannGibbsLab from '../src/components/learning/boltzmann-gibbs-lab';
import BoltzmannGibbsCover from '../src/components/learning/boltzmann-gibbs-cover';
import { LanguageProvider } from '../src/components/language-provider';
import { gaussianRelativeContours } from '../src/lib/learning-boltzmann-gibbs-visuals';
const cache = new Map<string, BoltzmannGibbsSuccess>();
function response(input: BoltzmannGibbsInput) {
  const key = boltzmannGibbsInputKey(input), saved = cache.get(key); if (saved) return saved;
  const r = evaluate({ version: 1, requestId: 1, input }); assert.ok(r.ok && isBoltzmannGibbsSuccessForInput(r, input, 1)); cache.set(key, r); return r;
}
function markup(html: string, en: boolean) {
  assert.doesNotMatch(html, /NaN|Infinity|undefined|katex-error/); if (en) assert.doesNotMatch(html, /[가-힣]/);
  const tree = parse(html);
  const visit = (node: { tagName?: string; attrs?: { name: string; value: string }[]; childNodes?: unknown[] }) => {
    if (node.tagName === 'svg') assert.ok(node.attrs?.some(a => a.name === 'aria-label' && a.value.length > 8) || node.attrs?.some(a => a.name === 'aria-hidden' && a.value === 'true'));
    if (node.attrs) for (const a of node.attrs) if (['x', 'y', 'cx', 'cy', 'r', 'rx', 'ry', 'width', 'height'].includes(a.name) && /^-?\d/.test(a.value)) assert.ok(Number.isFinite(Number(a.value)));
    for (const child of node.childNodes ?? []) visit(child as Parameters<typeof visit>[0]);
  }; visit(tree as Parameters<typeof visit>[0]);
}
for (const kind of kinds) for (const [i, p] of presets[kind].entries()) test(`${p.id}: both locales render scientific states with finite SVG geometry`, () => {
  const r = response(preset(kind, i));
  for (const language of ['ko', 'en'] as const) {
    const html = renderToStaticMarkup(React.createElement(BoltzmannGibbsResultView, { response: r, language })); markup(html, language === 'en');
    assert.ok(html.includes('data-boltzmann-result'));
    if (p.id === 'gaussian-flat' || p.id === 'gaussian-indefinite') { assert.match(html, /data-boltzmann-normalization="absent"/); assert.doesNotMatch(html, /<polygon/); }
  }
});
for (const kind of kinds) for (let step = 0; step < 4; step++) test(`${kind} step ${step}: fixed examples and both-locale diagrams are independently computed`, () => {
  const inputs = boltzmannConceptInputs(kind, step); assert.ok(Object.isFrozen(inputs)); inputs.forEach(i => assert.ok(Object.isFrozen(i.parameters)));
  const responses = inputs.map(response), text = boltzmannConceptSteps[kind][step];
  assert.ok(text.explanation.ko.length > 100 && text.explanation.en.length > 150); assert.ok(text.formula.length > 15); assert.doesNotMatch(text.explanation.en, /[가-힣]/);
  for (const language of ['ko', 'en'] as const) markup(renderToStaticMarkup(React.createElement(BoltzmannConceptResult, { kind, step, responses, language })), language === 'en');
});
test('the weighted fixed comparison marginalizes four microstates and separates label/reference entropy', () => {
  const [a, b] = boltzmannConceptInputs(kinds[0], 0).map(response); assert.ok(a.kind === kinds[0] && b.kind === kinds[0]);
  assert.equal(a.result.states.length, 3); assert.equal(b.result.states.length, 4);
  assert.ok(Math.abs(a.result.states[1].probability.value! - b.result.states[1].probability.value! - b.result.states[2].probability.value!) < 1e-14);
  assert.ok(Math.abs(a.result.relativeEntropyInKB.value! - b.result.labelEntropyNats.value!) < 1e-14);
});
test('Gaussian low-temperature concepts keep continuous and discrete entropies separate', () => {
  const r = boltzmannConceptInputs(kinds[2], 3).map(response);
  assert.equal(r.length, 4); assert.ok(r[0].kind === kinds[2] && r[0].result.normalizable); assert.ok(r[1].kind === kinds[2] && r[1].result.normalizable);
  assert.ok(r[2].kind === kinds[2] && r[2].result.normalizable); assert.equal(r[2].result.heatCapacityInKB.value, 1);
  assert.ok(r[3].kind === kinds[0]); assert.equal(r[3].result.heatCapacityInKB.value, 0); assert.equal(r[3].result.labelEntropyNats.value, 0);
});
test('six bilingual quizzes explain the actual model distinctions with three choices', () => {
  for (const kind of kinds) {
    assert.equal(boltzmannConceptQuestions[kind].length, 2);
    for (const q of boltzmannConceptQuestions[kind]) { assert.equal(q.options.length, 3); assert.ok(q.answer >= 0 && q.answer < 3); assert.ok(q.explanation.ko.length > 80 && q.explanation.en.length > 140); assert.doesNotMatch(q.explanation.en, /[가-힣]/); }
  }
  assert.deepEqual(kinds.flatMap(k => boltzmannConceptQuestions[k].map(q => q.answer)), [0, 1, 2, 0, 1, 2]);
});
test('relative contours preserve covariance orientation under a sign-changing coordinate map', () => {
  const r = response(preset(kinds[2], 7)); assert.ok(r.kind === kinds[2] && r.result.normalizable);
  for (const frame of ['x', 'z'] as const) {
    const plot = gaussianRelativeContours(r.result, frame); assert.ok(plot.status === 'ready');
    const points = plot.paths[0].points.split(' ').map(s => s.split(',').map(Number)); assert.equal(points.length, 97);
    assert.ok(points.every(([x, y]) => x >= 0 && x <= 350 && y >= 0 && y <= 280));
    const a = (points[0][0] - 175) / 32, b = (135 - points[0][1]) / 32, c = (135 - points[24][1]) / 32;
    const cov = frame === 'x' ? r.result.covariance : r.result.transformedCoordinates.covariance;
    assert.ok(Math.abs(a * b * plot.scale ** 2 - cov[0][1].value!) < 1e-5);
    assert.ok(Math.abs((b * b + c * c) * plot.scale ** 2 - cov[1][1].value!) < 1e-5);
  }
});
test('positive thin covariance and non-normalizability have different plotting states', () => {
  const positive = response(preset(kinds[2], 3)), flat = response(preset(kinds[2], 4)); assert.ok(positive.kind === kinds[2] && flat.kind === kinds[2]);
  assert.equal(gaussianRelativeContours(positive.result).status, 'too-thin'); assert.equal(gaussianRelativeContours(flat.result).status, 'not-normalizable');
});
test('fixed concept and experiment result landmarks have distinct accessible names', () => {
  const r = response(preset(kinds[1], 0));
  for (const language of ['ko', 'en'] as const) {
    const html = renderToStaticMarkup(React.createElement(React.Fragment, null,
      React.createElement(BoltzmannGibbsResultView, { response: r, language, scope: 'concept' }),
      React.createElement(BoltzmannGibbsResultView, { response: r, language })));
    const labels = [...html.matchAll(/role="region"[^>]*aria-label="([^"]+)"/g)].map(m => m[1]);
    assert.ok(labels.length >= 12); assert.equal(new Set(labels).size, labels.length);
  }
});
test('SSR panels and decorative covers never read Worker or execute calculation', () => {
  const original = Object.getOwnPropertyDescriptor(globalThis, 'Worker'); let reads = 0;
  Object.defineProperty(globalThis, 'Worker', { configurable: true, get() { reads++; throw Error('SSR worker'); } });
  try { for (const kind of kinds) { const html = renderToStaticMarkup(React.createElement(LanguageProvider, null, React.createElement(BoltzmannGibbsLab, { kind }))); assert.match(html, /Checking|저장된 설정/); markup(renderToStaticMarkup(React.createElement(BoltzmannGibbsCover, { kind })), true); } assert.equal(reads, 0); }
  finally { if (original) Object.defineProperty(globalThis, 'Worker', original); else Reflect.deleteProperty(globalThis, 'Worker'); }
});
test('parameter controls keep their names separate from values and error messages', () => {
  for (const kind of kinds) {
    const html = renderToStaticMarkup(React.createElement(LanguageProvider, null, React.createElement(BoltzmannGibbsLab, { kind })));
    const tree = parse(html), controls: { tagName: string; attrs: Record<string, string> }[] = [], ids = new Set<string>();
    function visit(node: { tagName?: string; attrs?: { name: string; value: string }[]; childNodes?: unknown[] }) {
      const attrs = Object.fromEntries((node.attrs ?? []).map(a => [a.name, a.value]));
      if (attrs.id) ids.add(attrs.id);
      if (node.tagName === 'select' || node.tagName === 'textarea' || node.tagName === 'input' && attrs.type === 'text' && attrs['aria-labelledby']) controls.push({ tagName: node.tagName, attrs });
      for (const child of node.childNodes ?? []) visit(child as Parameters<typeof visit>[0]);
    }
    visit(tree as Parameters<typeof visit>[0]); assert.ok(controls.length >= 2);
    for (const { attrs } of controls) assert.ok(attrs['aria-label'] || attrs['aria-labelledby']?.split(' ').every(id => ids.has(id)));
  }
});
