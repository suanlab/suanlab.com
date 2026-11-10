import test from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { EnsembleResultView, ensembleChartIndices, ensembleQuantity } from '../src/components/learning/ensemble-view';
import EnsembleCover from '../src/components/learning/ensemble-cover';
import { evaluateEnsembleRequest } from '../src/lib/learning-ensemble-evaluator';
import { ensemblePresets, type EnsembleKind } from '../src/lib/learning-ensembles';
(globalThis as typeof globalThis & { React?: typeof React }).React = React;
const kinds = Object.keys(ensemblePresets) as EnsembleKind[];
for (const kind of kinds) for (const [i, preset] of ensemblePresets[kind].entries()) for (const language of ['ko', 'en'] as const) {
  test(`${kind} preset ${i} ${language}: finite geometry, accessible tables and honest range states`, () => {
    const response = evaluateEnsembleRequest({ version: 1, requestId: 1, input: preset.input });
    assert.ok(response.ok);
    const html = renderToStaticMarkup(React.createElement(EnsembleResultView, { response, language }));
    assert.doesNotMatch(html, /NaN|Infinity|="undefined"/);
    if (language === 'en') assert.doesNotMatch(html, /[가-힣]/);
    assert.match(html, /<svg[^>]*role="img"[^>]*aria-label="[^"]+"/);
    assert.match(html, /<caption/); assert.match(html, /scope="row"/); assert.match(html, /role="region" tabindex="0"/);
    for (const [, attribute] of html.matchAll(/\b(?:width|height|x|y)="([^"]*)"/g)) assert.ok(Number.isFinite(Number(attribute)), attribute);
    assert.ok((html.match(/<rect\b/g) ?? []).length <= 136);
    const lastBody = [...html.matchAll(/<tbody>([\s\S]*?)<\/tbody>/g)].at(-1)?.[1] ?? '';
    assert.ok((lastBody.match(/<tr>/g) ?? []).length <= 32);
    if (language === 'en') {
      if (kind === kinds[0] && i === 3) { assert.match(html, /Mathematical infinity/); assert.match(html, /True zero/); }
      if (kind === kinds[1] && i >= 6) assert.match(html, /Nonzero; outside linear range/);
      if (kind === kinds[1] && i === 8) assert.match(html, /Log also outside range/);
      if (kind === kinds[2] && i === 3) { assert.match(html, /ln Xi=\+∞/); assert.match(html, /Mathematical infinity/); }
      if (kind === kinds[2] && i === 4) { assert.match(html, /data-ensemble-threshold/); assert.match(html, /Undefined/); }
      if (kind === kinds[2] && i === 7) assert.match(html, /negative sign at negative temperature/);
    }
  });
}
for (const kind of kinds) test(`${kind}: decorative cover has finite SVG geometry`, () => {
  const html = renderToStaticMarkup(React.createElement(EnsembleCover, { kind }));
  assert.match(html, /aria-hidden="true"/); assert.doesNotMatch(html, /NaN|Infinity/);
  for (const [, attr] of html.matchAll(/\b(?:width|height|x|y)="([^"]*)"/g)) assert.ok(Number.isFinite(Number(attr)));
});
test('large chart sampling stays bounded, ordered and includes endpoints and selected mean neighbors', () => {
  const indices = ensembleChartIndices(4097, [1, 2047, 2049]);
  assert.ok(indices.length <= 68); assert.equal(indices[0], 0); assert.equal(indices.at(-1), 4096);
  for (const n of [1, 2047, 2049]) assert.ok(indices.includes(n));
  assert.deepEqual(indices, [...new Set(indices)].sort((a, b) => a - b));
  assert.deepEqual(ensembleChartIndices(0), []); assert.deepEqual(ensembleChartIndices(1), [0]);
});
test('formatter distinguishes zero, positive out-of-range, undefined and mathematical infinity', () => {
  const en = (_ko: string, value: string) => value;
  assert.equal(ensembleQuantity(null, en), 'Undefined');
  assert.equal(ensembleQuantity({ value: 0, sign: 0, logMagnitude: null }, en), '0');
  assert.match(ensembleQuantity({ value: null, sign: 1, logMagnitude: -800 }, en), /outside linear range/);
  assert.equal(ensembleQuantity({ value: null, sign: 1, logMagnitude: null, status: 'infinite' }, en), '∞');
});
