import test from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { PartitionResultView, partitionQuantity, partitionQuantityStatus } from '../src/components/learning/partition-view';
import { evaluatePartitionRequest as evaluate } from '../src/lib/learning-partition-evaluator';
import { partitionPresets, partitionKinds, partitionGuides, partitionPresetInput } from '../src/lib/learning-partition';
(globalThis as typeof globalThis & { React?: typeof React }).React = React;
for (const kind of partitionKinds) for (const [i, preset] of partitionPresets[kind].entries()) test(`${kind} preset ${i}: bilingual finite geometry and accurate numeric statuses`, () => {
  const response = evaluate({ version: 1, requestId: 1, input: preset.input }); assert.ok(response.ok);
  for (const language of ['ko', 'en'] as const) {
    const html = renderToStaticMarkup(React.createElement(PartitionResultView, { response, language }));
    assert.doesNotMatch(html, /NaN|Infinity|="undefined"/); if (language === 'en') assert.doesNotMatch(html, /[가-힣]/);
    assert.match(html, /role="img" aria-label="[^"]+"/); assert.match(html, /<caption/); assert.match(html, /scope="row"/); assert.match(html, /role="region" tabindex="0"/);
    for (const [, attribute] of html.matchAll(/\b(?:width|height|x|y)="([^"]*)"/g)) assert.ok(Number.isFinite(Number(attribute)), attribute);
    assert.ok((html.match(/<rect\b/g) ?? []).length <= 128);
    const lastBody = [...html.matchAll(/<tbody>([\s\S]*?)<\/tbody>/g)].at(-1)?.[1] ?? '';
    assert.ok((lastBody.match(/<tr>/g) ?? []).length <= 32);
    if (language === 'en') {
      if (kind === partitionKinds[0] && i === 8) { assert.match(html, /Unresolved at precision ceiling/); assert.match(html, /positive sign established/); }
      if (kind === partitionKinds[1] && i === 2) { assert.match(html, /Forbidden/); assert.match(html, /Exact zero/); }
      if (kind === partitionKinds[1] && i === 7) assert.match(html, /Some displayed responses have no finite derivative/);
      if (kind === partitionKinds[2] && [4, 5].includes(i)) { assert.match(html, /cannot be normalized/); assert.doesNotMatch(html, /Infinite-ladder thermodynamics/); }
      if (kind === partitionKinds[2] && i === 7) assert.match(html, /Nonzero · outside display range/);
      if (kind === partitionKinds[2] && i === 8) assert.match(html, /log-tail approximation/);
    }
  }
});
test('nine guided preparations are valid snapshots with bilingual prediction/action/explanation', () => {
  for (const kind of partitionKinds) {
    assert.equal(partitionGuides[kind].length, 3);
    for (const guide of partitionGuides[kind]) {
      assert.equal(partitionPresetInput(kind, guide.preset).kind, kind);
      for (const key of ['title', 'predict', 'action', 'explain'] as const) { assert.ok(guide[key].ko.length > 5); assert.ok(guide[key].en.length > 5); assert.doesNotMatch(guide[key].en, /[가-힣]/); }
    }
  }
});
test('undefined, unresolved sign and mathematical infinity remain different labels', () => {
  const en = (_ko: string, text: string) => text;
  assert.equal(partitionQuantity(null, en), 'Undefined');
  const infinity = { value: null, sign: -1, logMagnitude: null, logResolution: 'not-applicable' as const, resolution: 'mathematical-infinity' as const, bounds: null };
  assert.equal(partitionQuantity(infinity, en), '−∞'); assert.equal(partitionQuantityStatus(infinity, en), 'Mathematical infinity');
  const unresolved = { ...infinity, sign: 1, resolution: 'unresolved-at-precision-ceiling' as const, logResolution: 'unresolved' as const };
  assert.match(partitionQuantity(unresolved, en), /unresolved/); assert.match(partitionQuantityStatus(unresolved, en), /positive sign established/);
});
