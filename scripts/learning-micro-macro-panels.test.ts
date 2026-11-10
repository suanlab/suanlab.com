import test from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { MicroMacroResultView, microMacroChartIndices } from '../src/components/learning/micro-macro-view';
import { evaluateMicroMacroRequest } from '../src/lib/learning-micro-macro-evaluator';
import { microMacroPresets, microMacroGuides, microMacroPresetInput, validMicroMacroSavedInput, binaryPatternExamples, microMacroPlotValue, type MicroMacroKind } from '../src/lib/learning-micro-macro';
(globalThis as typeof globalThis & { React?: typeof React }).React = React;
const kinds = Object.keys(microMacroPresets) as MicroMacroKind[];
for (const kind of kinds) for (const [i, preset] of microMacroPresets[kind].entries()) for (const language of ['ko', 'en'] as const) {
  test(`${kind} example ${i + 1} ${language}: accessible finite plots and distinct representation states`, () => {
    assert.ok(validMicroMacroSavedInput(kind, preset.input));
    const response = evaluateMicroMacroRequest({ version: 1, requestId: 1, input: preset.input }); assert.ok(response.ok);
    const html = renderToStaticMarkup(React.createElement(MicroMacroResultView, { response, language }));
    assert.doesNotMatch(html, /NaN|Infinity|="undefined"/);
    if (language === 'en') assert.doesNotMatch(html, /[가-힣]/);
    assert.match(html, /<svg/); if (kind === kinds[0] && language === 'en') assert.match(html, /Exact fixed-K block counts/); assert.match(html, /scope="row"/); assert.match(html, /role="region"/); assert.match(html, /tabindex="0"/);
    for (const [, attrs] of html.matchAll(/<svg\b([^>]*)>/g)) { assert.match(attrs, /role="img"/); assert.match(attrs, /aria-label="[^"]+"/); }
    for (const [, attribute] of html.matchAll(/\b(?:width|height|x|y)="([^"]*)"/g)) assert.ok(Number.isFinite(Number(attribute)), attribute);
    if (kind === kinds[0] && i === 4 && language === 'en') { assert.match(html, /Nonzero; outside linear range/); assert.match(html, /True zero/); }
    if (kind === kinds[1] && i === 3 && language === 'en') assert.match(html, /Mathematical infinity/);
    if (kind === kinds[1] && i === 4 && language === 'en') assert.match(html, /Undefined conditional/);
    if (kind === kinds[0] && i === 5) assert.ok((html.match(/<rect/g) ?? []).length <= 133);
  });
}
test('nine guides have real examples and complete bilingual predictions, actions and explanations', () => {
  let count = 0;
  for (const kind of kinds) {
    assert.equal(microMacroGuides[kind].length, 3);
    for (const guide of microMacroGuides[kind]) {
      const input = microMacroPresetInput(kind, guide.preset); assert.ok(validMicroMacroSavedInput(kind, input));
      for (const key of ['title', 'predict', 'action', 'explain'] as const) { assert.ok(guide[key].ko.length > 3); assert.ok(guide[key].en.length > 3); assert.doesNotMatch(guide[key].en, /[가-힣]/); }
      count++;
    }
  }
  assert.equal(count, 9);
});
test('sampled large plots retain endpoints and selected states without renormalizing or drawing every count', () => {
  const indices = microMacroChartIndices(4097, [17, 2048, 2049]);
  assert.ok(indices.length <= 68); assert.equal(indices[0], 0); assert.equal(indices.at(-1), 4096);
  for (const i of [17, 2048, 2049]) assert.ok(indices.includes(i));
  assert.equal(new Set(indices).size, indices.length); assert.deepEqual(indices, [...indices].sort((a, b) => a - b));
  assert.deepEqual(microMacroChartIndices(1), [0]); assert.deepEqual(microMacroChartIndices(0), []);
  assert.equal(microMacroPlotValue({ value: null, sign: 1, logMagnitude: -800 }), null);
  assert.equal(microMacroPlotValue({ value: 0, sign: 0, logMagnitude: null }), 0);
});
test('small labeled examples agree with exact shell counts and do not attempt exponential large-state rendering', () => {
  assert.deepEqual(binaryPatternExamples(4, 2), ['0011', '0101', '0110', '1001', '1010', '1100']);
  assert.deepEqual(binaryPatternExamples(0, 0), ['∅']); assert.equal(binaryPatternExamples(13, 4), null);
  assert.equal(binaryPatternExamples(4, -1), null); assert.equal(binaryPatternExamples(4, 2.5), null);
});
test('saved state rejects other families, invalid all-zero weights and malformed channel rows', () => {
  assert.equal(validMicroMacroSavedInput(kinds[0], microMacroPresets[kinds[1]][0].input), false);
  assert.equal(validMicroMacroSavedInput(kinds[1], { kind: kinds[1], parameters: { trialWeights: [0, 0], targetWeights: [1, 1], mapping: [0, 0], groupCount: 1 } }), false);
  assert.equal(validMicroMacroSavedInput(kinds[2], { kind: kinds[2], parameters: { preparationWeights: [1, 0], channelWeights: [[1, 0], [0, 0]] } }), false);
  assert.throws(() => microMacroPresetInput(kinds[0], -1), RangeError);
});
