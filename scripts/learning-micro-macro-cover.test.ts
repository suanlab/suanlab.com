import test from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import MicroMacroCover from '../src/components/learning/micro-macro-cover';
import { microMacroPresets, type MicroMacroKind } from '../src/lib/learning-micro-macro';
for (const kind of Object.keys(microMacroPresets) as MicroMacroKind[]) test(`${kind} decorative cover has finite coordinates and uses shared palette`, () => {
  const html = renderToStaticMarkup(React.createElement(MicroMacroCover, { kind }));
  assert.match(html, /aria-hidden="true"/); assert.match(html, /class="ai-book-art"/);
  assert.doesNotMatch(html, /NaN|Infinity|[가-힣]/); assert.match(html, /var\(--learning-blue\)/);
  for (const [, v] of html.matchAll(/\b(?:width|height|x|y|cx|cy|r)="([^"]+)"/g)) assert.ok(Number.isFinite(Number(v)));
  if (kind === 'binary-shell-multiplicity-block-counts') for (const pattern of ['1100', '1010', '1001', '0110', '0101', '0011']) assert.ok(html.includes(pattern));
});
