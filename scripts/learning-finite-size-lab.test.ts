import test from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import { renderToStaticMarkup as render } from 'react-dom/server';
import { build } from 'esbuild';
import { finiteSizeKinds as kinds, finiteSizePresets as presets, finiteSizePresetInput as preset } from '../src/lib/learning-finite-size';
import { finiteSizeFields as fields, finiteSizeChoices as choices, editFiniteSizeDraft as edit, editFiniteSizeChoice as choose, advanceFiniteSize as advance, rewindFiniteSize as rewind, finiteSizeSize as size, finiteSizeMaximum as maximum } from '../src/lib/learning-finite-size-controls';
import { isFiniteSizeInput, finiteSizeInputKey as key, type FiniteSizeInput } from '../src/lib/learning-finite-size-protocol';
import Lab from '../src/components/learning/finite-size-lab';
import { LanguageProvider } from '../src/components/language-provider';
import Cover from '../src/components/learning/finite-size-cover';
for (const kind of kinds) test(`${kind}: every preset exposes valid fields and choices without mutating its source`, () => {
 for (let i = 0; i < presets[kind].length; i++) {
  const input = preset(kind, i), before = key(input); assert.ok(fields(input).length); assert.ok(choices(input).length);
  for (const descriptor of fields(input)) { assert.ok(descriptor.value >= descriptor.minimum && descriptor.value <= descriptor.maximum); assert.ok(descriptor.label.ko.length); assert.doesNotMatch(descriptor.label.en, /[가-힣]/); const next = edit(input, descriptor.id, String(descriptor.value)); if (!descriptor.active) assert.equal(next, null); else { assert.ok(next); assert.ok(isFiniteSizeInput(next)); assert.equal(key(next), before); assert.notEqual(next.parameters, input.parameters); } }
  for (const descriptor of choices(input)) { assert.ok(descriptor.options.some(option => option.value === descriptor.value)); assert.doesNotMatch(descriptor.label.en, /[가-힣]/); const next = choose(input, descriptor.id, String(descriptor.value)); if (kind === kinds[2] && 'sizePath' in input.parameters && input.parameters.sizePath === 'fixed' && descriptor.id === 'approachSide') assert.equal(next, null); else { assert.ok(next); assert.equal(key(next), before); } }
  assert.equal(key(input), before);
 }
});
test('cross-field constraints reject invalid geometry, zero weights and denominator clipping', () => {
 const b = preset(kinds[0], 0); assert.equal(choose(b, 'dimension', '1'), null); const widthOne = edit(b, 'width', '1')!; const one = choose(widthOne, 'dimension', '1')!; assert.ok(one); assert.equal(choose(one, 'growthMode', 'fixed-thickness'), null); assert.equal(edit(one, 'width', '2'), null);
 const s = preset(kinds[1], 0); assert.equal(edit(s, 'commonDenominator', '0'), null); const full = preset(kinds[1], 2); assert.equal(edit(full, 'commonDenominator', '0'), null); const upper = edit(s, 'commonNumerator', '2')!; assert.equal(edit(upper, 'commonDenominator', '1'), null); assert.equal(edit(preset(kinds[1], 11), 'emptyWeight', '0'), null); assert.equal(edit(s, 'epsilonNumerator', '0'), null); assert.ok(edit(s, 'epsilonNumerator', '2'));
 const t = preset(kinds[2], 4), atOne = edit(t, 'offsetNumerator', '2')!; assert.equal(edit(atOne, 'offsetDenominator', '1'), null); assert.equal(edit(t, 'activityNumerator', '3'), null); const fixed = choose(t, 'sizePath', 'fixed')!; assert.ok(edit(fixed, 'activityNumerator', '3')); assert.equal(edit(fixed, 'offsetNumerator', '0'), null); assert.equal(choose(fixed, 'approachSide', '-1'), null);
 for (const input of [b, s, t]) { assert.equal(edit(input, 'unknown', '1'), null); assert.equal(choose(input, 'unknown', '1'), null); for (const invalid of ['1e-999', '1.0', '-1', 'Infinity', '', '1000001']) assert.equal(edit(input, fields(input).find(field => field.active)!.id, invalid), null); }
});
test('spatial scanning changes only the size control, doubles safely and stops at its declared bound', () => {
 for (const kind of kinds) { let current = preset(kind, 0); const original = { ...current.parameters } as Record<string, unknown>, field = kind === kinds[0] ? 'scale' : 'siteCount'; delete original[field]; let steps = 0;
  while (advance(current)) { const next = advance(current)!; assert.equal(size(next), Math.min(maximum(current), 2 * size(current))); const rest = { ...next.parameters } as Record<string, unknown>; delete rest[field]; assert.deepEqual(rest, original); assert.ok(isFiniteSizeInput(next)); current = next; assert.ok(++steps < 24); }
  assert.equal(size(current), maximum(current)); assert.equal(advance(current), null); const first = rewind(current); assert.equal(size(first), 1); const rest = { ...first.parameters } as Record<string, unknown>; delete rest[field]; assert.deepEqual(rest, original);
 }
 const odd = edit(preset(kinds[1], 0), 'siteCount', '511')!; assert.equal(size(advance(odd)!), 512); assert.equal(advance(null as unknown as FiniteSizeInput), null); assert.throws(() => rewind(null as unknown as FiniteSizeInput), RangeError);
});
test('decorative covers retain fixed, labeled mathematical examples independently of saved state', () => {
 const html = kinds.map(kind => render(React.createElement(Cover, { kind })));
 for (const [i, kind] of kinds.entries()) { assert.ok(html[i].includes(`data-finite-size-cover="${kind}"`)); assert.match(html[i], /aria-hidden="true"/); assert.doesNotMatch(html[i], /[가-힣]|NaN|Infinity/); }
 assert.equal((html[0].match(/<circle /g) ?? []).length, 12); assert.match(html[0], /Hnn = −11/); assert.match(html[0], /HKac = 1\/2/); assert.equal((html[1].match(/<rect /g) ?? []).length, 5); assert.match(html[1], /5\/32/); assert.equal((html[2].match(/<circle /g) ?? []).length, 6); assert.match(html[2], /ln2\/4/);
});
test('saved labs SSR read neither browser storage nor Worker and expose no ready result', () => {
 const descriptors = new Map(['Worker', 'localStorage'].map(name => [name, Object.getOwnPropertyDescriptor(globalThis, name)])); let reads = 0;
 for (const name of descriptors.keys()) Object.defineProperty(globalThis, name, { configurable: true, get() { reads++; throw Error(`SSR ${name}`); } });
 try { for (const kind of kinds) { const html = render(React.createElement(LanguageProvider, null, React.createElement(Lab, { kind }))); assert.match(html, /data-finite-size-ready="false"/); assert.doesNotMatch(html, /data-finite-size-result-key=/); assert.match(html, /data-finite-size-state="loading"/); } assert.equal(reads, 0); }
 finally { for (const [name, descriptor] of descriptors) { if (descriptor) Object.defineProperty(globalThis, name, descriptor); else Reflect.deleteProperty(globalThis, name); } }
});
test('lab and cover browser bundles exclude all exact numerical models and evaluators', async () => {
 const bundle = await build({ entryPoints: ['src/components/learning/finite-size-lab.tsx', 'src/components/learning/finite-size-cover.tsx'], outdir: '/tmp/finite-size-lab-bundle', bundle: true, write: false, platform: 'browser', format: 'esm', metafile: true, external: ['react', 'next/navigation'], logLevel: 'silent' });
 for (const path of Object.keys(bundle.metafile!.inputs)) assert.doesNotMatch(path, /learning-(boundary-extensivity|spatial-self-averaging|two-phase-rounding|finite-size-evaluator|finite-markov-exact|binary-products|rational-gcd|interval-transcendentals)\./);
});
