import test from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { FreeEnergyView } from '../src/components/learning/free-energy-lab';
import { freeEnergyControls, freeEnergyPresets, validFreeEnergyValues, evaluateHelmholtzValues, evaluateGasFreeEnergyValues, freeEnergyPlotValue, type FreeEnergyKind } from '../src/lib/learning-free-energy';
import { waveNumber } from '../src/components/learning/waves-plot';
(globalThis as typeof globalThis & { React?: typeof React }).React = React;
const kinds = Object.keys(freeEnergyPresets) as FreeEnergyKind[];
const render = (kind: FreeEnergyKind, values: number[], language: 'ko' | 'en') => renderToStaticMarkup(React.createElement(FreeEnergyView, { kind, values, language }));
function row(html: string, label: string) {
  const cells = [...html.matchAll(/<tr>(.*?)<\/tr>/g)].map(m => [...m[1].matchAll(/<(?:th|td)\b[^>]*>(.*?)<\/(?:th|td)>/g)].map(c => c[1]));
  const result = cells.find(c => c[0] === label); assert.ok(result, label); return result;
}
for (const kind of kinds) for (const [i, preset] of freeEnergyPresets[kind].entries()) for (const language of ['ko', 'en'] as const)
  test(`${kind} preset ${i + 1} ${language}: finite accessible figures`, () => {
    assert.ok(validFreeEnergyValues(kind, preset.values));
    const html = render(kind, preset.values, language);
    assert.doesNotMatch(html, /NaN|Infinity|="undefined"/);
    if (language === 'en') assert.doesNotMatch(html, /[가-힣]/);
    assert.equal((html.match(/<svg\b/g) ?? []).length, kind === kinds[0] ? 4 : 2);
    for (const [, attrs] of html.matchAll(/<svg\b([^>]*)>/g)) { assert.match(attrs, /role="img"/); assert.match(attrs, /aria-label="[^"]+"/); }
    assert.match(html, /role="region"/); assert.match(html, /scope="row"/);
    for (const [, x] of html.matchAll(/data-wave-[xy]-(?:min|max)="([^"]+)"/g)) assert.ok(Number.isFinite(Number(x)));
  });
test('visible observations come from the current original-Hamiltonian and gas calculations', () => {
  const v = freeEnergyPresets[kinds[0]][2].values, h = render(kinds[0], v, 'en'), r = evaluateHelmholtzValues(v);
  assert.equal(row(h, 'U')[1], waveNumber(r.trial.perCopy.meanEnergy.value));
  assert.equal(row(h, 'F−Feq')[1], waveNumber(r.perCopyGap?.value));
  const g = freeEnergyPresets[kinds[1]][0].values, gas = evaluateGasFreeEnergyValues(g), gh = render(kinds[1], g, 'en');
  assert.equal(row(gh, 'Φ−G*')[1], waveNumber(gas.availability.value));
  assert.equal(row(gh, 'Wby gas')[1], waveNumber(gas.conditionalEquilibrationBudget.workByGas.value));
});
test('cold tail and rescued total remain nonzero in the reader table', () => {
  const cold = render(kinds[0], freeEnergyPresets[kinds[0]][4].values, 'en');
  assert.equal(row(cold, 'D(p∥π)')[1], '—'); assert.equal(row(cold, 'D(p∥π)')[3], '1');
  assert.equal(row(cold, 'D(p∥π)')[4], 'Nonzero; outside linear range');
  assert.notEqual(row(cold, 'D(p∥π)')[7], '—');
  const rescued = render(kinds[0], freeEnergyPresets[kinds[0]][5].values, 'en');
  assert.equal(row(rescued, 'F−Feq')[1], '—'); assert.notEqual(row(rescued, 'N(F−Feq)')[1], '—');
});
test('display and input guards distinguish unresolved zero, integer copies and normalization', () => {
  assert.equal(freeEnergyPlotValue({ value: 0, sign: 0, logMagnitude: null, zeroStatus: 'unresolved-transcendental-cancellation' } as never), null);
  assert.equal(freeEnergyPlotValue({ value: 0, sign: 0, logMagnitude: null }), 0);
  const v = [...freeEnergyPresets[kinds[0]][0].values]; v[6] = .5; assert.equal(validFreeEnergyValues(kinds[0], v), false);
  v[6] = 1; v[2] = 0; v[3] = 0; v[4] = 0; assert.equal(validFreeEnergyValues(kinds[0], v), false);
  assert.equal(validFreeEnergyValues(kinds[0], new Array(7)), false);
  assert.equal(freeEnergyControls[kinds[0]].length + freeEnergyControls[kinds[1]].length, 13);
});
test('reader distinguishes an unresolved anchored zero from exact gas pressure equilibrium', () => {
  const R = 8.31446261815324;
  const cancellation = render(kinds[1], [1, 256, R * 8192, Math.E / 32, .03125, R * 256], 'en');
  assert.equal(row(cancellation, 'F')[1], '0');
  assert.equal(row(cancellation, 'F')[4], 'Approximate cancellation; zero unresolved');
  const equilibrium = render(kinds[1], [1, 256, R * 8192, .03125, .03125, 1000], 'en');
  assert.equal(row(equilibrium, 'Φ−G*')[1], '0');
  assert.equal(row(equilibrium, 'Φ−G*')[4], 'Structural or algebraic zero');
});
