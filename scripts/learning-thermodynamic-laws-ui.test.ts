import test from 'node:test';
import assert from 'node:assert/strict';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { ThermodynamicLawView } from '../src/components/learning/thermodynamic-laws-lab';
import { thermodynamicLawLabs, learningUnits, type BookLocale } from '../src/data/interactive-book';
import { thermodynamicLawKinds, thermodynamicLawDefaults, validCarnotValues, validReservoirValues, validSpectrumValues,
  evaluateCarnotLaw, evaluateReservoirLaw, evaluateSpectrumLaw, type ThermodynamicLawKind } from '../src/lib/learning-thermodynamic-laws';
import { waveNumber } from '../src/components/learning/waves-plot';

// The CLI uses classic JSX for shared components; Next uses its own transform.
(globalThis as typeof globalThis & { React?: typeof React }).React = React;
const render = (kind: ThermodynamicLawKind, values: number[], language: BookLocale) =>
  renderToStaticMarkup(React.createElement(ThermodynamicLawView, { kind, values, language }));
function row(html: string, label: string) {
  const rows = [...html.matchAll(/<tr>(.*?)<\/tr>/g)].map(match => [...match[1].matchAll(/<(?:td|th)\b[^>]*>(.*?)<\/(?:td|th)>/g)].map(cell => cell[1]));
  const found = rows.find(cells => cells[0] === label);
  assert.ok(found, `Missing row ${label}`); return found;
}
test('specifications contain 16 connected controls and 40 validated complete presets with registered units', () => {
  let controls = 0, presets = 0;
  thermodynamicLawKinds.forEach((kind, i) => {
    const spec = thermodynamicLawLabs[kind], valid = [validCarnotValues, validReservoirValues, validSpectrumValues][i];
    assert.deepEqual(spec.controls.map(c => c.initial), thermodynamicLawDefaults[kind]);
    controls += spec.controls.length; presets += spec.presets.length;
    for (const preset of spec.presets) {
      assert.ok(valid(preset.values)); assert.equal(preset.values.length, spec.controls.length);
      assert.doesNotMatch(preset.name.en, /[가-힣]/);
    }
    assert.ok(learningUnits.some(unit => (unit.id as string) === kind));
  });
  assert.equal(controls, 16); assert.equal(presets, 40); assert.ok(learningUnits.length >= 512);
});
for (const kind of thermodynamicLawKinds) {
  for (const [index, preset] of thermodynamicLawLabs[kind].presets.entries()) {
    for (const language of ['ko', 'en'] as const) {
      test(`${kind} preset ${index + 1} ${language}: finite SVG, accessible figures and correct language`, () => {
        const html = render(kind, preset.values, language);
        assert.doesNotMatch(html, /NaN|Infinity|="undefined"/);
        if (language === 'en') assert.doesNotMatch(html, /[가-힣]/);
        const invalidOrder = kind === thermodynamicLawKinds[0] ? preset.values[1] < preset.values[2]
          : kind === thermodynamicLawKinds[1] ? preset.values[0] < preset.values[1] : false;
        if (invalidOrder) {
          assert.match(html, /role="status"/); assert.doesNotMatch(html, /<svg/); return;
        }
        assert.match(html, /role="region"/); assert.match(html, /scope="col"/); assert.match(html, /scope="row"/);
        const svgs = [...html.matchAll(/<svg\b([^>]*)>/g)];
        assert.equal(svgs.length, kind === thermodynamicLawKinds[0] ? 4 : kind === thermodynamicLawKinds[1] ? 3 : 5);
        for (const [, attrs] of svgs) { assert.match(attrs, /role="img"/); assert.match(attrs, /aria-label="[^"]+"/); }
        for (const [, coordinate] of html.matchAll(/data-wave-[xy]-(?:min|max)="([^"]+)"/g)) assert.ok(Number.isFinite(Number(coordinate)));
      });
    }
  }
}
test('displayed current Carnot budgets and all four legs come directly from calculated observations', () => {
  const kind = thermodynamicLawKinds[0], values = [1, 600, 300, .01, 2, 1, 5e-324], result = evaluateCarnotLaw(values);
  const html = render(kind, values, 'en'); assert.ok(result.query && result.cycle);
  for (const [label, q] of [['W', result.query.workByGas], ['ΔU', result.query.energyChange], ['ΔS', result.query.entropyChangeGas],
    ['Complete-cycle work', result.cycle.complete.workByGas]] as const) {
    const cells = row(html, label); assert.equal(cells[1], waveNumber(q.value)); assert.equal(cells[3], String(q.sign)); assert.equal(cells[5], waveNumber(q.logMagnitude));
  }
  assert.equal(row(html, 'W')[3], '1'); assert.equal(row(html, 'Complete-cycle work')[3], '-1');
  for (let i = 1; i <= 4; i++) for (const key of ['Q', 'W', 'ΔU', 'ΔS']) assert.ok(row(html, `Leg ${i} · ${key}`));
});
test('counterfactual tiny entropy is nonzero/out-of-range with its negative sign, not physical zero', () => {
  const values = [600, 300, 5e-324, -5e-324], result = evaluateReservoirLaw(values);
  const html = render(thermodynamicLawKinds[1], values, 'en'), cells = row(html, 'Sgen');
  assert.equal(cells[1], '—'); assert.equal(cells[3], '-1'); assert.equal(cells[4], 'Nonzero, out of range');
  assert.equal(cells[5], waveNumber(result.query!.entropyGeneration.logMagnitude));
  assert.match(html, /Counterfactual second-law violation, not realizable/);
});
test('actual engine efficiency stays undefined for dissipation even when the formal work fraction is defined', () => {
  const html = render(thermodynamicLawKinds[1], [600, 300, 1000, 2], 'en');
  assert.equal(row(html, 'Engine efficiency')[4], 'Undefined'); assert.equal(row(html, 'Formal W/Qₕ')[1], '-0.5');
  assert.match(html, /not classified as a refrigerator/);
});
test('spectrum retains both log statuses and separately labelled equilibrium limit and exact integer counts', () => {
  const values = [64, 16, 16, 1000, 5e-324], result = evaluateSpectrumLaw(values), html = render(thermodynamicLawKinds[2], values, 'en');
  const cells = row(html, 'p₁'); assert.equal(cells[4], 'Nonzero, out of range'); assert.equal(cells[6], 'Out of range');
  assert.equal(cells[7], waveNumber(result.query.upperGroupProbability.logNegativeLogMagnitude)); assert.equal(cells[8], 'Finite');
  assert.ok(html.includes(result.query.totalStateCount)); assert.ok(html.includes(result.query.groundStateCount));
  assert.ok(row(html, 'T=0 limit S at selected gap')); assert.equal(row(html, 'dS/dT')[2], 'J/K²');
  assert.match(html, /log₁₀\(T \/ K\)/);
});
test('zero-temperature derivative and ratio are undefined, zero-gap group probability remains populated', () => {
  const html = render(thermodynamicLawKinds[2], [1, 4, 3, 0, 0], 'en');
  assert.equal(row(html, 'dS/dT')[4], 'Undefined'); assert.equal(row(html, 'Δ/(kB T)')[4], 'Undefined');
  assert.equal(row(html, 'p₁')[1], waveNumber(3 / 7)); assert.equal(row(html, 'C')[4], 'Physical zero');
  assert.match(html, /Zero gap: both groups are ground/);
});
