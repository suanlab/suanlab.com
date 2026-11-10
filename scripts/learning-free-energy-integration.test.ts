import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import prior from './fixtures/pre-free-energy-registry-hashes.json';
import { learningUnits, learningGuides, freeEnergyTeaching } from '../src/data/interactive-book';
import { curriculumDomains } from '../src/data/learning-curriculum';
import BookArt from '../src/components/learning/book-art';
import { freeEnergyPresets, validFreeEnergyValues, type FreeEnergyKind } from '../src/lib/learning-free-energy';
(globalThis as typeof globalThis & { React?: typeof React }).React = React;
const ids = Object.keys(freeEnergyTeaching) as FreeEnergyKind[];
const hash = (v: unknown) => createHash('sha256').update(JSON.stringify(v)).digest('hex');
test('both free-energy units append after all 507 unchanged units and guides', () => {
  assert.equal(prior.ids.length, 507); assert.ok(learningUnits.length >= 512);
  assert.deepEqual(learningUnits.slice(0, 507).map(u => u.id), prior.ids);
  assert.deepEqual(learningUnits.slice(507, 509).map(u => u.id), ids);
  for (const u of learningUnits.slice(0, 507)) {
    assert.equal(hash(u), (prior.units as Record<string, string>)[u.id], u.id);
    assert.equal(hash(learningGuides[u.id]), (prior.guides as Record<string, string>)[u.id], u.id);
  }
  assert.equal(learningUnits.slice(0, 512).reduce((n, u) => n + learningGuides[u.id].experiments.length, 0), 1536);
});
test('registered teaching, earlier prerequisites and thermo-statmech links agree', () => {
  const order = new Map(learningUnits.map((u, i) => [u.id as string, i]));
  const domain = curriculumDomains.find(d => d.id === 'thermo-statmech')!;
  for (const id of ids) {
    const unit = learningUnits.find(u => u.id === id)!;
    assert.deepEqual(unit, freeEnergyTeaching[id].unit);
    assert.deepEqual(learningGuides[id], freeEnergyTeaching[id].guide);
    assert.ok(domain.unitIds.includes(id));
    for (const p of unit.prerequisiteIds) assert.ok(order.get(p)! < order.get(id)!);
    for (const g of learningGuides[id].experiments) assert.ok(validFreeEnergyValues(id, freeEnergyPresets[id][g.preset].values));
  }
});
test('real card art dispatch uses free-energy covers', () => {
  for (const kind of ids) {
    const html = renderToStaticMarkup(React.createElement(BookArt, { kind }));
    assert.match(html, /aria-hidden="true"/); assert.match(html, /var\(--learning-green\)/);
    assert.doesNotMatch(html, /NaN|Infinity/);
  }
});
