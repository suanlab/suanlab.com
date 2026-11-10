import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { boltzmannGibbsKinds as kinds, boltzmannGibbsPresets as presets, boltzmannGibbsGuides as guides,
  boltzmannGibbsTitles as titles, boltzmannGibbsPresetInput as input, type BoltzmannGibbsKind } from '../src/lib/learning-boltzmann-gibbs';
import { weightedBoltzmannModel } from '../src/lib/learning-weighted-boltzmann';
import { finiteGibbsModel } from '../src/lib/learning-finite-gibbs';
import { quadraticGibbsModel } from '../src/lib/learning-quadratic-gibbs';
for (const kind of kinds) for (const [index, preset] of presets[kind].entries()) test(`${preset.id}: the prepared experiment runs with explicit scientific outcomes`, () => {
  const v = input(kind, index); assert.equal(v.kind, kind);
  const result = v.kind === kinds[0] ? weightedBoltzmannModel(v.parameters) : v.kind === kinds[1] ? finiteGibbsModel(v.parameters) : quadraticGibbsModel(v.parameters);
  assert.equal(result.fullyResolved, true);
  if ('normalizable' in result) assert.equal(result.normalizable, !['gaussian-flat', 'gaussian-indefinite'].includes(preset.id));
  assert.doesNotMatch(JSON.stringify(result), /NaN|Infinity/);
  const visit = (o: unknown): void => { if (typeof o === 'number') assert.ok(Number.isFinite(o)); else if (Array.isArray(o)) o.forEach(visit); else if (o && typeof o === 'object') Object.values(o).forEach(visit); };
  visit(result);
});
test('nine substantive bilingual guides cover all twenty-seven unique presets without registering a book reader', () => {
  const ids = kinds.flatMap(k => presets[k].map(p => p.id)); assert.equal(ids.length, 27); assert.equal(new Set(ids).size, 27);
  for (const kind of kinds) {
    assert.ok(titles[kind].ko && titles[kind].en); assert.equal(guides[kind].length, 3);
    assert.deepEqual(guides[kind].flatMap(g => [...g.presetIndices]), [0, 1, 2, 3, 4, 5, 6, 7, 8]);
    for (const g of guides[kind]) for (const text of [g.title, g.predict, g.action, g.explain]) { assert.ok(text.ko.length > 6 && text.en.length > 10); assert.doesNotMatch(text.en, /[가-힣]/); }
    presets[kind].forEach(p => { assert.ok(p.label.ko && p.label.en); assert.doesNotMatch(p.label.en, /[가-힣]/); });
  }
});
test('preset requests return owned snapshots and reject unknown selections', () => {
  for (const kind of kinds) {
    const a = input(kind, 0); a.parameters.temperature = 99;
    assert.notEqual(input(kind, 0).parameters.temperature, 99);
  }
  const a = input(kinds[0], 0); assert.equal(a.kind, kinds[0]); (a.parameters.energyLevels as number[])[0] = 99;
  const b = input(kinds[0], 0); assert.equal(b.kind, kinds[0]); assert.equal(b.parameters.energyLevels[0], 0);
  const c = input(kinds[2], 7); assert.equal(c.kind, kinds[2]); (c.parameters.coordinateScale as unknown as [number, number])[0] = 99;
  const d = input(kinds[2], 7); assert.equal(d.kind, kinds[2]); assert.equal(d.parameters.coordinateScale![0], -2);
  for (const i of [-1, .5, 9, NaN]) assert.throws(() => input(kinds[0], i), RangeError);
  assert.throws(() => input('unknown' as BoltzmannGibbsKind, 0), RangeError);
});
test('experiment configuration imports numerical models only as erased types', () => {
  const source = readFileSync('src/lib/learning-boltzmann-gibbs.ts', 'utf8');
  assert.doesNotMatch(source, /^import(?! type )/m);
  assert.doesNotMatch(source, /from ['"].*\/data\//);
});
