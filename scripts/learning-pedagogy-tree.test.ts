import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { build } from 'esbuild';
import { renderToString } from 'katex';
import { learningPaths, pathUnits, learningLocation, searchLearningPath } from '../src/data/learning-paths';
import { learningUnits, learningGuides } from '../src/data/interactive-book';
import { curriculumDomains } from '../src/data/learning-curriculum';
import { firstStepLessons } from '../src/data/learning-first-steps';
import { firstStepIds, firstStepEquation, firstStepDisplayEquation, fractionTex, projectionPreparation, descentPreparation, bayesPreparation, interferencePreparation } from '../src/lib/learning-first-steps';

test('seven paths cover all 51 domains and all 650 lessons exactly once, preserving the registry', () => {
  const domains = learningPaths.flatMap(p => p.domainIds), ids = learningPaths.flatMap(pathUnits);
  assert.equal(learningPaths.length, 7); assert.equal(domains.length, 51); assert.equal(new Set(domains).size, 51);
  assert.deepEqual(new Set(domains), new Set(curriculumDomains.map(d => d.id)));
  assert.equal(ids.length, 650); assert.equal(new Set(ids).size, 650);
  assert.deepEqual(new Set(ids), new Set(learningUnits.map(u => u.id)));
  assert.equal(Object.values(learningGuides).reduce((n, g) => n + g.experiments.length, 0), 1969);
  for (const u of learningUnits) { const location = learningLocation(u.id)!; assert(location); assert(location.domain.unitIds.includes(u.id)); assert(location.path.domainIds.includes(location.domain.id)); }
  assert.equal(learningLocation('research-methods')!.domain.id, 'research-methods');
});
test('search finds lesson titles, translations, subtitles and topics with matching ancestors', () => {
  for (const query of ['projection', '직교투영', 'ＰＣＡ', 'Bayes', 'interference']) {
    const result = searchLearningPath(query); assert(result.unitCount > 0, query);
    const ids = result.paths.flatMap(p => p.domains.flatMap(d => d.unitIds)); assert.equal(ids.length, new Set(ids).size);
    for (const p of result.paths) for (const d of p.domains) assert(p.path.domainIds.includes(d.domain.id));
  }
  assert.equal(searchLearningPath('no-such-topic-823591').unitCount, 0);
  assert.equal(searchLearningPath('  ').unitCount, 650);
  assert(searchLearningPath('projection').paths.some(p => p.path.id === 'mathematics'));
});
test('exact fractions never silently round non-dyadic or tiny values to a different number', () => {
  assert.equal(fractionTex(17 / 8), '\\frac{17}{8}'); assert.equal(fractionTex(-.5), '-\\frac{1}{2}');
  for (const v of [1 / 3, 1e-12, Math.PI, NaN, Infinity]) assert.throws(() => fractionTex(v), RangeError);
});
test('three projections satisfy closest-point and perpendicular identities', () => {
  const expected = [[3, 0], [2.5, 2.5], [0, 2]];
  for (let e = 0; e < 3; e++) {
    const r = projectionPreparation(e); assert.deepEqual(r.projected, expected[e]);
    assert.equal(r.v.reduce((s, x, i) => s + x * r.residual[i], 0), 0);
    for (const offset of [-2, -.1, .1, 2]) { const distance = r.u.reduce((s, x, i) => s + (x - (r.coefficient + offset) * r.v[i]) ** 2, 0); assert(distance > r.residualSquared); }
  }
  assert.throws(() => projectionPreparation(3));
});
test('descent examples agree with independently iterated gradients and exact reference losses', () => {
  for (let e = 0; e < 3; e++) {
    let x = 2, y = 2; const rate = [1 / 4, 3 / 4, 5 / 4][e];
    for (let k = 0; k < 4; k++) { const r = descentPreparation(e, k); assert.deepEqual([r.current.x, r.current.y, r.current.loss], [x, y, x * x + y * y / 2]); x -= rate * 2 * x; y -= rate * y; }
  }
  assert.equal(descentPreparation(0, 1).current.loss, 17 / 8);
  assert.equal(descentPreparation(2, 1).current.loss, 73 / 8);
  assert.throws(() => descentPreparation(0, 4));
});
test('Bayes weights agree with direct independent likelihoods and react to contrary evidence', () => {
  for (let e = 0; e < 2; e++) for (let k = 0; k < 4; k++) {
    const r = bayesPreparation(e, k); const fair = .5 ** k, biased = r.seen.reduce((p, x) => p * (x ? .75 : .25), 1);
    assert(Math.abs(r.biased - biased / (fair + biased)) < 1e-15); assert.equal(r.fair + r.biased, 1);
  }
  assert.equal(bayesPreparation(0, 1).biased, 3 / 5); assert.equal(bayesPreparation(0, 3).biased, 9 / 17);
  assert(bayesPreparation(0, 3).biased < bayesPreparation(0, 2).biased); assert.equal(bayesPreparation(1, 3).biased, 1 / 9);
});
test('interference probabilities agree with dense H/Z/I products at each prefix', () => {
  const H = [[1, 1], [1, -1]].map(row => row.map(v => v / Math.sqrt(2)));
  for (const e of [0, 1]) {
    let v = [1, 0];
    for (let k = 0; k < 4; k++) {
      const r = interferencePreparation(e, k); assert.equal(r.negative, v[1] < -1e-12); assert(Math.abs(r.p0 - v[0] ** 2) < 1e-14); assert(Math.abs(r.p1 - v[1] ** 2) < 1e-14);
      if (k === 0 || k === 2) v = H.map(row => row.reduce((n, x, i) => n + x * v[i], 0));
      else if (k === 1 && e === 0) v = [v[0], -v[1]];
    }
  }
});
test('the new inventory covers both manuscripts for every lesson without declaring a redesign complete', () => {
  const inventory = JSON.parse(readFileSync('docs/online-learning/physics-math-for-ai/pedagogy-redesign-inventory.json', 'utf8'));
  assert.equal(inventory.units.length, 650); assert.equal(inventory.counts.manuscripts, 1300);
  assert.equal(inventory.counts.fullChapterRedesignsVerified, 0); assert.equal(inventory.counts.firstReadingPrototypes, 4);
  for (const unit of inventory.units) for (const lang of ['ko', 'en']) {
    const bytes = readFileSync(`content/learning/physics-math-for-ai/${unit.id}.${lang}.md`);
    assert.equal(createHash('sha256').update(bytes).digest('hex'), unit.languages[lang].sha256);
    assert.equal(unit.fullChapterPedagogyReview, 'pending');
  }
});
test('both languages have four explanatory stages, two worked examples and strict LaTeX', () => {
  for (const id of firstStepIds) {
    const lesson = firstStepLessons[id]; assert.equal(lesson.stages.length, 4); assert.equal(lesson.worked.length, 2);
    for (const lang of ['ko', 'en'] as const) {
      const strings = [lesson.question, lesson.purpose, lesson.analogy, lesson.prediction, lesson.explanation, lesson.bridge, ...lesson.stages.flatMap(s => [s.title, s.body]), ...lesson.worked.flatMap(s => [s.title, s.body])].map(s => s[lang]);
      if (lang === 'en') assert.doesNotMatch(strings.join(' '), /[가-힣]/);
      for (const str of strings) { assert.equal((str.match(/\$/g) ?? []).length % 2, 0); for (const math of str.matchAll(/\$([^$]+)\$/g)) renderToString(math[1], { strict: 'error', throwOnError: true }); }
      for (const work of lesson.worked) renderToString(work.tex, { strict: 'error', throwOnError: true });
    }
  }
});
test('every dynamic example and stage renders strict LaTeX', () => {
  for (const id of firstStepIds) for (let example = 0; example < firstStepLessons[id].examples.length; example++) for (let stage = 0; stage < 4; stage++) {
    const math = firstStepEquation(id, example, stage); assert(math); renderToString(firstStepDisplayEquation(id, example, stage), { strict: 'error', throwOnError: true }); renderToString(math, { strict: 'error', throwOnError: true });
  }
});
test('initial first-reading components and both worked examples render accessible math during SSR', async () => {
  const bundle = await build({ stdin: { contents: "import {renderToStaticMarkup} from 'react-dom/server';import {createElement} from 'react';import Studio from './src/components/learning/learning-first-steps';export const render=(kind,language)=>renderToStaticMarkup(createElement(Studio,{kind,language}));", resolveDir: process.cwd(), loader: 'tsx' }, jsx: 'automatic', bundle: true, write: false, platform: 'node', format: 'cjs', packages: 'external' });
  const mod = { exports: {} as { render: (id: string, lang: string) => string } };
  new Function('require', 'module', 'exports', bundle.outputFiles[0].text)(createRequire(`${process.cwd()}/package.json`), mod, mod.exports);
  for (const id of firstStepIds) for (const lang of ['ko', 'en']) {
    const html = mod.exports.render(id, lang); assert(html.includes('data-learning-studio')); assert(html.includes('<math')); assert(!html.includes('katex-error'));
    assert.equal((html.match(/class="ai-learning-math ai-learning-math-block"/g) ?? []).length, id === 'gradients' ? 4 : 3);
    if (lang === 'en') assert.doesNotMatch(html, /[가-힣]/);
  }
});

test('rounded gradient summaries are marked approximate while exact fractions remain available', () => {
  const display = firstStepDisplayEquation('gradients', 2, 3);
  assert(display.includes('x_{3}=-6.75')); assert(display.includes('y_{3}\\approx-0.031')); assert(display.includes('L_{3}\\approx45.563'));
  assert(firstStepEquation('gradients', 2, 3).includes('93313'));
  assert(!firstStepDisplayEquation('gradients', 0, 1).includes('\\approx'));
});
