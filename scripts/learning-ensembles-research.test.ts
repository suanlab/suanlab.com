import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import { getLearningContent } from '../src/lib/learning-content';
import { learningUnits, learningGuides } from '../src/data/interactive-book';
import { curriculumDomains } from '../src/data/learning-curriculum';

const specs = [
  {
    "id": "finite-binary-reservoir-canonical-comparison",
    "image": "reservoir-identification",
    "columns": 6,
    "original": {
      "en": "5d0a8f746486d28aebd5b5164617b2594bdefe3ea3af3082238ba0fdf44f0407",
      "ko": "3916532391a89710021232e47249b076af621113990007c5a42547b904fb1ae9"
    }
  },
  {
    "id": "canonical-binary-energy-fluctuations",
    "image": "driven-canonical-response",
    "columns": 6,
    "original": {
      "en": "2a7473a89a0f5288a92b80057cca25f82c120b7ff421d98f130d098342e4b113",
      "ko": "a20b9e72f85794d578ace76e22118bd1eee6898a00bc2fb84a2b5326a0b9652e"
    }
  },
  {
    "id": "grand-canonical-finite-occupancy-fluctuations",
    "image": "grand-response-rank",
    "columns": 6,
    "original": {
      "en": "1ab612949a826ac3dea1558fa52f7c667648804503f57dbbd47508157d9eef11",
      "ko": "79fd2af07f305bd235b4e1d1324dde87a20b8bef4dcd1cc4333176f97af3d85b"
    }
  }
] as const;

for (const spec of specs) {
  test(`${spec.id}: original content and six solutions survive exactly`, () => {
    for (const lang of ['en', 'ko'] as const) {
      const source = readFileSync(`content/learning/physics-math-for-ai/${spec.id}.${lang}.md`, 'utf8');
      const start = source.search(/^## (Research extension:|연구 확장:)/m);
      const end = source.indexOf('\n## ', start + 4) + 1;
      assert.ok(start >= 0 && end > start);
      const original = source.slice(0, start) + source.slice(end);
      assert.equal(createHash('sha256').update(original).digest('hex'), spec.original[lang]);
      assert.equal((original.match(/^### (?:연습|Exercise) [1-6]:/gm) || []).length, 6);
    }
  });
  test(`${spec.id}: paired executable mathematics, table and actual reader`, async () => {
    const pairs = [];
    for (const lang of ['en', 'ko'] as const) {
      const source = readFileSync(`content/learning/physics-math-for-ai/${spec.id}.${lang}.md`, 'utf8');
      if (lang === 'en') assert.doesNotMatch(source, /[가-힣]/);
      assert.doesNotMatch(source, /[\u0000-\u0008\u000b-\u001f]/);
      const start = source.search(/^## (Research extension:|연구 확장:)/m);
      const end = source.indexOf('\n## ', start + 4) + 1;
      const ast = unified().use(remarkParse).use(remarkGfm).use(remarkMath).parse(source.slice(start, end));
      assert.doesNotMatch(JSON.stringify(ast), /"type":"emphasis"/);
      const math = ast.children.flatMap(n => n.type === 'math' ? [n.value] : []);
      const codes = ast.children.flatMap(n => n.type === 'code' && n.lang === 'python' ? [n.value] : []);
      assert.equal(math.length, 10);
      assert.equal(codes.length, 1);
      const tables = ast.children.filter(n => n.type === 'table');
      assert.equal(tables.length, 1);
      assert.equal(tables[0].children.length, 5);
      assert.ok(tables[0].children.every(row => row.children.length === spec.columns));
      const rows = tables[0].children.slice(1).map(row => row.children.map(cell => cell.children.map(n => 'value' in n ? n.value : '').join('')));
      const stdout = execFileSync('python3', ['-c', codes[0]], { encoding: 'utf8', timeout: 20000 });
      assert.deepEqual(stdout.trim().split('\n').map(line => line.split(/\s+/)), rows);
      assert.equal(ast.children.filter(n => n.type === 'heading' && n.depth === 4).length, 3);
      const images = ast.children.flatMap(n => n.type === 'paragraph' ? n.children.filter(c => c.type === 'image') : []);
      assert.equal(images.length, 1);
      assert.ok(images[0].alt && images[0].alt.length > 30);
      assert.equal(images[0].url, `/assets/learning/ensembles-${spec.image}.${lang}.svg`);
      pairs.push({ math, codes, stdout });
    }
    assert.deepEqual(pairs[0], pairs[1]);
    const reader = await getLearningContent(spec.id);
    for (const lang of ['en', 'ko'] as const) {
      assert.ok(!reader[lang].includes('katex-error'));
      assert.ok(reader[lang].includes(`ensembles-${spec.image}.${lang}.svg`));
      assert.equal(reader.outline[lang].filter(h => /Research extension:|연구 확장:/.test(h.title)).length, 1);
    }
  });
}

test('independent reservoir, nonlinear master-equation and Fisher calculations', () => {
 const report = JSON.parse(execFileSync('python3', ['scripts/learning/verify-ensembles-research.py'], {encoding:'utf8',timeout:30000}));
 assert.equal(report.status,'passed');
 assert.equal(report.finiteReservoirPairLaws,66);
 assert.equal(report.drivenCanonicalResponses.length,6);
 assert.equal(report.grandCovarianceAndFisherPreparations,27);
});

test('supplements remain in the full 32-unit domain without new lab controls', () => {
  const domain = curriculumDomains.find(d => d.id === 'thermo-statmech')!;
  assert.equal(domain.unitIds.length, 32);
  assert.equal(domain.topics.length, 11);
  assert.equal(learningUnits.length, 650);
  assert.equal(Object.values(learningGuides).reduce((n, g) => n + g.experiments.length, 0), 1969);
  for (const spec of specs) {
    assert.ok(domain.unitIds.includes(spec.id));
    assert.equal(learningGuides[spec.id].steps.length, 4);
    assert.equal(learningGuides[spec.id].experiments.length, 3);
  }
});
