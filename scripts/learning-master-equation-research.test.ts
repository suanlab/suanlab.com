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
    "id": "two-state-continuous-time-generator-euler",
    "image": "path-rate-information",
    "original": {
      "ko": "812ab9479b0eb6a44d40b3d0ed50a33b08cbfc2a11de0f84f36fdb3f412a555b",
      "en": "aceb4159e4a5efd526e6951282793b2a798a3107d09606958ce49b5a821fc67c"
    },
    "columns": 6,
    "equations": 10
  },
  {
    "id": "poisson-uniformization-potential-events",
    "image": "sampled-generator-alias",
    "original": {
      "ko": "522d1b2b194619627702993e5a896cbe8183df250aba8d3df5913cecbfb4be85",
      "en": "4231fba9eb578281aa22607fc7849b87112f25ba6f04dd08c5cb3d818aec0778"
    },
    "columns": 5,
    "equations": 8
  },
  {
    "id": "state-dependent-holding-time-jump-occupation",
    "image": "renewal-time-uncertainty",
    "original": {
      "ko": "7d1d07e4af60da235a8bdeee8940ffebe114e67304d328ef5af0fd695f53ed70",
      "en": "ee0791dd250ebd0b89806c714b1f7db4a52f084b30e0d9860bf387981c41c4b7"
    },
    "columns": 6,
    "equations": 10
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
      assert.equal((original.match(/^### [1-6]\. /gm) || []).length, 6);
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
      assert.equal(math.length, spec.equations);
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
      assert.equal(images[0].url, `/assets/learning/master-equation-research-${spec.image}.${lang}.svg`);
      pairs.push({ math, codes, stdout });
    }
    assert.deepEqual(pairs[0], pairs[1]);
    const reader = await getLearningContent(spec.id);
    for (const lang of ['en', 'ko'] as const) {
      assert.ok(!reader[lang].includes('katex-error'));
      assert.ok(reader[lang].includes(`master-equation-research-${spec.image}.${lang}.svg`));
      assert.equal(reader.outline[lang].filter(h => /Research extension:|연구 확장:/.test(h.title)).length, 1);
    }
  });
}

test('independent path-score integrals, stochastic sampling aliases and regeneration variances', () => {
 const report=JSON.parse(execFileSync('python3',['scripts/learning/verify-master-equation-research.py'],{encoding:'utf8',timeout:30000}));
 assert.equal(report.status,'passed');
 assert.equal(report.pathFisherPreparations,108);
 assert.equal(report.renewalPoissonComparisons,375);
 assert.equal(report.jumpCountTiltComparisons,125);
 assert.equal(report.aliasPreparations,9);
 assert.ok(report.maxPathMomentDifference<report.maxAnalyticOmittedScoreBound+3e-12);
});

test('supplements remain in the full 17-unit domain without new lab controls', () => {
  const domain = curriculumDomains.find(d => d.id === 'nonequilibrium')!;
  assert.equal(domain.unitIds.length, 17);
  assert.equal(domain.topics.length, 10);
  assert.equal(learningUnits.length, 650);
  assert.equal(Object.values(learningGuides).reduce((n, g) => n + g.experiments.length, 0), 1969);
  for (const spec of specs) {
    assert.ok(domain.unitIds.includes(spec.id));
    assert.equal(learningGuides[spec.id].steps.length, 6);
    assert.equal(learningGuides[spec.id].experiments.length, 3);
  }
});
