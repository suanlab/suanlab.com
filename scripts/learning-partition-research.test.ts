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
    "id": "finite-spectrum-partition-cumulant-generators",
    "image": "finite-reweighting",
    "columns": 6,
    "original": {
      "en": "e009fc56a05155ca0af0942b3c31268f51ef0064bcbd6b3e1910aa19ab336770",
      "ko": "4bb4d08e7209d1e79a3d182a534fffdf3ca91f86af1794b6ab11661640790308"
    }
  },
  {
    "id": "coupled-binary-sites-partition-factorization",
    "image": "hidden-spin-marginal",
    "columns": 6,
    "original": {
      "en": "db01c9360f23c77e8d4ff7bf214ebfc46b9e376fdc5dad3c21a9d83a43265b13",
      "ko": "4a2ac611c35b21482cff6e862e52de3a2cc65681ebbaf4d27c3777d256dc2fe7"
    }
  },
  {
    "id": "infinite-energy-ladder-normalization-cutoff",
    "image": "infinite-weight-variance",
    "columns": 6,
    "original": {
      "en": "ee3e1c2d3a2d8a927622c41b0d54884310e3b49a790b0860e28961a10b0bfc5e",
      "ko": "fb46a0950c5ba61d2aa4a304ed01b56ab05220e94791562d4c9c8fca8fb764b0"
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
      assert.equal((original.match(/^### [1-6]\./gm) || []).length, 6);
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
      assert.equal(images[0].url, `/assets/learning/partition-research-${spec.image}.${lang}.svg`);
      pairs.push({ math, codes, stdout });
    }
    assert.deepEqual(pairs[0], pairs[1]);
    const reader = await getLearningContent(spec.id);
    for (const lang of ['en', 'ko'] as const) {
      assert.ok(!reader[lang].includes('katex-error'));
      assert.ok(reader[lang].includes(`partition-research-${spec.image}.${lang}.svg`));
      assert.equal(reader.outline[lang].filter(h => /Research extension:|연구 확장:/.test(h.title)).length, 1);
    }
  });
}

test('independent batch laws, hidden states and geometric tail conditions', () => {
 const report=JSON.parse(execFileSync('python3',['scripts/learning/verify-partition-research.py'],{encoding:'utf8',timeout:30000}));
 assert.deepEqual(report,{status:'passed',finiteReweightingPreparations:81,exactIndependentBatchLaws:16,hiddenSpinPreparations:15,exactGeometricTailPairs:9,decimalPublishedTailCases:4});
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
