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
    "id": "boundary-growth-extensivity-additivity",
    "image": "boundary-free-energy-information",
    "original": {
      "en": "91cee8bb2d352358997a6eaf32691bf3c2f69578dcab5580e1a8aaef28a14305",
      "ko": "5717437eb357f7d78652ff00fe41d261dc3b00d0777dd2042ea026ac910c4b42"
    },
    "columns": 6,
    "equations": 8
  },
  {
    "id": "spatial-density-self-averaging-correlations",
    "image": "rare-mode-limit-moments",
    "original": {
      "en": "ac24140fd2b4e1a4091037a98d698e1b234f10b61a5ca65d9e3ae815862b93ba",
      "ko": "87169cbf63ed0c25110b8070f927ecb1068de31846a7ff2c05b470a548213dd6"
    },
    "columns": 6,
    "equations": 8
  },
  {
    "id": "two-phase-partition-zeros-rounding",
    "image": "degeneracy-shifted-zeros",
    "original": {
      "en": "c1b8cd125800b5381367f54ae276aa5b75e8de704000b52f2d21967cdaffdab2",
      "ko": "6bf2e4df0cdb337c2c207f76449e9cf50e6171cb34edf86ad06c4c927043f58c"
    },
    "columns": 6,
    "equations": 8
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
      assert.equal(images[0].url, `/assets/learning/finite-size-research-${spec.image}.${lang}.svg`);
      pairs.push({ math, codes, stdout });
    }
    assert.deepEqual(pairs[0], pairs[1]);
    const reader = await getLearningContent(spec.id);
    for (const lang of ['en', 'ko'] as const) {
      assert.ok(!reader[lang].includes('katex-error'));
      assert.ok(reader[lang].includes(`finite-size-research-${spec.image}.${lang}.svg`));
      assert.equal(reader.outline[lang].filter(h => /Research extension:|연구 확장:/.test(h.title)).length, 1);
    }
  });
}

test('independent thermal cut laws, rare-mode moments and shifted complex zeros', () => {
 const report=JSON.parse(execFileSync('python3',['scripts/learning/verify-finite-size-research.py'],{encoding:'utf8',timeout:30000}));
 assert.deepEqual(report,{status:'passed',openChainPreparations:36,fieldCutBounds:63,mixtureMomentPreparations:288,characteristicComparisons:36,rareTailEscapes:4,positiveFieldPreparations:256,complexRoots:544,responseQuadratures:4});
});

test('supplements remain in the full 32-unit domain without new lab controls', () => {
  const domain = curriculumDomains.find(d => d.id === 'thermo-statmech')!;
  assert.equal(domain.unitIds.length, 32);
  assert.equal(domain.topics.length, 11);
  assert.equal(learningUnits.length, 650);
  assert.equal(Object.values(learningGuides).reduce((n, g) => n + g.experiments.length, 0), 1969);
  for (const spec of specs) {
    assert.ok(domain.unitIds.includes(spec.id));
    assert.equal(learningGuides[spec.id].steps.length, 6);
    assert.equal(learningGuides[spec.id].experiments.length, 3);
  }
});
