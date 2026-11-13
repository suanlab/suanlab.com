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
import { learningUnits, learningGuides, thermodynamicLawLabs } from '../src/data/interactive-book';
import { curriculumDomains } from '../src/data/learning-curriculum';

const specs = [
  {
    "id": "carnot-cycle-engine-refrigerator",
    "image": "finite-source-work",
    "columns": 5,
    "controls": 7,
    "original": {
      "en": "28714dbb65f26093e5c00521e111b8e6d4452bc518be87b244d30d3215e8dc47",
      "ko": "65e254a26d33b431e1d79fd6d0b2496c36dc8241c9ef473f0933b95508ef545b"
    }
  },
  {
    "id": "two-reservoir-cycle-entropy-admissibility",
    "image": "unrestored-device-reset",
    "columns": 6,
    "controls": 4,
    "original": {
      "en": "1ba3913e39f372c132f95e7da4a55f17105e66e06236218c6ffaebc2c730e019",
      "ko": "faf477ca33a6fb917cf05568d4118cbb04b5969b3d9e6fff5589bbd321f76791"
    }
  },
  {
    "id": "gapped-spectrum-low-temperature-entropy",
    "image": "growing-copy-low-temperature",
    "columns": 6,
    "controls": 5,
    "original": {
      "en": "03951d879a75d0f28cdacb980160361538198dd4a09404d257e3091a5c8672b6",
      "ko": "07723d8e8729643d6d50079d1c510d2344ab9d9352fdda9522837d058c596add"
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
      assert.equal(images[0].url, `/assets/learning/laws-${spec.image}.${lang}.svg`);
      pairs.push({ math, codes, stdout });
    }
    assert.deepEqual(pairs[0], pairs[1]);
    const reader = await getLearningContent(spec.id);
    for (const lang of ['en', 'ko'] as const) {
      assert.ok(!reader[lang].includes('katex-error'));
      assert.ok(reader[lang].includes(`laws-${spec.image}.${lang}.svg`));
      assert.equal(reader.outline[lang].filter(h => /Research extension:|연구 확장:/.test(h.title)).length, 1);
    }
  });
}

test('independent finite sources, composite reset budgets and joint low-temperature limit', () => {
  const report = JSON.parse(execFileSync('python3', ['scripts/learning/verify-thermodynamic-laws-research.py'], { encoding: 'utf8', timeout: 30000 }));
  assert.deepEqual(report, {"status": "passed", "finiteBodyIntegrations": 12, "endpointWorkBounds": 48, "equalTemperaturePreparations": 3, "exactCompositeRunResetBudgets": 18, "nonzeroStoredEnergyBudgets": 2, "finiteCountEnumerations": 12, "jointLimitChecks": 3, "exactProbabilityThresholds": 12});
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
    assert.equal(thermodynamicLawLabs[spec.id].controls.length, spec.controls);
  }
});
