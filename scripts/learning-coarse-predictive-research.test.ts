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
    "id": "markov-partition-lumpability-memory",
    "image": "conditional-state",
    "columns": 5,
    "equations": 8,
    "exerciseStyle": "inline",
    "original": {
      "ko": "e78107c2d10ffa6121da1cb27ddb5ea1eeefdbc3f9407e07b3df63fd6b34172c",
      "en": "40110b88535219b6e20ff9dc6a9f01ddb51ea93df4a94c262baeab0fd7269786"
    }
  },
  {
    "id": "linear-hidden-memory-markov-limit",
    "image": "initial-layer",
    "columns": 5,
    "equations": 8,
    "exerciseStyle": "inline",
    "original": {
      "ko": "bcdd5ce3d56abfe3ed520ea217c195515a0308cff14e9a59b8a0eb0884ba02cd",
      "en": "b6b61f40ff28afa0e722c9dc7dd3ff6f474b9e702c8369c30ad5c2e6dd428f3c"
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
      assert.equal((original.match(spec.exerciseStyle === 'inline' ? /^\*\*[1-6]\./gm : /^### [1-6]\. /gm) || []).length, 6);
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
      assert.equal(images[0].url, `/assets/learning/coarse-predictive-research-${spec.image}.${lang}.svg`);
      pairs.push({ math, codes, stdout });
    }
    assert.deepEqual(pairs[0], pairs[1]);
    const reader = await getLearningContent(spec.id);
    for (const lang of ['en', 'ko'] as const) {
      assert.ok(!reader[lang].includes('katex-error'));
      assert.ok(reader[lang].includes(`coarse-predictive-research-${spec.image}.${lang}.svg`));
      assert.equal(reader.outline[lang].filter(h => /Research extension:|연구 확장:/.test(h.title)).length, 1);
    }
  });
}

test('independent conditional states, losses, poles and composite flow',()=>{
 const r=JSON.parse(execFileSync('python3',['scripts/learning/verify-coarse-predictive-research.py'],{encoding:'utf8',timeout:60000}));
 assert.equal(r.status,'passed');
 for(const [key,value]of Object.entries({filterTransitionFamilies:4,filterInitialPreparations:5,enumeratedMicroPaths:21840,positiveHistoryPreparations:628,impossibleHistoryPreparations:1892,futureConditionalPreparations:342,linearModePreparations:432,augmentedFlowObservations:3024,exactLaplacePreparations:1296}))assert.equal(r[key],value,key);
 assert.ok(Math.abs(r.stationaryConditionalInformation-0.02105596160130884)<1e-14);
 assert.equal(r.squaredLossImprovement,'1/96');
 assert.ok(r.maxTwoModeFlowDifference<2e-11);
 assert.ok(r.maxIntegrationByPartsResidual<2e-11);
});

test('supplements remain in the full 17-unit domain without new lab controls', () => {
  const domain = curriculumDomains.find(d => d.id === 'nonequilibrium')!;
  assert.equal(domain.unitIds.length, 17);
  assert.equal(domain.topics.length, 10);
  assert.equal(learningUnits.length, 650);
  assert.equal(Object.values(learningGuides).reduce((n, g) => n + g.experiments.length, 0), 1969);
  for (const spec of specs) {
    assert.ok(domain.unitIds.includes(spec.id));
    assert.equal(learningGuides[spec.id].steps.length, 3);
    assert.equal(learningGuides[spec.id].experiments.length, 3);
  }
});
