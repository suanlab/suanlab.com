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
    "id": "schmidt-cuts-bond-rank-truncation",
    "image": "sequential-discarded-weight",
    "columns": 5,
    "equations": 8,
    "solutions": 8,
    "exerciseStyle": "inline",
    "insertBefore": 15,
    "original": {
      "ko": "fd46810be1139ef3129363894fb74d5d6eae2329dbf4a121c1c6f4e7c53c3e4d",
      "en": "2700cf763f42f5edce769a789456e2fa6076a2148d50f3677428543a1f51ef16"
    }
  },
  {
    "id": "finite-qubit-circuit-order-interference",
    "image": "parameter-shift-frequency",
    "columns": 5,
    "equations": 8,
    "solutions": 3,
    "exerciseStyle": "named",
    "insertBefore": 7,
    "original": {
      "ko": "f0c4be3c57d0626717ab914371df3e39ddc4feaecb95c640d454c7497c58c09d",
      "en": "bedbcf5702981298f225f4965ec534f40bb62488502acee5598ab9f040e05cff"
    }
  }
] as const;

for (const spec of specs) {
  test(`${spec.id}: original content and original solutions survive exactly`, () => {
    for (const lang of ['en', 'ko'] as const) {
      const source = readFileSync(`content/learning/physics-math-for-ai/${spec.id}.${lang}.md`, 'utf8');
      const start = source.search(/^## (Research extension:|연구 확장:)/m);
      const end = source.indexOf('\n## ', start + 4) + 1;
      assert.ok(start >= 0 && end > start);
      const original = source.slice(0, start) + source.slice(end);
      assert.equal(createHash('sha256').update(original).digest('hex'), spec.original[lang]);
      assert.equal((original.match(spec.exerciseStyle === 'inline' ? /^\*\*[1-8]\./gm : /^\*\*(?:Predict:|Calculate:|Debug:|예측:|계산:|오류 찾기:)/gm) || []).length, spec.solutions);
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
      assert.equal(images[0].url, `/assets/learning/quantum-compression-gradient-research-${spec.image}.${lang}.svg`);
      pairs.push({ math, codes, stdout });
    }
    assert.deepEqual(pairs[0], pairs[1]);
    const reader = await getLearningContent(spec.id);
    for (const lang of ['en', 'ko'] as const) {
      assert.ok(!reader[lang].includes('katex-error'));
      assert.ok(reader[lang].includes(`quantum-compression-gradient-research-${spec.image}.${lang}.svg`));
      assert.equal(reader.outline[lang].filter(h => /Research extension:|연구 확장:/.test(h.title)).length, 1);
    }
  });
}

test('independent sequential projection geometry, dense circuit derivatives and exact shot variances',()=>{
 const r=JSON.parse(execFileSync('python3',['scripts/learning/verify-quantum-compression-gradient-research.py'],{encoding:'utf8',timeout:60000}));
 assert.equal(r.status,'passed');
 for(const [key,value]of Object.entries({sequentialStates:60,orthogonalDiscardPairs:240,bondRankChecks:180,localUnitaryCutComparisons:180,analyticSweepExamples:4,degenerateSweeps:2,circuitPreparations:52,generalTwoEigenvalueCases:24,controlledFourShiftCases:24,controlledInterferenceCases:17,exactBinomialVarianceCases:4}))assert.equal(r[key],value,key);
 assert.equal(r.strictlyBetterProductCompetitor,true);assert.ok(r.maxResidual<1e-10);assert.equal(r.symbolicSharedParameterCounterexample,true);
});

test('supplements remain in the full 34-unit domain without new lab controls', () => {
  const domain = curriculumDomains.find(d => d.id === 'quantum-physics')!;
  assert.equal(domain.unitIds.length, 34);
  assert.equal(domain.topics.length, 11);
  assert.equal(learningUnits.length, 650);
  assert.equal(Object.values(learningGuides).reduce((n, g) => n + g.experiments.length, 0), 1969);
  for (const spec of specs) {
    assert.ok(domain.unitIds.includes(spec.id));
    assert.equal(learningGuides[spec.id].steps.length, 3);
    assert.equal(learningGuides[spec.id].experiments.length, 3);
  }
});
