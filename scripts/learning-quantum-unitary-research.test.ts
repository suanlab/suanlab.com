import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import editorial from './fixtures/quantum-unitary-research-editorial.json';
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
    "id": "two-level-spectral-schrodinger-evolution",
    "image": "ray-speed-zeno",
    "columns": 5,
    "equations": 8,
    "exerciseStyle": "inline",
    "original": {
      "ko": "b9f5031b1fc31316d3d2ac0dd5ffe3660754e49b761a03af126f8fc6e1963f92",
      "en": "e75e24f58d5b83b0ff63417a74abbdf4b0666f8b4c77df0682e58799e6208ce2"
    }
  },
  {
    "id": "piecewise-hamiltonian-pulse-time-ordering",
    "image": "fixed-exposure-splitting",
    "columns": 5,
    "equations": 8,
    "exerciseStyle": "inline",
    "original": {
      "ko": "c92389e1ba499877986a1983b1012470a577765fc9cd56df2527059486425740",
      "en": "4d87fba43e7e87ab631abb873037a138dc1df02dad983e7dfd90a37f0ec43442"
    }
  }
] as const;

for (const spec of specs) {
  test(`${spec.id}: original mathematics and six solutions survive declared editorial corrections`, () => {
    for (const lang of ['en', 'ko'] as const) {
      const source = readFileSync(`content/learning/physics-math-for-ai/${spec.id}.${lang}.md`, 'utf8');
      const start = source.search(/^## (Research extension:|연구 확장:)/m);
      const end = source.indexOf('\n## ', start + 4) + 1;
      assert.ok(start >= 0 && end > start);
      let original = source.slice(0, start) + source.slice(end);
      const path=`content/learning/physics-math-for-ai/${spec.id}.${lang}.md`;
      for(const [before,after]of (editorial as Record<string,string[][]>)[path]||[]){assert.equal(original.split(after).length,2);original=original.replace(after,before);}
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
      assert.equal(images[0].url, `/assets/learning/quantum-unitary-research-${spec.image}.${lang}.svg`);
      pairs.push({ math, codes, stdout });
    }
    assert.deepEqual(pairs[0], pairs[1]);
    const reader = await getLearningContent(spec.id);
    for (const lang of ['en', 'ko'] as const) {
      assert.ok(!reader[lang].includes('katex-error'));
      assert.ok(reader[lang].includes(`quantum-unitary-research-${spec.image}.${lang}.svg`));
      assert.equal(reader.outline[lang].filter(h => /Research extension:|연구 확장:/.test(h.title)).length, 1);
    }
  });
}

test('independent geometry, measured branch products and general-matrix splitting',()=>{
 const r=JSON.parse(execFileSync('python3',['scripts/learning/verify-quantum-unitary-research.py'],{encoding:'utf8',timeout:60000}));
 assert.equal(r.status,'passed');assert.equal(r.geometryPreparations,216);assert.equal(r.zenoBranchPreparations,72);
 assert.equal(r.splittingPreparations,120);assert.equal(r.commutingSchedules,4);
 assert.ok(r.maxSpeedResidual<5e-14);assert.ok(r.maxUnitaryResidual<2e-13);assert.ok(r.maxLieBoundExcess<2e-13);assert.ok(r.maxSymmetricBoundExcess<2e-13);
 const source=readFileSync('content/learning/physics-math-for-ai/piecewise-hamiltonian-pulse-time-ordering.en.md','utf8');
 const code=source.match(/```python\n([\s\S]*?)```/)![1];
 const rows=execFileSync('python3',['-c',code],{encoding:'utf8'}).trim().split('\n').map(row=>row.split(/\s+/));
 for(let i=0;i<rows.length;i++){assert.equal(Number(rows[i][0]),r.pauliTable[i].n);assert.ok(Math.abs(Number(rows[i][1])-r.pauliTable[i].lie)<5.1e-7);assert.ok(Math.abs(Number(rows[i][3])-r.pauliTable[i].symmetric)<5.1e-7);}
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
