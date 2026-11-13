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
    "id": "finite-ensemble-density-gram-tomography",
    "image": "constrained-tomography",
    "columns": 5,
    "equations": 8,
    "exerciseStyle": "inline",
    "originalSolutions": 8,
    "original": {
      "ko": "0f5c3a96fbb51da5e4ce89654edd91bd83966982472feac9b58df1ea69262827",
      "en": "aa65b5f225aba24f1c3bb80e55bb47561a20747989fbbcff9479c3dd572754c5"
    }
  },
  {
    "id": "qubit-density-spectrum-entropy-dephasing",
    "image": "dephasing-divisibility",
    "columns": 5,
    "equations": 8,
    "exerciseStyle": "inline",
    "originalSolutions": 8,
    "original": {
      "ko": "0e464affa057bab443070826740a0e1e0115fa326ab60d3551a17a616c216bc2",
      "en": "aeb4460bfd254f96c1c2853b0dda5a3dbe25a112ea65668c3a54ce09603fc0c4"
    }
  }
] as const;

for (const spec of specs) {
  test(`${spec.id}: original content and eight solutions survive exactly`, () => {
    for (const lang of ['en', 'ko'] as const) {
      const source = readFileSync(`content/learning/physics-math-for-ai/${spec.id}.${lang}.md`, 'utf8');
      const start = source.search(/^## (Research extension:|연구 확장:)/m);
      const end = source.indexOf('\n## ', start + 4) + 1;
      assert.ok(start >= 0 && end > start);
      const original = source.slice(0, start) + source.slice(end);
      assert.equal(createHash('sha256').update(original).digest('hex'), spec.original[lang]);
      assert.equal((original.match(spec.exerciseStyle === 'inline' ? /^\*\*[1-8]\./gm : /^### [1-8]\. /gm) || []).length, 8);
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
      assert.equal(images[0].url, `/assets/learning/quantum-density-research-${spec.image}.${lang}.svg`);
      pairs.push({ math, codes, stdout });
    }
    assert.deepEqual(pairs[0], pairs[1]);
    const reader = await getLearningContent(spec.id);
    for (const lang of ['en', 'ko'] as const) {
      assert.ok(!reader[lang].includes('katex-error'));
      assert.ok(reader[lang].includes(`quantum-density-research-${spec.image}.${lang}.svg`));
      assert.equal(reader.outline[lang].filter(h => /Research extension:|연구 확장:/.test(h.title)).length, 1);
    }
  });
}

test('independent count sums, constrained likelihood and channel dilation',()=>{
 const r=JSON.parse(execFileSync('python3',['scripts/learning/verify-quantum-density-research.py'],{encoding:'utf8',timeout:60000}));
 assert.equal(r.status,'passed');
 for(const [key,value]of Object.entries({binomialMoments:30,confidencePreparations:27,compositionPairs:36,intermediatePairs:36,dilationPreparations:75,ratePreparations:12}))assert.equal(r[key],value,key);
 assert.ok(r.minimumCoverage>=.5);assert.ok(r.maxDilationResidual<2e-14);
 const source=readFileSync('content/learning/physics-math-for-ai/finite-ensemble-density-gram-tomography.en.md','utf8');
 const code=source.match(/```python\n([\s\S]*?)```/)![1];
 const rows=execFileSync('python3',['-c',code],{encoding:'utf8'}).trim().split('\n').map(row=>row.split(/\s+/));
 assert.equal(rows.length,4);assert.equal(r.maximumLikelihoodFits.length,4);
 for(let i=0;i<rows.length;i++){const expected=r.maximumLikelihoodFits[i];for(const [col,key]of [[0,'ratio'],[1,'x'],[2,'y'],[3,'normSquared'],[4,'gain']]as const)assert.ok(Math.abs(Number(rows[i][col])-expected[key])<5.1e-7,key);}
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
