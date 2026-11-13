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
    "id": "boson-fock-cutoff-boundary-algebra",
    "image": "boundary-history-dynamics",
    "columns": 5,
    "equations": 8,
    "exerciseStyle": "inline",
    "solutions": 8,
    "insertBefore": 16,
    "original": {
      "ko": "fb83c87a0a026e70b5844564092db7472757e932bd2a0b4538decc0f51ce3d26",
      "en": "6573c3de438e3eb18b5c6bd04b96a69648fbe545f47224f2a50f7bcb864601d6"
    }
  },
  {
    "id": "matrix-product-state-contraction-gauge",
    "image": "transfer-correlation-gauge",
    "columns": 5,
    "equations": 8,
    "exerciseStyle": "inline",
    "solutions": 8,
    "insertBefore": 15,
    "original": {
      "ko": "b313bdf6c515ea5d6040ffeda78bdfbf1634bc4e640e4cfb0656bf716f9f7a87",
      "en": "bc510c0e6c812c2575ec5282c843b8f086b423120d68ab19a312119180bf44d5"
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
      assert.equal((original.match(spec.exerciseStyle === 'inline' ? /^\*\*[1-8]\./gm : /^### [1-8]\. /gm) || []).length, spec.solutions);
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
      assert.equal(images[0].url, `/assets/learning/quantum-representation-research-${spec.image}.${lang}.svg`);
      pairs.push({ math, codes, stdout });
    }
    assert.deepEqual(pairs[0], pairs[1]);
    const reader = await getLearningContent(spec.id);
    for (const lang of ['en', 'ko'] as const) {
      assert.ok(!reader[lang].includes('katex-error'));
      assert.ok(reader[lang].includes(`quantum-representation-research-${spec.image}.${lang}.svg`));
      assert.equal(reader.outline[lang].filter(h => /Research extension:|연구 확장:/.test(h.title)).length, 1);
    }
  });
}

test('independent oscillator boundary history and dense MPS transfer/gauge geometry',()=>{
 const r=JSON.parse(execFileSync('python3',['scripts/learning/verify-quantum-representation-research.py'],{encoding:'utf8',timeout:60000}));
 assert.equal(r.status,'passed');
 for(const [key,value]of Object.entries({boundaryPreparations:96,dynamicalComparisons:96,analyticSchrodingerPreparations:24,finiteMpsStates:30,gaugeStateComparisons:60,gaugePowerComparisons:24,jordanComparisons:24,exactAngleCases:4,exactTransferCases:4}))assert.equal(r[key],value,key);
 assert.ok(r.maxMatrixResidual<1e-10);assert.equal(r.exactFiniteReturn,true);assert.ok(r.falseReturnNumericalTop<1e-27);assert.ok(r.falseReturnTraceDistance>.98);
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
