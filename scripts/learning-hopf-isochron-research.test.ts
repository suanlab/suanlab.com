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
    "id": "hopf-amplitude-critical-relaxation",
    "image": "phase-response",
    "columns": 5,
    "equations": 8,
    "exerciseStyle": "inline",
    "original": {
      "ko": "4c0c04a3927cb1b1a021094394c6ceb8f279edf7f1e9434cbd3317a4d3bf3c87",
      "en": "3457c3617f645ea91d97ee473d63ab05fd59db3f331f88ccc2aa8508312ed010"
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
      let original = source.slice(0, start) + source.slice(end);
      const corrections = {"en": ["Reader, Worker and visualization registration are still pending, as are the pairwise and collective families.", "The reader, Worker, visualization and the pairwise and collective families are now registered; their integration evidence is linked in the research progress record."], "ko": ["아직 읽기 화면·Worker·시각화 등록을 하지 않았으며 두 진동자와 다수 진동자의 가족도 남아 있다.", "현재는 읽기 화면·Worker·시각화와 두 진동자·다수 진동자 가족이 모두 등록되어 있으며 통합 검증 근거는 연구 진행표에 연결되어 있다."]} as const;
      assert.ok(original.includes(corrections[lang][1]));
      original = original.replace(corrections[lang][1], corrections[lang][0]);
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
      assert.equal(images[0].url, `/assets/learning/hopf-isochron-research-${spec.image}.${lang}.svg`);
      pairs.push({ math, codes, stdout });
    }
    assert.deepEqual(pairs[0], pairs[1]);
    const reader = await getLearningContent(spec.id);
    for (const lang of ['en', 'ko'] as const) {
      assert.ok(!reader[lang].includes('katex-error'));
      assert.ok(reader[lang].includes(`hopf-isochron-research-${spec.image}.${lang}.svg`));
      assert.equal(reader.outline[lang].filter(h => /Research extension:|연구 확장:/.test(h.title)).length, 1);
    }
  });
}

test('independent isochron, adjoint and finite-pulse calculations', () => {
 const report=JSON.parse(execFileSync('python3',['scripts/learning/verify-hopf-isochron-research.py'],{encoding:'utf8',timeout:60000}));
 assert.equal(report.status,'passed');
 assert.equal(report.exactRationalPreparations,306);
 assert.equal(report.cartesianFlowPreparations,108);
 assert.equal(report.complexPulsePreparations,720);
 assert.equal(report.pulseRelaxationPreparations,18);
 assert.ok(report.maxPhaseCircleError<2e-10);
 assert.ok(report.maxRadialCoordinateError<2e-10);
 assert.ok(report.maxPulseRelaxationError<2e-10);
 assert.ok(report.maxRemainderBoundRatio<=1);
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
