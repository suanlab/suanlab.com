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
    "id": "two-oscillator-locking-phase-slips",
    "image": "slip-occupation",
    "columns": 5,
    "equations": 8,
    "exerciseStyle": "inline",
    "original": {
      "ko": "5586c26ee23bb1897bb1619a58ccd6cc1292e3da6700c965792bcbb3c686f215",
      "en": "ad0a65ed8a882e7b3802d92b16a386a9432ce425c61e9e88d65a551e5f9b8456"
    }
  },
  {
    "id": "finite-kuramoto-collective-order",
    "image": "balanced-spectrum",
    "columns": 5,
    "equations": 8,
    "exerciseStyle": "inline",
    "original": {
      "ko": "7296249ad60024dcc8072c2af6b31df75fab16315c68aa02fbabf8d7aa245b19",
      "en": "b80fa56d26982673929d53077df445de6a7efa6f210a782d31c3eef9714a38e8"
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
      const corrections={"two-oscillator-locking-phase-slips": {"en": ["The present numerical foundation and draft still await Worker, teaching UI and reader registration.", "The numerical foundation, Worker, teaching UI and reader are now registered; the research progress record links their integration evidence."], "ko": ["현재 원고·수치 기반은 Worker·교수 화면·읽기 등록 전 단계다.", "현재 원고·수치 기반과 Worker·교수 화면·reader가 등록되어 있으며 연구 진행표에 통합 검증 근거를 연결했다."]}} as Record<string, Record<string, readonly string[]>>;
      const change=corrections[spec.id]?.[lang];
      if(change){assert.ok(original.includes(change[1]));original=original.replace(change[1],change[0]);}
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
      assert.equal(images[0].url, `/assets/learning/synchronization-statistics-research-${spec.image}.${lang}.svg`);
      pairs.push({ math, codes, stdout });
    }
    assert.deepEqual(pairs[0], pairs[1]);
    const reader = await getLearningContent(spec.id);
    for (const lang of ['en', 'ko'] as const) {
      assert.ok(!reader[lang].includes('katex-error'));
      assert.ok(reader[lang].includes(`synchronization-statistics-research-${spec.image}.${lang}.svg`));
      assert.equal(reader.outline[lang].filter(h => /Research extension:|연구 확장:/.test(h.title)).length, 1);
    }
  });
}

test('independent occupation, window bias and balanced spectra', () => {
 const report=JSON.parse(execFileSync('python3',['scripts/learning/verify-synchronization-statistics-research.py'],{encoding:'utf8',timeout:60000}));
 assert.equal(report.status,'passed');
 assert.equal(report.occupationPreparations,20);
 assert.equal(report.reflectionPoints,820);
 assert.equal(report.periodSensitivityPreparations,16);
 assert.equal(report.scalarFlowPreparations,80);
 assert.equal(report.finiteWindowObservations,240);
 assert.equal(report.exactBalancedPreparations,627);
 assert.equal(report.jacobianDifferencePreparations,90);
 assert.equal(report.alignedPreparations,48);
 assert.equal(report.exactCohesiveLockedPreparations,14);
 assert.ok(report.maxOccupationQuadratureError<2e-11);
 assert.ok(report.maxFullTurnPhaseError<2e-8);
 assert.ok(report.maxUniformClockError<2e-8);
 assert.ok(report.maxSpectrumError<2e-12);
 assert.ok(report.maxJacobianDifferenceError<2e-9);
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
test('integration update retains the previous Hopf contract',()=>{
 let source=readFileSync('scripts/learning-synchronization-content.test.ts','utf8');
 const revisions=[["  assert.equal(original,draft.slice", "  if(kind!==kinds[0]){\n   const start=source.search(/^## (Research extension:|연구 확장:)/m),end=source.indexOf('\\n## ',start+4)+1;\n   assert(start>=0&&end>start);original=source.slice(0,start)+source.slice(end);\n   const corrections={\"two-oscillator-locking-phase-slips\": {\"en\": [\"The present numerical foundation and draft still await Worker, teaching UI and reader registration.\", \"The numerical foundation, Worker, teaching UI and reader are now registered; the research progress record links their integration evidence.\"], \"ko\": [\"현재 원고·수치 기반은 Worker·교수 화면·읽기 등록 전 단계다.\", \"현재 원고·수치 기반과 Worker·교수 화면·reader가 등록되어 있으며 연구 진행표에 통합 검증 근거를 연결했다.\"]}} as Record<string,Record<string,readonly string[]>>;\n   const correction=corrections[kind]?.[lang];\n   if(correction)original=original.replace(correction[1],correction[0]);\n  }\n  assert.equal(original,draft.slice"], ["length,kind===kinds[0]?15:14)", "length,15)"], ["length,kind===kinds[0]?18:10)", "length,18)"]] as const;
 for(const [before,after]of [...revisions].reverse()){assert.ok(source.includes(after));source=source.replaceAll(after,before);}
 assert.equal(createHash('sha256').update(source).digest('hex'),"f8d7a41bc1ee6ac5c59b02f48983a6a52bfc9a6724659d1672279592ed35718e");
});
