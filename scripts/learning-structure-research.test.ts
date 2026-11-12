import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import { getLearningContent } from '../src/lib/learning-content';
import { quarticDiscreteActionStep, circleRattleProjectionPhase } from '../src/lib/learning-structure-preserving';
for (const [id, equations, image] of [
  ['quartic-discrete-action-splitting', 8, 'analytical-structure-long-phase'],
  ['circle-rattle-projection-phase', 7, 'analytical-structure-observation-clock'],
] as const) {
  test(`structure research paired proof, table and executable cell: ${id}`, async () => {
    const pairs = ['en', 'ko'].map(lang => {
      const source = readFileSync(`content/learning/physics-math-for-ai/${id}.${lang}.md`, 'utf8');
      assert.doesNotMatch(source, /[\u0000-\u0008\u000b-\u001f]/);
      const start = source.search(/^## (Research extension:|연구 확장:)/m);
      const tail = source.slice(start);
      const relativeEnd = tail.search(/^## (Work through six checks with explanations|해설과 함께 여섯 가지를 확인한다)$/m);
      const end = relativeEnd < 0 ? -1 : start+relativeEnd;
      assert.ok(start >= 0 && end > start);
      const tree = unified().use(remarkParse).use(remarkGfm).use(remarkMath).parse(source.slice(start, end));
      const rejectAccidentalEmphasis = (node: { type: string; children?: { type: string; children?: any[] }[] }): void => {
        assert.notEqual(node.type, 'emphasis', 'Starred reference symbols must use inline math rather than Markdown emphasis.');
        node.children?.forEach(rejectAccidentalEmphasis);
      };
      rejectAccidentalEmphasis(tree);
      const math = tree.children.flatMap(node => node.type === 'math' ? [node.value] : []);
      const python = tree.children.flatMap(node => node.type === 'code' && node.lang === 'python' ? [node.value] : []);
      const tables = tree.children.filter(node => node.type === 'table');
      assert.equal(math.length, equations);
      assert.equal(python.length, 1);
      assert.equal(tables.length, 1);
      for (const row of tables[0].children) assert.equal(row.children.length, 4);
      if (lang === 'en') assert.doesNotMatch(source, /[가-힣]/);
      return { math, python, stdout: execFileSync('python3', ['-c', python[0]], { encoding: 'utf8', timeout: 20_000 }) };
    });
    assert.deepEqual(pairs[0], pairs[1]);
    const rows = pairs[0].stdout.trim().split('\n').map(line => line.split(/\s+/).map(value => value === 'None' ? null : Number(value)));
    assert.deepEqual(rows, id === 'quartic-discrete-action-splitting'
      ? [[.2, 3000, 1.004527, .005], [.1, 24000, 1.001127, .00125], [.05, 192000, 1.000281, .0003125], [.025, 1536000, 1.000070, .000078125]]
      : [[1.645811, 4.495782, 6.141593, .887583], [3.141593, 3, 6.141593, -.5], [4.637374, 1.504218, 6.141593, .887583]]);
    const content = await getLearningContent(id);
    for (const lang of ['en', 'ko'] as const) {
      assert.ok(!content[lang].includes('katex-error'));
      assert.ok(content[lang].includes(`${image}.${lang}.svg`));
      assert.equal(content.outline[lang].filter(h => /Research extension:|연구 확장:/.test(h.title)).length, 1);
    }
  });
}

test('independent invariant, phase, controller and clock checks cover the declared families', () => {
  const result = JSON.parse(execFileSync('python3', ['scripts/learning/verify-structure-research.py', '--json'], { encoding: 'utf8', timeout: 20_000 }));
  assert.deepEqual(result, { accumulatedClockTrajectories: 12, closedFormStates: 1620, controlledRattleStates: 108, exactInvariantStates: 1944, foldCollisionStates: 3, harmonicMatrixCases: 54, longHorizonCases: 4, modifiedGeneratorCases: 36, observationClockStates: 1458, stabilityBoundaryStates: 405 });
});

test('native harmonic Verlet preserves the exact quadratic invariant in its regular limit', () => {
  const h = .25;
  const r = quarticDiscreteActionStep('verlet', 1, .5, h, { mass: 1, stiffness: 1, quartic: 0 });
  assert.ok(r.state && r.jacobian);
  assert.equal(r.state.position, 1.09375);
  assert.equal(r.state.momentum, .23828125);
  const k = (q: number, p: number) => p*p/2+(1-h*h/4)*q*q/2;
  assert.ok(Math.abs(k(r.state.position,r.state.momentum)-k(1,.5)) < 1e-12);
});

test('native fixed-step RATTLE keeps unit reduced determinant and its known phase error', () => {
  const q = circleRattleProjectionPhase([1, 2, 2, 10, 8, 0, 0, 8]).rattle.query!;
  assert.ok(Math.abs(q.determinant!-1) < 1e-12);
  assert.ok(Math.abs(q.phaseError-.1887902047863923) < 1e-12);
  assert.ok(Math.abs(q.energy-.5) < 1e-12);
});
