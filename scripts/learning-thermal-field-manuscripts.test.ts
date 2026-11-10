import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { getLearningContent } from '../src/lib/learning-content';

const id = 'finite-field-measure-and-thermal-free-energy';
const root = '.runtime/learning/thermal-field-foundation';
const draft = `docs/online-learning/physics-math-for-ai/drafts/${id}`;
const source = { en: readFileSync(`${draft}.en.md`, 'utf8'), ko: readFileSync(`${draft}.ko.md`, 'utf8') };
const equations = (s: string) => [...s.matchAll(/^\$\$\n([\s\S]*?)\n\$\$/gm)].map(x => x[1]);
const code = (s: string) => [...s.matchAll(/```python\n([\s\S]*?)\n```/g)].map(x => x[1]);
mkdirSync(root, { recursive: true });

test('unregistered bilingual drafts render through the actual learning pipeline without math errors or language leakage', async () => {
  // Exercise the real loader in an isolated cwd; never add draft routes to the catalog.
  const cwd = process.cwd(), temporary = mkdtempSync(join(tmpdir(), 'thermal-field-manuscripts-'));
  const folder = join(temporary, 'content/learning/physics-math-for-ai');
  mkdirSync(folder, { recursive: true });
  for (const locale of ['en', 'ko'] as const) writeFileSync(join(folder, `${id}.${locale}.md`), source[locale]);
  try {
    process.chdir(temporary);
    const content = await getLearningContent(id);
    const rows = [];
    for (const locale of ['en', 'ko'] as const) {
      assert.equal(content.outline[locale].length, 21);
      assert.equal(new Set(content.outline[locale].map(x => x.id)).size, 21);
      assert.equal(equations(source[locale]).length, 9);
      assert(!content[locale].includes('katex-error'));
      assert(content[locale].includes('<math') && content[locale].includes('tabindex="0"'));
      assert(content[locale].includes('language-python'));
      assert.equal((source[locale].match(/^\*\*[1-8]\. /gm) ?? []).length, 8);
      rows.push({ locale, sections: 21, displayEquations: 9, explainedQuestions: 8, tables: 1 });
    }
    assert(!/[가-힣]/.test(content.en));
    assert.deepEqual(equations(source.en), equations(source.ko));
    assert.deepEqual(code(source.en), code(source.ko));
    writeFileSync(join(cwd, root, 'manuscripts.json'), JSON.stringify({ status: 'draft-render-verified', registered: false, browserVerified: false, rows }, null, 2) + '\n');
  } finally {
    process.chdir(cwd);
    rmSync(temporary, { recursive: true, force: true });
  }
});

test('the actual bilingual NumPy example reproduces all three thermal-path counterexamples', () => {
  assert.equal(code(source.en).length, 1);
  const check = `
np.testing.assert_allclose(thermal(2, 2.0, 1.0, 0.0, 0.0), (2, 2, 1))
np.testing.assert_allclose(thermal(2, 2.0, 2.0, 1.0, 0.0), (2, 0, 0))
np.testing.assert_allclose(thermal(2, 2.0, 4.0, 4.0, 2.0), (2, -2, -1))
for q in (1, 3, 8):
    for T in (0.5, 1.0, 3.0):
        np.testing.assert_allclose(thermal(q, T, 1.0, 0.0, 0.0), (q*T/2, q*T/2, q/2))
        np.testing.assert_allclose(thermal(q, T, T, 1.0, 0.0), (q*T/2, 0, 0))
        np.testing.assert_allclose(thermal(q, T, T*T, 2*T, 2.0), (q*T/2, -q*T/2, -q/2))
`;
  const result = spawnSync('python3', ['-c', code(source.en)[0] + check], { encoding: 'utf8', timeout: 30000 });
  assert.equal(result.status, 0, result.stderr);
  writeFileSync(`${root}/python-exercise.json`, JSON.stringify({ exitCode: result.status, output: result.stdout,
    numericalDiagnostic: true, independentRationalEvidence: 'independent-fraction-references.json' }, null, 2) + '\n');
});
