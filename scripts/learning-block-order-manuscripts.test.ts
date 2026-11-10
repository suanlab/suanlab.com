import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { getLearningContent } from '../src/lib/learning-content';

const id = 'microscopic-blocks-and-local-order-parameters';
const root = '.runtime/learning/block-order-foundation';
const stem = `docs/online-learning/physics-math-for-ai/drafts/${id}`;
const source = { en: readFileSync(`${stem}.en.md`, 'utf8'), ko: readFileSync(`${stem}.ko.md`, 'utf8') };
const equations = (s: string) => [...s.matchAll(/^\$\$\n([\s\S]*?)\n\$\$/gm)].map(x => x[1]);
const python = (s: string) => [...s.matchAll(/```python\n([\s\S]*?)\n```/g)].map(x => x[1]);
mkdirSync(root, { recursive: true });

test('both detailed spatial-block drafts render with matching equations, runnable code and explained questions', async () => {
  const cwd = process.cwd(), temp = mkdtempSync(join(tmpdir(), 'block-order-drafts-'));
  const folder = join(temp, 'content/learning/physics-math-for-ai');
  mkdirSync(folder, { recursive: true });
  for (const locale of ['en', 'ko'] as const) writeFileSync(join(folder, `${id}.${locale}.md`), source[locale]);
  try {
    process.chdir(temp);
    const rendered = await getLearningContent(id);
    for (const locale of ['en','ko'] as const) {
      assert.equal(rendered.outline[locale].length, 20);
      assert.equal(new Set(rendered.outline[locale].map(row => row.id)).size, 20);
      assert.equal(equations(source[locale]).length, 9);
      assert.equal((source[locale].match(/^[1-8]\. \*\*/gm) ?? []).length, 8);
      assert(!rendered[locale].includes('katex-error'));
      assert(rendered[locale].includes('<math') && rendered[locale].includes('language-python'));
      assert.equal((rendered[locale].match(/<table\b/g) ?? []).length, 2);
    }
    assert(!/[가-힣]/.test(rendered.en));
    assert.deepEqual(equations(source.en), equations(source.ko));
    assert.deepEqual(python(source.en), python(source.ko));
    writeFileSync(join(cwd, root, 'manuscripts.json'), JSON.stringify({ status: 'draft-render-verified', locales: 2, sectionsPerLocale: 20,
      displayEquationsPerLocale: 9, explainedQuestionsPerLocale: 8, tablesPerLocale: 2, englishHangul: 0, registered: false, browserVerified: false }, null, 2) + '\n');
  } finally {
    process.chdir(cwd);
    rmSync(temp, { recursive: true, force: true });
  }
});

test('the exact same Python example in both drafts reproduces profile and nonneighbor conditional probabilities', () => {
  assert.deepEqual(python(source.en), python(source.ko));
  assert.equal(python(source.en).length, 1);
  const followup = `
uncoupled = chain_weights(6, 2, F(1))
assert uncoupled[2,1,2]*uncoupled[0,1,0] == uncoupled[2,1,0]*uncoupled[0,1,2]
antiferro = chain_weights(6, 2, F(2))
assert all(w > 0 for w in antiferro.values())
`;
  const r = spawnSync('python3', ['-c', python(source.en)[0] + followup], { encoding: 'utf8', timeout: 10000 });
  assert.equal(r.status, 0, r.stderr);
  assert(r.stdout.includes('1/9') && r.stdout.includes('16/25'));
  writeFileSync(`${root}/python-exercise.json`, JSON.stringify({ exitCode: r.status, output: r.stdout, arithmetic: 'Fraction',
    uncoupledFollowupPassed: true, antiferromagneticFollowupPassed: true, publishedColabCourse: false }, null, 2) + '\n');
});
