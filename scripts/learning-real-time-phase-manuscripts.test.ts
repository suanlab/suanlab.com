import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { getLearningContent } from '../src/lib/learning-content';

const id = 'real-time-phase-and-classical-limit';
const root = '.runtime/learning/real-time-phase-manuscripts';
mkdirSync(root, { recursive: true });
const source = (language: string) => readFileSync(`content/learning/physics-math-for-ai/${id}.${language}.md`, 'utf8');
const equations = (raw: string) => [...raw.matchAll(/^\$\$\n([\s\S]*?)\n\$\$/gm)].map(match => match[1]);
const code = (raw: string) => [...raw.matchAll(/~~~python\n([\s\S]*?)\n~~~/g)].map(match => match[1]);

test('real-time manuscripts render bilingual math and preserve the distinctions between phase, probability and classical limits', async () => {
  const rendered = await getLearningContent(id);
  const rows = [];
  for (const language of ['ko', 'en'] as const) {
    const raw = source(language), html = rendered[language], headings = rendered.outline[language];
    assert.equal(headings.length, 33);
    assert.equal(new Set(headings.map(heading => heading.id)).size, 33);
    assert.equal(equations(raw).length, 22);
    const questions = raw.split(/^## 32\./m)[1].split(/^## 33\./m)[0];
    assert.equal((questions.match(/^\d+\. \*\*/gm) ?? []).length, 8);
    assert.equal((raw.match(/^\|---\|/gm) ?? []).length, 2);
    assert(!/\[\[(?:MATH|PYTHON)/.test(raw));
    assert(!html.includes('katex-error'));
    assert(!/<h1(?:\s|>)/.test(html));
    assert(html.includes('<math') && html.includes('tabindex="0"') && html.includes('language-python'));
    rows.push({ language, sections: headings.length, displayEquations: 22, explainedQuestions: 8, tables: 2, predictionComparisons: 6 });
  }
  assert(!/[가-힣]/.test(rendered.en));
  assert.deepEqual(equations(source('en')), equations(source('ko')));
  assert.deepEqual(code(source('en')), code(source('ko')));
  writeFileSync(`${root}/manuscripts.json`, JSON.stringify(rows, null, 2) + '\n');
});

test('published Python exercise checks negative modes, invertible zero pivot, caustic compatibility and a position-only focusing counterexample', () => {
  const snippets = code(source('en'));
  assert.equal(snippets.length, 1);
  const result = spawnSync('python3', ['-c', snippets[0]], { encoding: 'utf8', timeout: 30000 });
  assert.equal(result.status, 0, result.stderr);
  assert.equal(result.stdout.trim(), [
    'negative-mode q=-1 S=1/4 nu=1',
    'zero-pivot det=-1 inverse=H nu=1',
    'caustic compatible=True incompatible=False',
    'h=1/2 focus_qq=3/32 focus_pp=73/96 unchirped_qq=73/96 scaled_qq=25/48',
    'h=1/4 focus_qq=3/128 focus_pp=265/384 unchirped_qq=265/384 scaled_qq=25/96',
    'h=1/8 focus_qq=3/512 focus_pp=1033/1536 unchirped_qq=1033/1536 scaled_qq=25/192',
  ].join('\n'));
  writeFileSync(`${root}/python-exercise.json`, JSON.stringify({ exitCode: result.status, output: result.stdout }, null, 2) + '\n');
});
