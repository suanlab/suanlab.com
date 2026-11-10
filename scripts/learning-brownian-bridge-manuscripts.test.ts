import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { getLearningContent } from '../src/lib/learning-content';
const id = 'brownian-bridge-time-slicing-measure';
const root = '.runtime/learning/brownian-bridge-numerics';
mkdirSync(root, { recursive: true });
const source = (language: string) => readFileSync(`content/learning/physics-math-for-ai/${id}.${language}.md`, 'utf8');
const equations = (raw: string) => [...raw.matchAll(/^\$\$\n([\s\S]*?)\n\$\$/gm)].map(m => m[1]);
const code = (raw: string) => [...raw.matchAll(/```python\n([\s\S]*?)\n```/g)].map(m => m[1]);
test('both bridge manuscripts render strict equations, unique sections and explained questions', async () => {
  const rendered = await getLearningContent(id), rows = [];
  for (const language of ['ko', 'en'] as const) {
    const raw = source(language), html = rendered[language], headings = rendered.outline[language];
    assert.equal(headings.length, 20); assert.equal(new Set(headings.map(h => h.id)).size, 20);
    assert.equal(equations(raw).length, 12); assert.equal((raw.match(/^\d+\. \*\*/gm) ?? []).length, 8);
    assert(!html.includes('katex-error')); assert(!/<h1(?:\s|>)/.test(html));
    assert(html.includes('<math') && html.includes('tabindex="0"'));
    rows.push({ language, sections: headings.length, displayEquations: equations(raw).length, workedQuestions: 8 });
  }
  assert(!/[가-힣]/.test(rendered.en)); assert.deepEqual(equations(source('en')), equations(source('ko'))); assert.deepEqual(code(source('en')), code(source('ko')));
  writeFileSync(`${root}/manuscripts.json`, JSON.stringify(rows, null, 2) + '\n');
});
test('published exact Python exercise verifies covariance inversion and changing area quadrature', () => {
  const snippets = code(source('en')); assert.equal(snippets.length, 1);
  const result = spawnSync(process.env.BROWNIAN_BRIDGE_PYTHON ?? 'python3', ['-c', snippets[0]], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr); assert.equal(result.stdout.trim(), '1/2 5/32 21/128');
  writeFileSync(`${root}/python-exercise.json`, JSON.stringify({ exitCode: result.status, output: result.stdout }, null, 2) + '\n');
});
