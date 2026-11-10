import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { getLearningContent } from '../src/lib/learning-content';

const id = 'euclidean-quantum-kernels-and-time-slicing';
const root = '.runtime/learning/euclidean-kernel-foundation';
mkdirSync(root, { recursive: true });
const source = (language: string) => readFileSync(`content/learning/physics-math-for-ai/${id}.${language}.md`, 'utf8');
const equations = (raw: string) => [...raw.matchAll(/^\$\$\n([\s\S]*?)\n\$\$/gm)].map(match => match[1]);
const code = (raw: string) => [...raw.matchAll(/```python\n([\s\S]*?)\n```/g)].map(match => match[1]);

test('Euclidean manuscripts render matching equations and complete bilingual teaching structure', async () => {
  const rendered = await getLearningContent(id), rows = [];
  for (const language of ['ko', 'en'] as const) {
    const raw = source(language), html = rendered[language], headings = rendered.outline[language];
    assert.equal(headings.length, 24);
    assert.equal(new Set(headings.map(heading => heading.id)).size, 24);
    assert.equal(equations(raw).length, 15);
    assert.equal((raw.match(/^\d+\. \*\*/gm) ?? []).length, 8);
    assert(!html.includes('katex-error'));
    assert(!/<h1(?:\s|>)/.test(html));
    assert(html.includes('<math') && html.includes('tabindex="0"'));
    const tables = [...raw.matchAll(/^\|---\|/gm)];
    assert.equal(tables.length, 2);
    rows.push({ language, sections: headings.length, displayEquations: equations(raw).length, workedQuestions: 8, handCalculationTables: 1, predictionComparisons: 6 });
  }
  assert(!/[가-힣]/.test(rendered.en));
  assert.deepEqual(equations(source('en')), equations(source('ko')));
  assert.deepEqual(code(source('en')), code(source('ko')));
  writeFileSync(`${root}/manuscripts.json`, JSON.stringify(rows, null, 2) + '\n');
});

test('the published Python exercise checks the pinned one-variable kernel and trace without sampling', () => {
  const snippets = code(source('en'));
  assert.equal(snippets.length, 1);
  const result = spawnSync(process.env.EUCLIDEAN_KERNEL_PYTHON ?? 'python3', ['-c', snippets[0]], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  assert.equal(result.stdout.trim(), '4/9 2/9 49/72 16/17 -1/8');
  writeFileSync(`${root}/python-exercise.json`, JSON.stringify({ exitCode: result.status, output: result.stdout }, null, 2) + '\n');
});
