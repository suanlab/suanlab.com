import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { getLearningContent } from '../src/lib/learning-content';

const id = 'saddle-fluctuation-and-nongaussian-approximation';
const root = '.runtime/learning/saddle-fluctuation-manuscripts';
mkdirSync(root, { recursive: true });
const source = (language: string) => readFileSync(`content/learning/physics-math-for-ai/${id}.${language}.md`, 'utf8');
const equations = (raw: string) => [...raw.matchAll(/^\$\$\n([\s\S]*?)\n\$\$/gm)].map(match => match[1]);
const code = (raw: string) => [...raw.matchAll(/~~~python\n([\s\S]*?)\n~~~/g)].map(match => match[1]);

test('saddle manuscripts render complete bilingual derivations with matching equations and executable teaching code', async () => {
  const rendered = await getLearningContent(id), rows = [];
  for (const language of ['ko', 'en'] as const) {
    const raw = source(language), html = rendered[language], headings = rendered.outline[language];
    assert.equal(headings.length, 28);
    assert.equal(new Set(headings.map(heading => heading.id)).size, 28);
    assert.equal(equations(raw).length, 24);
    assert.equal((raw.match(/^\d+\. \*\*/gm) ?? []).length, 8);
    assert.equal((raw.match(/^\|---\|/gm) ?? []).length, 4);
    assert(!/\[\[(?:MATH|PYTHON)/.test(raw));
    assert(!html.includes('katex-error'));
    assert(!/<h1(?:\s|>)/.test(html));
    assert(html.includes('<math') && html.includes('tabindex="0"') && html.includes('language-python'));
    rows.push({ language, sections: headings.length, displayEquations: equations(raw).length, explainedQuestions: 8, tables: 4, predictionComparisons: 6 });
  }
  assert(!/[가-힣]/.test(rendered.en));
  assert.deepEqual(equations(source('en')), equations(source('ko')));
  assert.deepEqual(code(source('en')), code(source('ko')));
  writeFileSync(`${root}/manuscripts.json`, JSON.stringify(rows, null, 2) + '\n');
});

test('published exact Python exercise retains cubic corrections, bounded ratio, residual, endpoint action and a coupled inverse', () => {
  const snippets = code(source('en'));
  assert.equal(snippets.length, 1);
  const result = spawnSync(process.env.SADDLE_FLUCTUATION_PYTHON ?? 'python3', ['-c', snippets[0]], { encoding: 'utf8', timeout: 30000 });
  assert.equal(result.status, 0, result.stderr);
  assert.equal(result.stdout.trim(), [
    'q=0 H=9/2 c3=0 c4=-1/54 c1=-1/54',
    'q=1/2 H=39/8 c3=80/19773 c4=-8/507 c1=-232/19773',
    'q=2 H=21/2 c3=20/3087 c4=-1/294 c1=19/6174',
    'bounds=53/54,17207/17496',
    'residual=97651/2000000',
    'S*=17277169/16777216',
    'N=3 det=105/16 R11=44/105 R12=16/105',
  ].join('\n'));
  writeFileSync(`${root}/python-exercise.json`, JSON.stringify({ exitCode: result.status, output: result.stdout }, null, 2) + '\n');
});
