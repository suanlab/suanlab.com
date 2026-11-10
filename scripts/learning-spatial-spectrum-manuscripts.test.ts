import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import test from 'node:test';
import { getLearningContent } from '../src/lib/learning-content';
import { spatialSpectrumExamples } from '../src/lib/learning-spatial-spectrum-input';
mkdirSync('.runtime/learning/spatial-spectrum-numerics', { recursive: true });

const id = 'spatial-spectra-decay-and-correlation-length';
const source = (locale: 'ko' | 'en') => readFileSync(`content/learning/physics-math-for-ai/${id}.${locale}.md`, 'utf8');
const equations = (raw: string) => [...raw.matchAll(/^\$\$\n([\s\S]*?)\n\$\$/gm)].map(match => match[1]);
const code = (raw: string) => [...raw.matchAll(/```python\n([\s\S]*?)\n```/g)].map(match => match[1]);

test('both manuscripts render with strict math, unique navigable sections and no Korean in English', async () => {
  const rendered = await getLearningContent(id);
  const rows = [];
  for (const locale of ['ko', 'en'] as const) {
    const raw = source(locale), html = rendered[locale], headings = rendered.outline[locale];
    assert.equal(headings.length, 22);
    assert.equal(new Set(headings.map(heading => heading.id)).size, 22);
    assert.equal((raw.match(/^\d+\. \*\*/gm) ?? []).length, 8);
    assert.equal(equations(raw).length, 13);
    assert(!html.includes('katex-error'));
    assert(!/<h1(?:\s|>)/.test(html));
    assert(html.includes('7/15') && html.includes('28/15') && html.includes('21/32'));
    assert(html.includes('role="region"') && html.includes('tabindex="0"'));
    for (const example of spatialSpectrumExamples) assert(raw.includes('`' + example.id + '`'), example.id);
    rows.push({ locale, sections: headings.length, displayEquations: equations(raw).length, workedQuestions: 8, characters: raw.length });
  }
  assert(!/[가-힣]/.test(rendered.en));
  assert.deepEqual(equations(source('en')), equations(source('ko')));
  assert.deepEqual(code(source('en')), code(source('ko')));
  writeFileSync('.runtime/learning/spatial-spectrum-numerics/manuscripts.json', JSON.stringify(rows, null, 2) + '\n');
});

test('the published Python exercise executes its covariance, spectrum and Green assertions', () => {
  const snippets = code(source('en'));
  assert.equal(snippets.length, 1);
  const execution = spawnSync(process.env.SPATIAL_SPECTRUM_PYTHON ?? 'python3', ['-c', snippets[0]], { encoding: 'utf8' });
  assert.equal(execution.status, 0, execution.stderr);
  assert(execution.stdout.includes('1.866666'));
  writeFileSync('.runtime/learning/spatial-spectrum-numerics/python-exercise.json', JSON.stringify({ exitCode: execution.status, output: execution.stdout }, null, 2) + '\n');
});
