import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { getLearningContent } from '../src/lib/learning-content';

test('finite-spectrum manuscripts render accessible mathematics and bilingual worked exercises', async () => {
  const id = 'gapped-spectrum-low-temperature-entropy';
  const rendered = await getLearningContent(id);
  const equations: string[][] = [];
  for (const language of ['ko', 'en'] as const) {
    const source = await readFile(`content/learning/physics-math-for-ai/${id}.${language}.md`, 'utf8');
    const html = rendered[language];
    assert.doesNotMatch(html, /katex-error|<h1/);
    assert.match(html, /katex-display/); assert.match(html, /tabindex="0"/); assert.match(html, /role="region"/);
    assert.equal((source.match(/^### (?:연습|Exercise) [1-6]:/gm) || []).length, 6);
    assert.equal((source.match(/\*\*(?:해설|Answer):\*\*/g) || []).length, 6);
    const headings = rendered.outline[language];
    assert.equal(new Set(headings.map(h => h.id)).size, headings.length);
    for (const heading of headings) assert.ok(html.includes(`id="${heading.id}"`));
    if (language === 'en') assert.doesNotMatch(html, /[가-힣]/);
    equations.push([...source.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(x => x[1].replace(/\s+/g, ' ').trim()));
    // Product prose must link to published sources, not private authoring documents.
    assert.doesNotMatch(html, /href="\.\.\//);
  }
  assert.deepEqual(equations[0], equations[1]);
});
