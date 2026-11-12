import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { getLearningContent } from '../src/lib/learning-content';

test('Gaussian filtering chapter renders detailed bilingual theory, matched equations and six answered exercises accessibly', async () => {
  const id = 'gaussian-filtered-burgers-subgrid-transfer', result = await getLearningContent(id), equations: string[][] = [];
  for (const locale of ['ko', 'en'] as const) {
    const fullSource = await readFile(`content/learning/physics-math-for-ai/${id}.${locale}.md`, 'utf8'), html = result[locale], outline = result.outline[locale];
    const start = fullSource.search(/^## (?:Research extension:|연구 확장:)/m); assert.ok(start >= 0);
    const end = fullSource.indexOf('\n## ', start + 4) + 1; assert.ok(end > start);
    const source = fullSource.slice(0, start) + fullSource.slice(end);
    assert.equal((source.match(/^## /gm) || []).length, 14);
    assert.equal(outline.length, 15); assert.equal(new Set(outline.map(h => h.id)).size, 15); for (const heading of outline) assert.ok(html.includes(`id="${heading.id}"`));
    assert.doesNotMatch(html, /katex-error|<h1/); assert.match(html, /katex-display/); assert.match(html, /tabindex="0"/); assert.match(html, /aria-label=/); assert.match(html, /<table/);
    assert.equal((source.match(/^\d\. \*\*/gm) || []).length, 6); assert.match(source, /courses.washington.edu/);
    if (locale === 'en') { assert.doesNotMatch(html, /[가-힣]/); assert.ok(source.split(/\s+/).length > 1800); } else assert.ok(source.length > 8000);
    equations.push([...source.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m => m[1].replace(/\s+/g, ' ').trim()));
  }
  assert.equal(equations[0].length, 16); assert.deepEqual(equations[0], equations[1]);
});
