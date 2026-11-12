import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { getLearningContent } from '../src/lib/learning-content';

test('Reynolds closure reading has aligned bilingual equations, graduate explanations, actual units and six answered exercises', async () => {
  const id = 'reynolds-ensemble-stress-closure-defect', result = await getLearningContent(id), equations: string[][] = [];
  for (const locale of ['ko', 'en'] as const) {
    const fullSource = await readFile(`content/learning/physics-math-for-ai/${id}.${locale}.md`, 'utf8'), html = result[locale], headings = result.outline[locale];
    const start = fullSource.search(/^## (?:Research extension:|연구 확장:)/m); assert.ok(start >= 0);
    const end = fullSource.indexOf('\n## ', start + 4) + 1; assert.ok(end > start);
    const source = fullSource.slice(0, start) + fullSource.slice(end);
    assert.equal((source.match(/^## /gm) || []).length, 12);
    assert.equal(headings.length, 13); assert.equal(new Set(headings.map(h => h.id)).size, 13);
    for (const h of headings) assert.ok(html.includes(`id="${h.id}"`));
    assert.doesNotMatch(html, /katex-error|<h1/); assert.match(html, /katex-display/); assert.match(html, /tabindex="0"/); assert.match(html, /aria-label=/);
    assert.equal((source.match(/^\d\. \*\*/gm) || []).length, 6);
    if (locale === 'en') { assert.doesNotMatch(html, /[가-힣]/); assert.ok(source.split(/\s+/).length > 1800); }
    else assert.ok(source.length > 7000);
    assert.match(source, /mastodon.inl.gov/); assert.match(source, /grc.nasa.gov/);
    equations.push([...source.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m => m[1].replace(/\s+/g, ' ').trim()));
  }
  assert.equal(equations[0].length, 11); assert.deepEqual(equations[0], equations[1]);
});
