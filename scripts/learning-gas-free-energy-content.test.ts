import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { getLearningContent } from '../src/lib/learning-content';

test('gas free-energy manuscripts render paired equations, accessible formulas and six worked exercises', async () => {
  const id = 'ideal-gas-helmholtz-gibbs-volume-availability', result = await getLearningContent(id);
  const equations: string[][] = [];
  for (const language of ['ko', 'en'] as const) {
    const source = await readFile(`content/learning/physics-math-for-ai/${id}.${language}.md`, 'utf8'), html = result[language];
    assert.doesNotMatch(html, /katex-error|<h1|href="\.\.\//);
    assert.match(html, /katex-display/); assert.match(html, /role="region"/); assert.match(html, /tabindex="0"/);
    assert.equal((source.match(/^### (?:연습|Exercise) [1-6]:/gm) ?? []).length, 6);
    assert.equal((source.match(/\*\*(?:해설|Answer):\*\*/g) ?? []).length, 6);
    assert.equal((source.match(/^## /gm) ?? []).length, 16);
    assert.equal(new Set(result.outline[language].map(h => h.id)).size, result.outline[language].length);
    for (const heading of result.outline[language]) assert.ok(html.includes(`id="${heading.id}"`));
    if (language === 'en') { assert.doesNotMatch(source, /[가-힣]/); assert.ok(source.split(/\s+/).length > 1600); }
    else assert.ok((source.match(/[가-힣]/g) ?? []).length > 2200);
    equations.push([...source.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(match => match[1].replace(/\s+/g, ' ').trim()));
    assert.match(source, /https:\/\/davidtong\.org\//); assert.match(source, /https:\/\/physics\.nist\.gov\//);
  }
  assert.ok(equations[0].length >= 12); assert.deepEqual(equations[0], equations[1]);
  assert.equal(result.outline.en.length, result.outline.ko.length);
});
