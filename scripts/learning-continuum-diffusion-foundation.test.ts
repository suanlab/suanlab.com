import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import * as data from '../src/data/interactive-book';
import prior from './fixtures/pre-continuum-diffusion-registry-hashes.json';
import { getLearningContent } from '../src/lib/learning-content';
const hash = (value: unknown) => createHash('sha256').update(JSON.stringify(value)).digest('hex');
test('488 prior lessons, guides and 43 lab groups with 479 entries retain canonical content and order', () => {
  const ids = new Set(prior.ids), units = data.learningUnits.filter(u => ids.has(u.id));
  assert.equal(units.length, 488); assert.deepEqual(units.map(u => u.id), prior.ids); assert.equal(hash(units), prior.units);
  assert.equal(hash(Object.fromEntries(units.map(u => [u.id, data.learningGuides[u.id]]))), prior.guides);
  const labs = Object.fromEntries(Object.entries(prior.labEntries).map(([name, keys]) => [name,
    Object.fromEntries(keys.map(key => [key, (data as Record<string, any>)[name][key]]))]));
  assert.equal(hash(labs), prior.labs);
});
for (const [id, count] of [['fourier-heat-diffusion-mode-budget', 15], ['gaussian-fick-diffusion-spreading', 17]] as const) {
  test(`${id}: detailed matched bilingual equations, full outlines and six answered exercises render`, async () => {
    const result = await getLearningContent(id), math: string[][] = [];
    for (const locale of ['ko', 'en'] as const) {
      const fullSource = await readFile(`content/learning/physics-math-for-ai/${id}.${locale}.md`, 'utf8');
      const start = fullSource.search(/^## (?:Research extension:|연구 확장:)/m);
      assert.ok(start >= 0);
      const end = fullSource.indexOf('\n## ', start + 4) + 1;
      assert.ok(end > start);
      const source = fullSource.slice(0, start) + fullSource.slice(end);
      assert.equal((source.match(/^## /gm) || []).length, count);
      const html = result[locale], outline = result.outline[locale];
      assert.equal(outline.length, count + 1); assert.equal(new Set(outline.map(h => h.id)).size, count + 1);
      for (const h of outline) assert.ok(html.includes(`id="${h.id}"`));
      assert.doesNotMatch(html, /katex-error|<h1/); assert.match(html, /katex-display/);
      assert.match(html, /tabindex="0"/); assert.match(html, /aria-label=/);
      assert.equal((source.match(/^### (?:Exercise|연습) \d/gm) || []).length, 6);
      assert.equal((source.match(/\*\*(?:Answer|해설):\*\*/g) || []).length, 6);
      if (locale === 'en') { assert.doesNotMatch(html, /[가-힣]/); assert.ok(source.split(/\s+/).length > 1700); }
      else assert.ok(source.length > 6500);
      math.push([...source.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m => m[1].replace(/\s+/g, ' ').trim()));
    }
    assert.equal(math[0].length, id === 'fourier-heat-diffusion-mode-budget' ? 12 : 14); assert.deepEqual(math[0], math[1]);
  });
}
