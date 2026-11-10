import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import * as data from '../src/data/interactive-book';
import prior from './fixtures/pre-continuum-euler-navier-stokes-registry-hashes.json';
import { getLearningContent } from '../src/lib/learning-content';
const hash = (value: unknown) => createHash('sha256').update(JSON.stringify(value)).digest('hex');
test('486 prior lessons, guides and 43 lab groups with 477 entries retain canonical content and order', () => {
  const ids = new Set(prior.ids), units = data.learningUnits.filter(u => ids.has(u.id));
  assert.equal(units.length, 486); assert.deepEqual(units.map(u => u.id), prior.ids); assert.equal(hash(units), prior.units);
  assert.equal(hash(Object.fromEntries(units.map(u => [u.id, data.learningGuides[u.id]]))), prior.guides);
  const labs = Object.fromEntries(Object.entries(prior.labEntries).map(([name, keys]) => [name,
    Object.fromEntries(keys.map(key => [key, (data as Record<string, any>)[name][key]]))]));
  assert.equal(hash(labs), prior.labs);
});
for (const [id, count] of [['linearized-euler-acoustic-characteristics', 14], ['taylor-green-navier-stokes-viscous-decay', 16]] as const) {
  test(`${id}: detailed matched bilingual equations, full outlines and six answered exercises render`, async () => {
    const result = await getLearningContent(id), math: string[][] = [];
    for (const locale of ['ko', 'en'] as const) {
      const source = await readFile(`content/learning/physics-math-for-ai/${id}.${locale}.md`, 'utf8');
      const html = result[locale], outline = result.outline[locale];
      assert.equal(outline.length, count); assert.equal(new Set(outline.map(h => h.id)).size, count);
      for (const h of outline) assert.ok(html.includes(`id="${h.id}"`));
      assert.doesNotMatch(html, /katex-error|<h1/); assert.match(html, /katex-display/);
      assert.match(html, /tabindex="0"/); assert.match(html, /aria-label=/);
      assert.equal((source.match(/^### (?:Exercise|연습) \d/gm) || []).length, 6);
      assert.equal((source.match(/\*\*(?:Answer|해설):\*\*/g) || []).length, 6);
      if (locale === 'en') { assert.doesNotMatch(html, /[가-힣]/); assert.ok(source.split(/\s+/).length > 1700); }
      else assert.ok(source.length > 6500);
      math.push([...source.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m => m[1].replace(/\s+/g, ' ').trim()));
    }
    assert.equal(math[0].length, 13); assert.deepEqual(math[0], math[1]);
  });
}
