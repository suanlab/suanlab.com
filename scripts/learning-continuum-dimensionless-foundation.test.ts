import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import * as data from '../src/data/interactive-book';
import prior from './fixtures/pre-continuum-dimensionless-registry-hashes.json';
import { getLearningContent } from '../src/lib/learning-content';

const hash = (value: unknown) => createHash('sha256').update(JSON.stringify(value)).digest('hex');
test('494 prior units and guides and 43 lab groups with 485 entries keep their canonical contents and order', () => {
  const ids = new Set(prior.ids), units = data.learningUnits.filter(u => ids.has(u.id));
  assert.equal(units.length, 494); assert.deepEqual(units.map(u => u.id), prior.ids); assert.equal(hash(units), prior.units);
  assert.equal(hash(Object.fromEntries(units.map(u => [u.id, data.learningGuides[u.id]]))), prior.guides);
  const labs = Object.fromEntries(Object.entries(prior.labEntries).map(([name, keys]) => [name,
    Object.fromEntries(keys.map(key => [key, (data as Record<string, any>)[name][key]]))]));
  assert.equal(hash(labs), prior.labs);
});
for (const [id, equations] of [['advection-diffusion-peclet-boundary-layer', 8], ['reaction-diffusion-damkohler-penetration', 9]] as const) {
  test(`${id}: complete bilingual explanations, matching equations and six answered problems render accessibly`, async () => {
    const result = await getLearningContent(id), math: string[][] = [];
    for (const locale of ['ko', 'en'] as const) {
      const fullSource = await readFile(`content/learning/physics-math-for-ai/${id}.${locale}.md`, 'utf8');
      const start = fullSource.search(/^## (?:Research extension:|연구 확장:)/m);
      assert.ok(start >= 0);
      const end = fullSource.indexOf('\n## ', start + 4) + 1;
      assert.ok(end > start);
      const source = fullSource.slice(0, start) + fullSource.slice(end);
      assert.equal((source.match(/^## /gm) || []).length, 13);
      const html = result[locale], outline = result.outline[locale];
      assert.equal(outline.length, 14); assert.equal(new Set(outline.map(h => h.id)).size, 14);
      for (const h of outline) assert.ok(html.includes(`id="${h.id}"`));
      assert.doesNotMatch(html, /katex-error|<h1/); assert.match(html, /katex-display/);
      assert.match(html, /tabindex="0"/); assert.match(html, /aria-label=/);
      assert.equal((source.match(/^### (?:Exercise|문제) \d/gm) || []).length, 6);
      assert.equal((source.match(/\*\*(?:Answer|풀이):\*\*/g) || []).length, 6);
      if (locale === 'en') { assert.doesNotMatch(html, /[가-힣]/); assert.ok(source.split(/\s+/).length > 1800); }
      else assert.ok(source.length > 6500);
      math.push([...source.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m => m[1].replace(/\s+/g, ' ').trim()));
    }
    assert.equal(math[0].length, equations); assert.deepEqual(math[0], math[1]);
  });
}
