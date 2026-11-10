import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkRehype from 'remark-rehype';
import rehypeKatex from 'rehype-katex';
import rehypeStringify from 'rehype-stringify';
const root = 'docs/online-learning/physics-math-for-ai/';
const sources = ['ko', 'en'].map(language => readFileSync(root + `reservoir-two-state-thermodynamics-chapter.${language}.md`, 'utf8'));
type Reference = { label: string; expected: { stationaryLaw: string[]; physicalEntropyRate: string; channels: { upRate: string; downRate: string; current: string }[]; path: { conditionalBathWordProbability: string; totalEntropy: string } } };
const references = JSON.parse(readFileSync('scripts/fixtures/reservoir-two-state-thermal-reference.json', 'utf8')) as { cases: Reference[] };
for (const [i, language] of ['ko', 'en'].entries()) test(`reservoir-resolved ${language} manuscript has14 sections,10 equations,two tables and six explanations`, async () => {
  const source = sources[i]; assert.equal((source.match(/^## /gm) || []).length, 14); assert.equal((source.match(/\$\$[\s\S]*?\$\$/g) || []).length, 10); assert.equal((source.match(/^\*\*\d\./gm) || []).length, 6);
  const rendered = await unified().use(remarkParse).use(remarkGfm).use(remarkMath).use(remarkRehype).use(rehypeKatex, { strict: 'error' }).use(rehypeStringify).process(source);
  assert.equal(rendered.messages.length, 0); assert(!String(rendered).includes('katex-error')); assert.equal((String(rendered).match(/<table>/g) || []).length, 2);
  assert(!/\]\((?!https:\/\/)[^)]+\.md\)/.test(source)); if (language === 'en') assert(!/[가-힣]/.test(source));
});
test('bilingual equations and all four conditional-word table entries match independent references', () => {
  const equations = sources.map(source => [...source.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m => m[1].replace(/\s/g, ''))); assert.deepEqual(equations[0], equations[1]);
  const worked = references.cases.find(c => c.label === 'worked/word2/jumps2')!.expected;
  assert.deepEqual(worked.stationaryLaw, ['33/40', '7/40']);
  for (const source of sources) {
    const normalized = source.replace(/−/g, '-'); assert(normalized.includes('π=(33,7)/40')); assert(normalized.includes(Number(worked.physicalEntropyRate).toFixed(6)));
    worked.channels.forEach((c, i) => assert(normalized.includes(`| ${i === 0 ? 'A' : 'B'} | ${c.upRate} | ${c.downRate} | ${c.current} |`)));
    for (const [code, word, entropy] of [[0, 'AA', '0'], [2, 'AB', 'ln3'], [1, 'BA', '-ln3'], [3, 'BB', '0']] as const) {
      const ref = references.cases.find(c => c.label === `worked/word${code}/jumps2`)!.expected.path;
      assert(normalized.includes(`| ${word} | ${ref.conditionalBathWordProbability} | ${entropy} |`));
      const expectedEntropy = code === 2 ? Math.log(3) : code === 1 ? -Math.log(3) : 0; assert(Math.abs(Number(ref.totalEntropy) - expectedEntropy) < 1e-14);
    }
  }
});
