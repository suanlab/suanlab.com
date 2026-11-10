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
const sources = ['ko', 'en'].map(language => readFileSync(root + `driven-two-state-thermodynamics-chapter.${language}.md`, 'utf8'));
const references = JSON.parse(readFileSync('scripts/fixtures/driven-two-state-thermal-reference.json', 'utf8')) as { cases: { label: string; expected: { meanWork: string; heatIntoSystem: string; totalEntropyProduction: string; canonicalFreeEnergyChange: string; workExponentialMoment: string; stages: { stateOneProbability: string; meanEnergy: string }[] } }[] };
for (const [i, language] of ['ko', 'en'].entries()) test(`unregistered driven-thermal ${language} manuscript has graduate theory, valid math and six explained exercises`, async () => {
  const source = sources[i]; assert.equal((source.match(/^## /gm) || []).length, 14); assert.equal((source.match(/\$\$[\s\S]*?\$\$/g) || []).length, 10); assert.equal((source.match(/^\*\*\d\./gm) || []).length, 6);
  const rendered = await unified().use(remarkParse).use(remarkGfm).use(remarkMath).use(remarkRehype).use(rehypeKatex, { strict: 'error' }).use(rehypeStringify).process(source);
  assert.equal(rendered.messages.length, 0); assert(!String(rendered).includes('katex-error'));
  assert(!/\]\((?!https:\/\/)[^)]+\.md\)/.test(source));
  if (language === 'en') assert(!/[가-힣]/.test(source));
});
test('both languages share all10 equations and worked values from direct independent references', () => {
  const equations = sources.map(source => [...source.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m => m[1].replace(/\s/g, ''))); assert.deepEqual(equations[0], equations[1]);
  const worked = references.cases.find(c => c.label === 'equilibrium start')!.expected;
  const numbers = [worked.meanWork, worked.heatIntoSystem, worked.totalEntropyProduction, worked.canonicalFreeEnergyChange, worked.stages[2].stateOneProbability, worked.stages[4].stateOneProbability, worked.stages[4].meanEnergy].map(x => Number(x).toFixed(6));
  numbers.push(Number(references.cases.find(c => c.label === 'nonequilibrium start')!.expected.workExponentialMoment).toFixed(6));
  for (const source of sources) for (const number of numbers) assert(source.replace(/−/g, '-').includes(number), number);
});
