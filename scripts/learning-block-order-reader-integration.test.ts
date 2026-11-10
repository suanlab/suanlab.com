import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { build } from 'esbuild';
import { learningUnits, learningGuides } from '../src/data/interactive-book';
import { curriculumDomains } from '../src/data/learning-curriculum';
import { blockOrderReader } from '../src/data/block-order-reader';
import { blockOrderTeachingId as id } from '../src/lib/learning-block-order-teaching';
import { getLearningContent } from '../src/lib/learning-content';
const before = JSON.parse(readFileSync('scripts/fixtures/pre-block-order-registry-hashes.json','utf8')) as {
  unitOrder: string[]; guideOrder: string[]; units: Record<string,string>; guides: Record<string,string>; domains: Record<string,string>; manuscripts: Record<string,string>; preservedFiles: Record<string,string>;
};
const hash = (value: unknown) => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const root = '.runtime/learning/block-order-reader-integration'; mkdirSync(root,{ recursive: true });
test('new local-order reader appends after602 and preserves prior units,guides,1204manuscripts,menu andshared graphics', () => {
  assert(learningUnits.length >= 603); assert.equal(new Set(learningUnits.map(u => u.id)).size,learningUnits.length);
  assert.deepEqual(learningUnits.slice(0,602).map(u => u.id),before.unitOrder);
  assert.deepEqual(Object.keys(learningGuides).slice(0,602),before.guideOrder);
  for (const unit of learningUnits.slice(0,602)) { assert.equal(hash(unit),before.units[unit.id]); assert.equal(hash(learningGuides[unit.id]),before.guides[unit.id]); }
  for (const [p,h] of Object.entries({ ...before.manuscripts,...before.preservedFiles })) assert.equal(createHash('sha256').update(readFileSync(p)).digest('hex'),h,p);
  for (const domain of curriculumDomains) assert.equal(hash({ ...domain,unitIds: domain.unitIds.filter(u => before.unitOrder.includes(u)) }),before.domains[domain.id]);
  assert.deepEqual(learningUnits[602],blockOrderReader.unit); assert.deepEqual(learningGuides[id],blockOrderReader.guide);
  assert.equal(learningUnits.slice(0,603).reduce((s,u) => s+learningGuides[u.id].experiments.length,0),1875);
  assert(blockOrderReader.unit.prerequisiteIds.every(u => before.unitOrder.includes(u)));
  assert(curriculumDomains.find(d => d.id === 'field-multiscale')!.unitIds.includes(id));
  const coverage = JSON.parse(readFileSync('docs/online-learning/physics-math-for-ai/curriculum/implementation-coverage.json','utf8'));
  assert(coverage.domains.find((d: { id: string }) => d.id === 'field-multiscale').implementedUnitIds.includes(id));
  writeFileSync(`${root}/preservation.json`,JSON.stringify({ previousUnits: 602,preservedManuscripts: 1204,registeredPrefixUnits: 603,registeredPrefixGuides: 1875,
    menuPreserved: true,sharedGraphicsPreserved: true,coverageLedgerIncludesUnit: true },null,2)+'\n');
});
test('registered bilingual theory renders twenty sections,nine equations andthe identical executed Python example', async () => {
  const content = await getLearningContent(id);
  const rows = [];
  const snippets = (s: string) => [...s.matchAll(/```python\n([\s\S]*?)\n```/g)].map(m => m[1]);
  for (const locale of ['ko','en'] as const) {
    assert.equal(content.outline[locale].length,20); assert(!content[locale].includes('katex-error')); assert(content[locale].includes('<math'));
    const source = readFileSync(`content/learning/physics-math-for-ai/${id}.${locale}.md`,'utf8');
    assert.equal((source.match(/^\$\$/gm) ?? []).length,18); assert(!source.includes('preparation draft') && !source.includes('published reader and production assets remains pending'));
    assert.deepEqual(snippets(source),snippets(readFileSync(`docs/online-learning/physics-math-for-ai/drafts/${id}.${locale}.md`,'utf8')));
    rows.push({ locale,sections: 20,displayEquations: 9,pythonExamples: snippets(source).length,pythonSnippetHash: hash(snippets(source)),sameAsExecutedDraft: true });
  }
  assert(!/[가-힣]/.test(content.en)); assert.equal(blockOrderReader.unit.questions.length,8); assert.equal(blockOrderReader.guide.steps.length,6); assert.equal(blockOrderReader.guide.experiments.length,6);
  const lab = readFileSync('src/components/learning/block-order-lab.tsx','utf8'); assert(lab.includes('!embedded'));
  assert(readFileSync('src/components/learning/block-order-reader-lab.tsx','utf8').includes('embedded'));
  assert(readFileSync('src/components/learning/interactive-book.tsx','utf8').includes('isBlockOrderTeachingId(unit.id) ? <BlockOrderReaderLab'));
  writeFileSync(`${root}/registered-manuscripts.json`,JSON.stringify({ status: 'registered-manuscripts-verified',rows,englishHangul: 0,readerQuestions: 8,readerGuideSteps: 6,
    readerExperiments: 6,pythonExecutionEvidence: '.runtime/learning/block-order-foundation/python-exercise.json' },null,2)+'\n');
});
test('complete reader excludes local-order engines,andthe new reader lab excludes shared transcendental arithmetic', async () => {
  const bundle = await build({ entryPoints: ['src/components/learning/interactive-book.tsx'],bundle: true,write: false,platform: 'browser',format: 'esm',splitting: true,outdir: root+'/graph',metafile: true,jsx: 'automatic',logLevel: 'silent' });
  const inputs = Object.keys(bundle.metafile!.inputs);
  for (const name of ['learning-block-order.ts','learning-block-order-evaluator.ts','learning-block-order-observations.ts','learning-block-order-validation.ts','learning-block-order-validation-evaluator.ts','learning-real-time-checker-interval.ts']) assert(!inputs.includes('src/lib/'+name),name);
  writeFileSync(`${root}/reader-dependency-graph.json`,JSON.stringify(bundle.metafile,null,2)+'\n');
  // The older free-energy reader already imports interval-transcendentals via
  // canonical-gap. Check that arithmetic is absent from this unit's own lab.
  const unit = await build({ entryPoints: ['src/components/learning/block-order-reader-lab.tsx'],bundle: true,write: false,platform: 'browser',format: 'esm',splitting: true,outdir: root+'/unit-graph',metafile: true,logLevel: 'silent' });
  assert(!Object.keys(unit.metafile!.inputs).includes('src/lib/learning-interval-transcendentals.ts'));
  writeFileSync(`${root}/unit-dependency-graph.json`,JSON.stringify(unit.metafile,null,2)+'\n');
});
