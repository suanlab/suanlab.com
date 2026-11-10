import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { build } from 'esbuild';
import { createRequire } from 'node:module';
import { mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { calculateBlockOrder } from '../src/lib/learning-block-order';
import { blockOrderDefault, blockOrderExamples, type BlockOrderInput } from '../src/lib/learning-block-order-input';
import { blockOrderDraft, parseBlockOrderDraft, blockOrderSpatial } from '../src/lib/learning-block-order-view';
import { blockOrderLabels, blockOrderSteps, blockOrderExperiments, blockOrderQuestions } from '../src/data/block-order-teaching';

const wire = (input: BlockOrderInput) => { const { profiles,...rest } = calculateBlockOrder(input); return { ...rest,profileChunks: [profiles] }; };
test('form drafts round-trip all examples; invalid activity, geometry and profile edits do not become applied inputs', () => {
  for (const e of blockOrderExamples) assert.deepEqual(parseBlockOrderDraft(blockOrderDraft(e.input)),e.input);
  for (const patch of [{ shape: '1,4' },{ shape: '17' },{ blockShape: '3' },{ disagreementActivity: '0' },{ disagreementActivity: '1e-12' },
    { disagreementActivity: '0.5' },{ blockUpActivities: '1' },{ selectedUpCounts: '1.5,1' },{ selectedUpCounts: '3,0' },{ witnessPair: '1,0' },{ witnessPair: '0,0' }])
    assert.equal(parseBlockOrderDraft({ ...blockOrderDraft(blockOrderDefault),...patch }),null);
});
test('spatial projection preserves actual positions, fixed spin units and the distinction between conditional, block and ensemble means', () => {
  const input = blockOrderExamples.find(e => e.id === 'boundary-versus-mean')!.input, exact = wire(input);
  assert.deepEqual(blockOrderSpatial(input,exact,'conditional',0).points.map(p => p.exact),['1/3','2/3','2/3','1/3']);
  assert.deepEqual(blockOrderSpatial(input,exact,'profile',0).points.map(p => p.exact),['1/2','1/2','1/2','1/2']);
  assert(blockOrderSpatial(input,exact,'ensemble',0).points.every(p => p.exact === '0'));
  const grid = blockOrderExamples.find(e => e.id === 'two-dimensional')!.input, large = wire(grid), model = blockOrderSpatial(grid,large,'configuration',7);
  assert.deepEqual([model.points[0].x,model.points[0].z,model.points[15].x,model.points[15].z],[-3,-3,3,3]);
  assert(model.points.every(p => p.value === 1 || p.value === -1));
  assert.deepEqual(model.points.map(p => p.value),model.points.map((_,i) => ((model.example.mask >> i) & 1) ? 1 : -1));
  assert.equal(model.available,true); assert(!('profiles' in large));
  assert.equal(blockOrderSpatial(input,exact,'configuration',7).exampleIndex,exact.selected.examples.length-1);
});
test('guided claims reproduce weighted sums, field cancellation, locality and full profile counts', () => {
  const byId = (id: string) => calculateBlockOrder(blockOrderExamples.find(e => e.id === id)!.input);
  const base = byId('unequal-hidden-weights'), free = byId('uncoupled-counting'), biased = byId('biased-profiles'), grid = byId('two-dimensional');
  assert.equal(base.profiles[base.selected.profileIndex].probability,'1/9'); assert.equal(free.profiles[free.selected.profileIndex].probability,'1/4');
  assert(free.selected.examples.every(e => e.conditionalProbability === '1/4'));
  assert.equal(base.globalUpProbabilities[2],'7/27');
  assert.deepEqual(base.selected.examples,biased.selected.examples); assert.deepEqual(base.selected.conditionalSpinMeans,biased.selected.conditionalSpinMeans);
  assert.notDeepEqual(base.meanMagnetizations,biased.meanMagnetizations);
  assert.equal(base.selected.meanProductResidualPower,'9/8'); assert.equal(base.selected.meanProductResidualDivisor,'2');
  assert.equal(byId('nonneighbor-closure').localityWitness?.crossRatio,'16/25');
  assert.equal(byId('nonneighbor-closure').localityWitness?.rulesOutNearestNeighborPairFactorization,true);
  assert.equal(byId('periodic-neighbors').localityWitness?.pairIsNeighbor,true);
  assert.equal(byId('periodic-neighbors').localityWitness?.rulesOutNearestNeighborPairFactorization,false);
  assert.equal(grid.profiles.length,625); assert.equal(grid.profiles[grid.selected.profileIndex].multiplicity,1296);
  assert.equal(grid.selected.examples.length,8);
  assert.equal(blockOrderSteps.length,6); assert.equal(blockOrderExperiments.length,6); assert.equal(blockOrderQuestions.length,8);
  for (const g of blockOrderExperiments) for (const id of g.examples) assert(blockOrderExamples.some(e => e.id === id) && blockOrderLabels[id]);
  for (const q of blockOrderQuestions) { assert(q.correct >= 0 && q.correct < 3); assert(q.explanation.en.length > 70); }
  for (const data of [blockOrderLabels,blockOrderSteps,blockOrderExperiments,blockOrderQuestions]) {
    const visit = (v: unknown) => { if (v && typeof v === 'object') for (const [k,value] of Object.entries(v)) { if (k === 'en') assert(!/[가-힣]/.test(String(value))); else visit(value); } };
    visit(data);
  }
});
test('lab SSR avoids Worker, storage and media access and renders bilingual teaching without calculation engines', async () => {
  const folder = '.runtime/learning/block-order-view'; mkdirSync(folder,{ recursive: true });
  const built = await build({ entryPoints: ['src/components/learning/block-order-lab.tsx'],bundle: true,write: false,platform: 'node',format: 'cjs',packages: 'external',loader: { '.css': 'empty' },logLevel: 'silent' });
  const file = path.resolve(folder,'lab-ssr.cjs'); writeFileSync(file,built.outputFiles[0].text);
  const saved = new Map<string,PropertyDescriptor | undefined>(); let calls = 0;
  for (const key of ['Worker','localStorage','matchMedia']) { saved.set(key,Object.getOwnPropertyDescriptor(globalThis,key)); Object.defineProperty(globalThis,key,{ configurable: true,get() { calls++; throw Error('SSR browser access'); } }); }
  try {
    const BlockOrderLab = createRequire(path.resolve('package.json'))(file).default as React.ComponentType<{ language: 'ko' | 'en' }>;
    const en = renderToStaticMarkup(createElement(BlockOrderLab,{ language: 'en' })), ko = renderToStaticMarkup(createElement(BlockOrderLab,{ language: 'ko' }));
    assert(en.includes('data-block-storage-ready="false"') && en.includes('data-block-status="idle"'));
    assert(!/[가-힣]/.test(en)); assert(ko.includes('국소 질서변수')); assert.equal(calls,0);
  } finally { for (const [key,value] of saved) { if (value) Object.defineProperty(globalThis,key,value); else Reflect.deleteProperty(globalThis,key); } }
  const bundle = await build({ entryPoints: ['src/components/learning/block-order-lab.tsx'],bundle: true,write: false,platform: 'browser',format: 'esm',splitting: true,outdir: '.runtime/learning/block-order-view-graph',metafile: true,logLevel: 'silent' });
  for (const file of ['learning-block-order.ts','learning-block-order-evaluator.ts','learning-block-order-observations.ts','learning-block-order-validation.ts',
    'learning-block-order-validation-evaluator.ts','learning-finite-markov-exact.ts','learning-real-time-checker-interval.ts','learning-interval-transcendentals.ts'])
    assert(!Object.keys(bundle.metafile!.inputs).includes(`src/lib/${file}`),file);
});
