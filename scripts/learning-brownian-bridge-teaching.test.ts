import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { build } from 'esbuild';
import { brownianBridgeSteps, brownianBridgeExperiments, brownianBridgeQuestions, brownianBridgeLabels } from '../src/data/brownian-bridge-teaching';
import { brownianBridgeExamples, brownianBridgeDefault } from '../src/lib/learning-brownian-bridge-input';
import { calculateBrownianBridge } from '../src/lib/learning-brownian-bridge';
import { brownianBridgeDraft, parseBrownianBridgeDraft, brownianBridgeMatrixView, brownianBridgePathView, brownianBridgeDefaultDisplay, readBrownianBridgeDisplay, brownianBridgeDefaultLearning, readBrownianBridgeLearning } from '../src/lib/learning-brownian-bridge-view';
import { parseStoredBrownianBridge } from '../src/components/learning/use-saved-brownian-bridge';
test('eight substantive bilingual concepts, six guided comparisons, eight explained questions and16 named inputs agree', () => {
  assert.equal(brownianBridgeSteps.length, 8); assert.equal(brownianBridgeExperiments.length, 6); assert.equal(brownianBridgeQuestions.length, 8);
  assert.equal(brownianBridgeExamples.length, 16);
  for (const e of brownianBridgeExamples) assert(brownianBridgeLabels[e.id].ko && brownianBridgeLabels[e.id].en);
  for (const s of brownianBridgeSteps) { assert(s.body.en.length > 100 && s.body.ko.length > 50); assert(s.diagram.ko && s.diagram.en); }
  for (const g of brownianBridgeExperiments) { assert(g.predict.en && g.explanation.en.length > 100); for (const id of g.examples) assert(brownianBridgeExamples.some(e => e.id === id)); }
  for (const q of brownianBridgeQuestions) { assert(q.correct >= 0 && q.correct < q.options.length); assert(q.explanation.ko && q.explanation.en); }
  assert(!/[가-힣]/.test(JSON.stringify({ steps: brownianBridgeSteps.map(s => [s.title.en, s.body.en, s.diagram.en]), guides: brownianBridgeExperiments.map(g => [g.title.en, g.predict.en, g.explanation.en]), questions: brownianBridgeQuestions.map(q => [q.prompt.en, q.options.map(o => o.en), q.explanation.en]), labels: Object.values(brownianBridgeLabels).map(l => l.en) })));
});
test('every input projects all three complete matrices without truncation or nonfinite coordinates', () => {
  for (const e of brownianBridgeExamples) {
    const result = calculateBrownianBridge(e.input), path = brownianBridgePathView(result); assert(path);
    assert.equal(path.times.length, e.input.intervals + 1); assert(path.sigma.every(v => Number.isFinite(v) && v >= 0));
    for (const kind of ['position', 'increment', 'precision'] as const) for (const scale of ['linear', 'signed-log'] as const) {
      const m = brownianBridgeMatrixView(result, { ...brownianBridgeDefaultDisplay, kind, scale }); assert(m.projectionAvailable);
      assert.equal(m.points.length, m.n * m.n); assert(m.points.every(p => Number.isFinite(p.x) && Number.isFinite(p.y) && Number.isFinite(p.z)));
      assert.equal(m.n, e.input.intervals + (kind === 'position' ? 1 : kind === 'precision' ? -1 : 0));
      if (m.n === 1) assert.deepEqual(m.points.map(p => [p.x, p.z]), [[0, 0]]);
    }
  }
});
test('drawing limits hide unavailable numbers while fixed endpoints remain exact zero', () => {
  const r = calculateBrownianBridge(brownianBridgeDefault); assert.equal(brownianBridgePathView(r)!.sigma[0], 0);
  const changed = structuredClone(r); changed.covariance[2][2] = '1/' + '1' + '0'.repeat(400);
  assert.equal(brownianBridgePathView(changed), null); assert.equal(brownianBridgeMatrixView(changed, brownianBridgeDefaultDisplay).projectionAvailable, false);
});
test('draft parsing retains rational syntax and rejects integer rounding, silent clamping and source-index overflow', () => {
  for (const e of brownianBridgeExamples) assert.deepEqual(parseBrownianBridgeDraft(brownianBridgeDraft(e.input)), e.input);
  const longest = { ...brownianBridgeDefault, start: '-999999999983/999999999989', source: '-999999999983/999999999989' };
  assert.deepEqual(parseBrownianBridgeDraft(brownianBridgeDraft(longest)), longest);
  for (const patch of [{ intervals: '4.0' }, { intervals: '4.0000000000000001' }, { intervals: '33' }, { intervals: '04' }, { probe: '1e0' }, { probe: '5' }, { time: '0.5' }, { diffusion: '0' }, { source: '17' }]) assert.equal(parseBrownianBridgeDraft({ ...brownianBridgeDraft(brownianBridgeDefault), ...patch }), null);
});
test('owned display and versioned storage preserve selection without accepting unknown fields or out-of-range state', () => {
  const display = { kind: 'precision' as const, scale: 'signed-log' as const, selectedTime: 4, selectedCell: 8 }, value = { version: 1, input: brownianBridgeDefault, display, learning: brownianBridgeDefaultLearning };
  assert.deepEqual(parseStoredBrownianBridge(JSON.stringify(value)), { input: brownianBridgeDefault, display, learning: brownianBridgeDefaultLearning });
  for (const patch of [{ selectedTime: 5 }, { selectedCell: 9 }, { selectedCell: -1 }, { selectedTime: 1.5 }, { kind: 'other' }, { scale: 'log' }, { unknown: true }]) assert.equal(parseStoredBrownianBridge(JSON.stringify({ ...value, display: { ...display, ...patch } })), null);
  for (const raw of ['{', '{}', JSON.stringify({ ...value, version: 2 }), JSON.stringify({ ...value, input: { ...brownianBridgeDefault, diffusion: '0' } }), ' '.repeat(4097)]) assert.equal(parseStoredBrownianBridge(raw), null);
  let reads = 0; const getter = { ...display }; Object.defineProperty(getter, 'kind', { get() { reads++; return 'position'; } });
  assert.equal(readBrownianBridgeDisplay(getter, brownianBridgeDefault), null); assert.equal(reads, 0);
});
test('reference figures and concept/cover render in both languages with no Worker access', async () => {
  const require = createRequire(`${process.cwd()}/package.json`); require.extensions['.css'] = () => {};
  const React = await import('react'), { renderToStaticMarkup } = await import('react-dom/server');
  const savedReact = Object.getOwnPropertyDescriptor(globalThis, 'React'), savedWorker = Object.getOwnPropertyDescriptor(globalThis, 'Worker');
  Object.defineProperty(globalThis, 'React', { configurable: true, value: React }); let reads = 0;
  Object.defineProperty(globalThis, 'Worker', { configurable: true, get() { reads++; throw Error('SSR Worker'); } });
  try {
    const { default: Graphic } = await import('../src/components/learning/brownian-bridge-graphic'), { default: Concept } = await import('../src/components/learning/brownian-bridge-concept'), { default: Cover } = await import('../src/components/learning/brownian-bridge-cover');
    const { LanguageProvider } = require('./src/components/language-provider') as typeof import('../src/components/language-provider');
    for (const language of ['ko', 'en'] as const) {
      for (let step = 0; step < 8; step++) { const html = renderToStaticMarkup(React.createElement(Graphic, { step, language })); assert(html.includes('<title') && html.includes('<desc') && html.includes(`data-brownian-reference="${step}"`)); assert(!/NaN|Infinity/.test(html)); if (language === 'en') assert(!/[가-힣]/.test(html)); }
      assert(renderToStaticMarkup(React.createElement(Concept, { language })).includes('href="#experiment"'));
    }
    const cover = renderToStaticMarkup(React.createElement(LanguageProvider, null, React.createElement(Cover))); assert(cover.includes('min-width:0')); assert.equal(reads, 0);
  } finally { if (savedReact) Object.defineProperty(globalThis, 'React', savedReact); else Reflect.deleteProperty(globalThis, 'React'); if (savedWorker) Object.defineProperty(globalThis, 'Worker', savedWorker); else Reflect.deleteProperty(globalThis, 'Worker'); }
});
test('the complete learning UI excludes producer and independent mathematical engines', async () => {
  const bundle = await build({ entryPoints: ['src/components/learning/brownian-bridge-lab.tsx'], bundle: true, write: false, platform: 'browser', format: 'esm', metafile: true, jsx: 'automatic' });
  const inputs = Object.keys(bundle.metafile!.inputs);
  for (const p of ['learning-brownian-bridge.ts', 'learning-brownian-bridge-validation.ts', 'learning-brownian-bridge-evaluator.ts', 'learning-brownian-bridge-validation-evaluator.ts', 'learning-quantum-hilbert-core.ts']) assert(!inputs.includes('src/lib/' + p));
});

test('learning progress rejects invalid steps, answer indices and sparse/accessor arrays', () => {
  assert.deepEqual(readBrownianBridgeLearning(brownianBridgeDefaultLearning), brownianBridgeDefaultLearning);
  for (const patch of [{step:8},{experiment:6},{explanationOpen:1},{answers:Array(8).fill(3)},{answers:Array(7).fill(-1)},{answers:new Array(8)}]) assert.equal(readBrownianBridgeLearning({...brownianBridgeDefaultLearning,...patch}),null);
  const answers=Array(8).fill(-1); let reads=0; Object.defineProperty(answers,'0',{get(){reads++;return 0;}});assert.equal(readBrownianBridgeLearning({...brownianBridgeDefaultLearning,answers}),null);assert.equal(reads,0);
});
