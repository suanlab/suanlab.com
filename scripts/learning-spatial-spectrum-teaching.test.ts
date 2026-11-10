import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { spatialSpectrumDefault, spatialSpectrumExamples } from '../src/lib/learning-spatial-spectrum-input';
import { calculateSpatialSpectrum } from '../src/lib/learning-spatial-spectrum';
import { spatialSpectrumView, spatialSpectrumDraft, parseSpatialSpectrumDraft } from '../src/lib/learning-spatial-spectrum-view';
import { spatialSpectrumSteps, spatialSpectrumExperiments, spatialSpectrumQuestions, spatialSpectrumLabels } from '../src/data/spatial-spectrum-teaching';
import { parseStoredSpatialSpectrum } from '../src/components/learning/use-saved-spatial-spectrum';
import Lab from '../src/components/learning/spatial-spectrum-lab';
import Result from '../src/components/learning/spatial-spectrum-result';
import Graphic from '../src/components/learning/spatial-spectrum-graphic';
import fixture from './fixtures/spatial-spectrum-reference.json';
const cases = fixture.cases.map(c => ({ ...c, result: calculateSpatialSpectrum(c.input) }));
test('eight detailed concepts,six prediction guides,eight explained questions and21 labels cover the bilingual scope', () => {
  assert.equal(spatialSpectrumSteps.length, 8); assert.equal(spatialSpectrumExperiments.length, 6); assert.equal(spatialSpectrumQuestions.length, 8); assert.equal(Object.keys(spatialSpectrumLabels).length, 21);
  assert.deepEqual(new Set(spatialSpectrumExperiments.flatMap(g => g.examples)), new Set(spatialSpectrumExamples.map(c => c.id)));
  for (const locale of ['ko', 'en'] as const) { for (const s of spatialSpectrumSteps) assert(s.body[locale].length > 70); for (const q of spatialSpectrumQuestions) { assert(q.options[q.correct][locale]); assert(q.explanation[locale].length > 40); } }
  assert(!/[가-힣]/.test(JSON.stringify([spatialSpectrumSteps, spatialSpectrumExperiments, spatialSpectrumQuestions].map(rows => rows.map(row => JSON.stringify(row, (_k, v) => v && typeof v === 'object' && 'en' in v ? v.en : v))))));
});
test('all30 independently referenced outputs have finite bounded projections for four matrices and two scales', () => {
  for (const c of cases) for (const kind of ['covariance', 'precision', 'fourierReal', 'fourierImaginary'] as const) for (const scale of ['linear', 'signed-log'] as const) for (const selected of [0, c.input.sites ** 2 - 1]) {
    const m = spatialSpectrumView(c.result, kind, scale, selected);
    assert(m.projectionAvailable, c.id); assert.equal(m.selected, selected); assert.equal(m.points.length, c.input.sites ** 2);
    assert.equal(m.points.filter(p => p.selected).length, 1); assert(m.maximum > 0 && Number.isFinite(m.maximum));
    for (const p of m.points) assert(Number.isFinite(p.y) && Math.abs(p.x) <= 3 && Math.abs(p.z) <= 3 && Math.abs(p.y) <= 2.50000001);
    assert(m.powers.every(v => v >= 0)); assert.equal(m.axis, kind.startsWith('fourier') ? 'mode' : 'site');
  }
});
test('finite and infinite curves preserve geometry,origin and exact projected data', () => {
  const ring = cases.find(c => c.id === 'four-site-hand-check')!.result, m = spatialSpectrumView(ring, 'covariance', 'linear');
  assert.deepEqual(m.covariance, [7 / 15, 1 / 5, 2 / 15, 1 / 5]); assert.deepEqual(m.distances, [0, 1, 2, 3]);
  const shifted = cases.find(c => c.id === 'shifted-origin')!.result, normal = cases.find(c => c.id === 'massive-ring')!.result;
  assert.deepEqual(spatialSpectrumView(shifted, 'covariance', 'linear').covariance, spatialSpectrumView(normal, 'covariance', 'linear').covariance);
  const open = spatialSpectrumView(cases.find(c => c.id === 'open-chain')!.result, 'fourierReal', 'linear'); assert.equal(open.image, null); assert(open.values.some((row, i) => row.some((v, j) => i !== j && v !== 0)));
  const massless = spatialSpectrumView(cases.find(c => c.id === 'massless-ring-constrained')!.result, 'covariance', 'linear'); assert.equal(massless.infinite, null); assert(massless.covariance.some(v => v < 0));
});
test('zero-enclosing drawing intervals are explicitly projected to baseline while exact intervals remain available', () => {
  const r = structuredClone(cases[0].result), interval = { lower: '-1/' + String(BigInt(1) << BigInt(512)), upper: '1/' + String(BigInt(1) << BigInt(512)) };
  r.fourierCovariance[1][2].imaginary = interval;
  const m = spatialSpectrumView(r, 'fourierImaginary', 'signed-log', 10); assert.equal(m.values[1][2], 0); assert.equal(m.ranges[1][2], interval);
});
test('drafts preserve raw fraction spellings and reject invalid indices,spacing and nonnormalizable ensembles', () => {
  for (const c of spatialSpectrumExamples) assert.deepEqual(parseSpatialSpectrumDraft(spatialSpectrumDraft(c.input)), c.input);
  const d = spatialSpectrumDraft(spatialSpectrumDefault);
  for (const patch of [{ sites: '2' }, { sites: '17' }, { origin: '8' }, { selectedMode: '1.5' }, { sites: '08' }, { massSquared: '0' }, { spacing: '1/17' }, { temperature: '0' }, { speedSquared: '-1' }]) assert.equal(parseSpatialSpectrumDraft({ ...d, ...patch }), null);
});
test('versioned saved input is bounded and validated before restoring', () => {
  const input = { ...spatialSpectrumDefault, temperature: '2/2' }, parsed = parseStoredSpatialSpectrum(JSON.stringify({ version: 1, value: input })); assert.deepEqual(parsed, input); assert(Object.isFrozen(parsed));
  for (const raw of ['{', ' '.repeat(4097), JSON.stringify({ version: 2, value: input }), JSON.stringify({ version: 1, value: { ...input, sites: 2 } })]) assert.equal(parseStoredSpatialSpectrum(raw), null);
});
test('full/embedded labs,reference diagrams and exact diagnostics server-render in both languages without Workers', async () => {
  const React = await import('react'), { renderToStaticMarkup } = await import('react-dom/server'), old = Object.getOwnPropertyDescriptor(globalThis, 'React'), worker = Object.getOwnPropertyDescriptor(globalThis, 'Worker'); Object.defineProperty(globalThis, 'React', { configurable: true, value: React }); let calls = 0; Object.defineProperty(globalThis, 'Worker', { configurable: true, get() { calls++; throw Error('SSR Worker'); } });
  try { for (const language of ['ko', 'en'] as const) {
    const full = renderToStaticMarkup(React.createElement(Lab, { language })), embedded = renderToStaticMarkup(React.createElement(Lab, { language, embedded: true })); assert(full.includes('information-question')); assert(!embedded.includes('information-question'));
    for (const c of cases) { const html = renderToStaticMarkup(React.createElement(Result, { language, result: c.result })); assert(html.includes(c.result.covariance[0][0])); assert(html.includes('PKC−TP')); if (language === 'en') assert(!/[가-힣]/.test(html)); }
    for (let step = 0; step < 8; step++) { const html = renderToStaticMarkup(React.createElement(Graphic, { step, language })); assert(html.includes('role="img"')); if (language === 'en') assert(!/[가-힣]/.test(html)); }
    if (language === 'en') assert(!/[가-힣]/.test(full));
  } assert.equal(calls, 0); } finally { if (old) Object.defineProperty(globalThis, 'React', old); else Reflect.deleteProperty(globalThis, 'React'); if (worker) Object.defineProperty(globalThis, 'Worker', worker); else Reflect.deleteProperty(globalThis, 'Worker'); }
});
test('out-of-range projections suppress all plots while preserving exact fractions', async () => {
  const React = await import('react'), { renderToStaticMarkup } = await import('react-dom/server'), old = Object.getOwnPropertyDescriptor(globalThis, 'React'); Object.defineProperty(globalThis, 'React', { configurable: true, value: React });
  try { const r = structuredClone(cases[0].result), value = '1' + '0'.repeat(400); r.covariance[0][0] = value; assert(!spatialSpectrumView(r, 'covariance', 'linear').projectionAvailable); const html = renderToStaticMarkup(React.createElement(Result, { result: r, language: 'en' })); assert(html.includes('data-spatial-projection-unavailable')); assert(html.includes(value)); assert(!html.includes('<svg')); } finally { if (old) Object.defineProperty(globalThis, 'React', old); else Reflect.deleteProperty(globalThis, 'React'); }
});
test('display bundle includes Three.js and the controller hook without numerical engines', async () => {
  const bundle = await build({ entryPoints: ['src/components/learning/spatial-spectrum-lab.tsx'], bundle: true, write: false, platform: 'browser', format: 'esm', metafile: true });
  const inputs = Object.keys(bundle.metafile!.inputs); assert(inputs.some(p => p.endsWith('spatial-spectrum-matrix-engine.ts'))); assert(inputs.some(p => p.endsWith('use-spatial-spectrum-calculation.ts')));
  for (const name of ['learning-spatial-spectrum.ts', 'learning-spatial-spectrum-evaluator.ts', 'learning-spatial-spectrum-validation.ts', 'learning-quantum-hilbert-core.ts', 'learning-spatial-spectrum-validation-math.ts']) assert(!inputs.includes('src/lib/' + name));
});
