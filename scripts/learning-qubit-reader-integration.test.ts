import { mkdirSync as ensureTestArtifactDirectory } from 'node:fs';
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { build } from 'esbuild';
import { learningUnits, learningGuides } from '../src/data/interactive-book';
import { curriculumDomains } from '../src/data/learning-curriculum';
import { qubitTeaching } from '../src/data/qubit-teaching';
import { qubitTeachingIds as ids } from '../src/lib/learning-qubit-teaching-input';
import { getLearningContent } from '../src/lib/learning-content';
import before from './fixtures/pre-qubit-registry-hashes.json';
ensureTestArtifactDirectory('.runtime/learning/quantum-qubit-reader-integration', { recursive: true });

const hash = (value: unknown) => createHash('sha256').update(JSON.stringify(value)).digest('hex');
for (const [index, id] of ids.entries()) test(`registered qubit reader: ${id}`, async () => {
  const unit = learningUnits.find(unit => unit.id === id); assert(unit);
  assert.equal(learningUnits.indexOf(unit), 571 + index);
  assert.deepEqual(unit, qubitTeaching[index].unit);
  assert.deepEqual(learningGuides[id], qubitTeaching[index].guide);
  assert.equal(unit.questions.length, index === 3 ? 4 : 3);
  assert.equal(learningGuides[id].experiments.length, index === 3 ? 4 : 3);
  assert(unit.prerequisiteIds.every(prior => learningUnits.some(unit => unit.id === prior)));
  assert(curriculumDomains.find(domain => domain.id === 'quantum-physics')!.unitIds.includes(id));
  const content = await getLearningContent(id);
  for (const language of ['ko', 'en'] as const) {
    assert(content.outline[language].length >= 7);
    assert(content[language].includes('katex') && !content[language].includes('katex-error'));
  }
  assert(!/[가-힣]/.test(content.en));
});
test('all previous571 units/guides keep exact content/order and all51 domains retain scope', () => {
  assert(learningUnits.length >= 575);
  assert.equal(new Set(learningUnits.slice(0,575).map(unit => unit.id)).size, 575);
  assert.equal(learningUnits.slice(0,575).reduce((n, unit) => n + learningGuides[unit.id].experiments.length, 0), 1726);
  assert.deepEqual(learningUnits.slice(0, 571).map(unit => unit.id), before.unitOrder);
  assert.deepEqual(Object.keys(learningGuides).slice(0, 571), before.guideOrder);
  assert.deepEqual(learningUnits.slice(571,575).map(unit => unit.id), [...ids]);
  assert.deepEqual(Object.keys(learningGuides).slice(571,575), [...ids]);
  for (const unit of learningUnits.slice(0, 571)) {
    assert.equal(hash(unit), before.units[unit.id as keyof typeof before.units]);
    assert.equal(hash(learningGuides[unit.id]), before.guides[unit.id as keyof typeof before.guides]);
  }
  assert.equal(curriculumDomains.length, 51);
  for (const domain of curriculumDomains) {
    const historical = before.domains[domain.id as keyof typeof before.domains];
    const restored = { ...domain, unitIds: domain.unitIds.filter(id => (before.unitOrder as string[]).includes(id)) };
    assert.equal(hash(restored), historical.sha256, domain.id);
  }
});
test('reader/concept/cover dispatch and the standard experiment anchor are connected', () => {
  const source = (name: string) => readFileSync('src/components/learning/' + name, 'utf8');
  assert(source('interactive-book.tsx').includes('isQubitTeachingId(unit.id) ? <QubitReaderLab'));
  assert(source('concept-walkthrough.tsx').includes('isQubitTeachingId(kind)) return <QubitConcept'));
  assert(source('concept-walkthrough.tsx').includes('experimentHref="#experiment"'));
  assert(source('book-art.tsx').includes('isQubitTeachingId(kind)) return <QubitCover'));
  assert(source('qubit-reader-lab.tsx').includes('id="experiment"'));
  assert(source('qubit-reader-lab.tsx').includes("dynamic(() => import('./qubit-lab')"));
});
test('registered cover and concept SSR preserve diagrams without touching Worker', async () => {
  createRequire(`${process.cwd()}/package.json`).extensions['.css'] = () => {};
  const React = await import('react'), { renderToStaticMarkup } = await import('react-dom/server');
  const originalReact = Object.getOwnPropertyDescriptor(globalThis, 'React'), originalWorker = Object.getOwnPropertyDescriptor(globalThis, 'Worker');
  Object.defineProperty(globalThis, 'React', { configurable: true, value: React });
  let reads = 0; Object.defineProperty(globalThis, 'Worker', { configurable: true, get() { reads++; throw Error('SSR Worker'); } });
  try {
    const { LanguageProvider } = createRequire(`${process.cwd()}/package.json`)('./src/components/language-provider') as typeof import('../src/components/language-provider');
    const { default: Concept } = await import('../src/components/learning/concept-walkthrough');
    const { default: Art } = await import('../src/components/learning/book-art');
    for (const kind of ids) {
      const concept = renderToStaticMarkup(React.createElement(LanguageProvider, null, React.createElement(Concept, { kind })));
      assert(concept.includes(`data-qubit-concept="${kind}"`)); assert(concept.includes('href="#experiment"'));
      const art = renderToStaticMarkup(React.createElement(LanguageProvider, null, React.createElement(Art, { kind })));
      assert(art.includes(`data-qubit-cover="${kind}"`)); assert(!art.includes('<canvas'));
    }
    assert.equal(reads, 0);
  } finally {
    if (originalReact) Object.defineProperty(globalThis, 'React', originalReact); else Reflect.deleteProperty(globalThis, 'React');
    if (originalWorker) Object.defineProperty(globalThis, 'Worker', originalWorker); else Reflect.deleteProperty(globalThis, 'Worker');
  }
});
test('the complete reader dependency graph excludes qubit numerical and validation engines', async () => {
  const result = await build({ entryPoints: ['src/components/learning/interactive-book.tsx'], bundle: true, write: false, format: 'esm', platform: 'browser', splitting: true, outdir: '.runtime/learning/quantum-qubit-reader-integration/graph', metafile: true, jsx: 'automatic', logLevel: 'silent' });
  const inputs = Object.keys(result.metafile!.inputs);
  for (const file of ['learning-qubit-channel.ts', 'learning-qubit-channel-algebra.ts', 'learning-qubit-scalar.ts', 'learning-qubit-decode.ts', 'learning-qubit-circuit-exact.ts', 'learning-qubit-circuit-report.ts', 'learning-qubit-observations.ts', 'learning-qubit-observations-validator.ts', 'learning-qubit-evaluator.ts', 'learning-qubit-validation-evaluator.ts']) assert(!inputs.includes('src/lib/' + file), file);
  writeFileSync('.runtime/learning/quantum-qubit-reader-integration/reader-main-thread-dependency-graph.json', JSON.stringify({ qubitNumericalEnginesExcluded: true, inputs, dependencyHashes: Object.fromEntries(inputs.map(p => [p, createHash('sha256').update(readFileSync(p)).digest('hex')])), metafile: result.metafile }, null, 2) + '\n');
});
