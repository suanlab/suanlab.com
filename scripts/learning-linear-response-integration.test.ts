import test from 'node:test'; import assert from 'node:assert/strict'; import { readFileSync } from 'node:fs'; import { createHash } from 'node:crypto'; import { resolve, dirname } from 'node:path'; import ts from 'typescript';
import { learningUnits, learningGuides } from '../src/data/interactive-book'; import { curriculumDomains } from '../src/data/learning-curriculum';
import { getLearningContent } from '../src/lib/learning-content';
import { lrKinds, lrPresets, lrPresetInput, lrFields, editLrDraft } from '../src/lib/learning-linear-response';
import { lrConceptSteps, lrConceptQuestions, lrConceptInputs, lrGuides } from '../src/lib/learning-linear-response-concepts';
import { isLrInput, lrInputKey, isLrSuccessForInput, parseLrStoredInput, type LrInput, type LrSuccess } from '../src/lib/learning-linear-response-protocol';
import { evaluateLrRequest as evaluate } from '../src/lib/learning-linear-response-evaluator';
import { LrCalculationController, type LrWorkerPort } from '../src/lib/learning-linear-response-controller';
import React from 'react'; import { renderToStaticMarkup } from 'react-dom/server';
const kind = lrKinds[0], hash = (v: unknown) => createHash('sha256').update(JSON.stringify(v)).digest('hex');
const baseline = JSON.parse(readFileSync('scripts/fixtures/pre-linear-response-registry-hashes.json', 'utf8'));
const cases = JSON.parse(readFileSync('scripts/fixtures/linear-response-integrated-cases.json', 'utf8')) as { label: string; input: LrInput; resultSha256: string }[];

test('linear-response milestone preserves its542-unit/1626-guide prefix and all541 previous values after later additions', () => {
  assert(learningUnits.length >= 542); assert.equal(learningUnits.slice(0, 542).reduce((n, u) => n + learningGuides[u.id].experiments.length, 0), 1626);
  assert.deepEqual(learningUnits.slice(0, 541).map(u => u.id), baseline.units.map((u: { id: string }) => u.id)); assert.equal(learningUnits[541].id, kind);
  assert.deepEqual(Object.keys(learningGuides).slice(0, 541), baseline.guideOrder); assert.equal(Object.keys(learningGuides)[541], kind);
  for (let i = 0; i < 541; i++) { assert.equal(hash(learningUnits[i]), baseline.units[i].hash); assert.equal(hash(learningGuides[learningUnits[i].id]), baseline.units[i].guideHash); }
  for (const d of curriculumDomains) { const old = baseline.domains.find((x: { id: string }) => x.id === d.id); const expectedIds = d.id === 'nonequilibrium' ? [...old.unitIds, kind] : old.unitIds; assert.deepEqual(d.unitIds.slice(0, expectedIds.length), expectedIds); assert.deepEqual({ ...d, unitIds: expectedIds }, { ...old, unitIds: expectedIds }); }
});
for (const [i, c] of cases.entries()) test(`worker payload retains prepared numerical model: ${c.label}`, () => {
  const input = lrPresetInput(kind, i); assert.deepEqual(input, c.input); assert(isLrInput(input)); assert(Object.isFrozen(input.parameters));
  const reply = evaluate({ version: 1, requestId: 7, input }); assert(reply.ok); assert(isLrSuccessForInput(reply, input, 7)); assert.equal(reply.inputKey, lrInputKey(input)); assert.equal(hash(reply.result), c.resultSha256);
});
test('independent concepts, all24 guide presets and registered quizzes match their contracts', () => {
  assert.equal(lrPresets[kind].length, 24); assert.equal(lrConceptSteps[kind].length, 6);
  for (let i = 0; i < 6; i++) { const samples = lrConceptInputs(kind, i); assert.equal(samples.length, 4); assert(samples.every(isLrInput)); assert.notEqual(samples[0], lrConceptSteps[kind][i].samples[0]); assert.equal(lrConceptSteps[kind][i].check.options.length, 3); }
  assert.deepEqual([...lrGuides[kind].flatMap(g => g.presetIndices)].sort((a, b) => a - b), Array.from({ length: 24 }, (_, i) => i));
  assert.deepEqual(learningUnits.find(u => u.id === kind)!.questions, lrConceptQuestions[kind]);
  assert.deepEqual(learningGuides[kind].steps, lrConceptSteps[kind].map(({ title, explanation, formula, diagram }) => ({ title, explanation, formula, diagram })));
});
for (const lang of ['ko', 'en'] as const) test(`graduate reader manuscript renders in ${lang}`, async () => {
  const content = await getLearningContent(kind); assert.equal(content.outline[lang].length, 14); assert(!content[lang].includes('katex-error'));
  const source = readFileSync(`content/learning/physics-math-for-ai/${kind}.${lang}.md`, 'utf8'); assert.equal((source.match(/^\*\*\d\./gm) || []).length, 6); assert.equal((source.match(/\$\$[\s\S]*?\$\$/g) || []).length, 11);
  const independent = JSON.parse(readFileSync('scripts/fixtures/linear-response-exact-preparation.json', 'utf8')).cases.find((c: { input: { a: number; b: number; c: number; d: number; e: number; f: number; kineticNumerator: number } }) => c.input.a === 2 && c.input.b === 3 && c.input.c === 4 && c.input.d === 1 && c.input.e === 1 && c.input.f === 1 && c.input.kineticNumerator === 0);
  const fractions = (independent.stationary as string[]).map(q => { const [n, d = '1'] = q.split('/'); return [BigInt(n), BigInt(d)]; });
  const gcd = (a: bigint, b: bigint): bigint => { while (b) [a, b] = [b, a % b]; return a; };
  const denominator = fractions.reduce((d, q) => d * q[1] / gcd(d, q[1]), BigInt(1));
  assert(source.includes(`π=(${fractions.map(([n, d]) => String(n * denominator / d)).join(',')})/${denominator}`));
  assert(source.includes(`πo=${independent.stationary[1]}`));
});
test('strict scalar edits and lexical storage never replace a valid preparation with rounded invalid input', () => {
  const input = lrPresetInput(kind, 0); assert.equal(lrFields(input).length, 16);
  for (const raw of ['1.0000000000000000001', '1e-999', 'Infinity', '1.5', '']) assert.equal(editLrDraft(input, 'a', raw), null);
  assert.equal(editLrDraft(input, 'a', '0'), null); assert.equal(editLrDraft(input, 'sampleSteps', '65'), null); assert.equal(editLrDraft(input, 'amplitudeNumerator', '11'), null);
  assert.equal(editLrDraft(input, 'amplitudeNumerator', '-1')?.parameters.amplitudeNumerator, -1);
  const raw = JSON.stringify({ version: 1, value: input }); assert.deepEqual(parseLrStoredInput(raw, kind), input);
  assert.equal(parseLrStoredInput(raw.replace('"a":1', '"a":1.00000000000000001'), kind), null); assert.equal(parseLrStoredInput(raw.replace('"timeNumerator":4', '"timeNumerator":1e-999'), kind), null);
  assert.equal(parseLrStoredInput(raw.replace(kind, 'thermal-ou-fluctuation-response'), kind), null);
});
test('identity, order, time, phase, structural zero and availability corruption are rejected', () => {
  const input = lrPresetInput(kind, 0), reply = evaluate({ version: 1, requestId: 1, input }); assert(reply.ok);
  const corrupt = (edit: (x: LrSuccess) => void) => { const next = structuredClone(reply); edit(next); assert.equal(isLrSuccessForInput(next, input, 1), false); };
  corrupt(x => x.requestId++); corrupt(x => x.inputKey += ' '); corrupt(x => x.result.history[1].step++);
  corrupt(x => x.result.history[1].time.bounds!.lower.numerator.coefficient += '1'); corrupt(x => x.result.reference.observables[1].frequencySusceptibility.phaseDefined = false);
  corrupt(x => x.result.history[0].actualOccupancy = null); corrupt(x => x.result.history[1].actualMinusLinearPrediction = null); corrupt(x => x.result.classification.sourceExactlyZero = true);
  corrupt(x => x.result.classification.finiteInputRatesEnclosed = false); corrupt(x => x.result.current = x.result.history[0]);
  assert.equal(evaluate({ version: 2, requestId: 1, input }).ok, false);
});
class Fake implements LrWorkerPort {
  onmessage: Worker['onmessage'] = null; onerror: Worker['onerror'] = null; onmessageerror: Worker['onmessageerror'] = null; sent: unknown; terminated = 0;
  postMessage(v: unknown) { this.sent = v; } terminate() { this.terminated++; }
  send(data: unknown) { this.onmessage?.call(this as unknown as Worker, { data } as MessageEvent); }
}
test('ACK, replacement, obsolete reply, cancel, malformed output, retry, asset error and dispose own worker lifetimes', () => {
  const ports: Fake[] = [], controller = new LrCalculationController(() => { const p = new Fake(); ports.push(p); return p; }, () => {}), a = lrPresetInput(kind, 0), b = lrPresetInput(kind, 23);
  controller.run(a); const obsolete = ports[0].onmessage; ports[0].send({ version: 1, requestId: 1, phase: 'started', kind }); assert.equal(controller.getSnapshot().status, 'pending');
  controller.run(b); assert.equal(ports[0].terminated, 1); obsolete?.call({} as Worker, { data: evaluate(ports[0].sent) } as MessageEvent); assert.equal(controller.getSnapshot().status, 'pending');
  ports[1].send(evaluate(ports[1].sent)); assert.equal(controller.getSnapshot().status, 'ready'); assert.equal(ports[1].terminated, 1);
  controller.run(a); controller.cancel(); assert.equal(ports[2].terminated, 1); controller.run(a); ports[3].send({ version: 1, requestId: 4, ok: true }); assert.equal(controller.getSnapshot().status, 'error');
  controller.run(a); ports[4].send(evaluate(ports[4].sent)); assert.equal(controller.getSnapshot().status, 'ready');
  controller.run(a); ports[5].onerror?.call({} as Worker, { preventDefault() {} } as ErrorEvent); assert.equal(controller.getSnapshot().status, 'error'); assert.equal(ports[5].terminated, 1);
  controller.run(b); controller.dispose(); assert.equal(ports[6].terminated, 1); assert.equal(controller.run(a), null);
  const unavailable = new LrCalculationController(() => { throw Error('asset unavailable'); }, () => {}); unavailable.run(a); assert.equal(unavailable.getSnapshot().status, 'error'); unavailable.dispose();
});
test('reader dependency graph excludes both numerical models and the evaluator', () => {
  const seen = new Set<string>(), visit = (path: string) => { if (seen.has(path)) return; seen.add(path); const ast = ts.createSourceFile(path, readFileSync(path, 'utf8'), ts.ScriptTarget.Latest, true); for (const statement of ast.statements) { if (!ts.isImportDeclaration(statement) || !ts.isStringLiteral(statement.moduleSpecifier) || statement.importClause?.isTypeOnly) continue; const c = statement.importClause; if (c && !c.name && c.namedBindings && ts.isNamedImports(c.namedBindings) && c.namedBindings.elements.every(e => e.isTypeOnly)) continue; const spec = statement.moduleSpecifier.text; if (!spec.startsWith('.') && !spec.startsWith('@/')) continue; const base = spec.startsWith('@/') ? resolve('src', spec.slice(2)) : resolve(dirname(path), spec); for (const suffix of ['.ts', '.tsx', '/index.ts', '/index.tsx']) { try { readFileSync(base + suffix); visit(base + suffix); break; } catch {} } } };
  visit(resolve('src/components/learning/linear-response-lab.tsx')); visit(resolve('src/components/learning/linear-response-concept.tsx'));
  for (const file of ['learning-finite-state-linear-response', 'learning-finite-state-response-experiment', 'learning-linear-response-evaluator']) assert(!seen.has(resolve(`src/lib/${file}.ts`)));
  assert(seen.has(resolve('src/lib/learning-linear-response-worker-factory.ts')));
});
test('fixed and free results have distinct region names and render without starting a worker', async () => {
  (globalThis as typeof globalThis & { React?: typeof React }).React = React;
  const { LrResultView } = await import('../src/components/learning/linear-response-view');
  const input = lrPresetInput(kind, 0), reply = evaluate({ version: 1, requestId: 1, input }); assert(reply.ok);
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'Worker'); let accesses = 0;
  Object.defineProperty(globalThis, 'Worker', { configurable: true, get() { accesses++; throw Error('Worker accessed during SSR'); } });
  try {
    for (const language of ['ko', 'en'] as const) {
      const labels = language === 'ko' ? ['자유 실습', '개념1 · 고정 준비1'] : ['Free experiment', 'Concept1 · Fixed preparation1'];
      const html = renderToStaticMarkup(React.createElement(React.Fragment, null, ...labels.map(contextLabel => React.createElement(LrResultView, { key: contextLabel, response: reply, language, contextLabel }))));
      const names = [...html.matchAll(/role="region"[^>]*aria-label="([^"]+)"/g)].map(m => m[1]);
      assert(names.length >= 16); assert.equal(new Set(names).size, names.length); assert(names.every(name => labels.some(label => name.endsWith(label))));
    }
    const unavailable = evaluate({ version: 1, requestId: 2, input: lrPresetInput(kind, 16) }); assert(unavailable.ok);
    const missing = renderToStaticMarkup(React.createElement(LrResultView, { response: unavailable, language: 'en', contextLabel: 'Free experiment' }));
    assert(missing.includes('Not computed; positive-rate upper bound unresolved')); assert(!missing.includes('Unavailable by definition')); assert(!missing.includes('>Undefined<'));
    const zeroResponse = evaluate({ version: 1, requestId: 3, input: lrPresetInput(kind, 11) }); assert(zeroResponse.ok);
    const zeroHtml = renderToStaticMarkup(React.createElement(LrResultView, { response: zeroResponse, language: 'en', contextLabel: 'Free experiment' })); assert(zeroHtml.includes('Phase undefined; exact zero response'));
    assert.equal(accesses, 0);
  } finally { if (descriptor) Object.defineProperty(globalThis, 'Worker', descriptor); else delete (globalThis as { Worker?: unknown }).Worker; }
});
