import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { learningUnits, learningGuides } from '../src/data/interactive-book';
import { curriculumDomains } from '../src/data/learning-curriculum';
import { getLearningContent } from '../src/lib/learning-content';
import { entropyKinds, entropyPresets, entropyPresetInput, entropyFields, editEntropyDraft } from '../src/lib/learning-entropy-production';
import { entropyConceptSteps, entropyConceptInputs, entropyGuides } from '../src/lib/learning-entropy-production-concepts';
import { evaluateEntropyRequest as evaluate } from '../src/lib/learning-entropy-production-evaluator';
import { isEntropySuccessForInput, type EntropySuccess } from '../src/lib/learning-entropy-production-protocol';
import { EntropyCalculationController, type EntropyWorkerPort } from '../src/lib/learning-entropy-production-controller';
import type { PartitionObservation as Q, PartitionRational } from '../src/lib/learning-partition-exact-core';
const kind = entropyKinds[0], hash = (value: unknown) => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const baseline = JSON.parse(readFileSync('scripts/fixtures/pre-entropy-production-registry-hashes.json', 'utf8'));
const references = JSON.parse(readFileSync('scripts/fixtures/entropy-production-integrated-reference.json', 'utf8')) as { cases: { label: string; stationary: string[]; current: string; affinity: string; entropyRate: string; pathTotal: string; rows: { time: string; meanEntropy: string; logMoment: string }[] }[] };
type Fraction = { n: bigint; d: bigint };
const fraction = (n: bigint, d = BigInt(1)): Fraction => ({ n, d });
const lessEqual = (a: Fraction, b: Fraction) => a.n * b.d <= b.n * a.d;
function referenceFraction(value: string): Fraction {
  if (value.includes('/')) { const [n, d] = value.split('/'); return fraction(BigInt(n), BigInt(d)); }
  const m = /^(-?)(\d+)(?:\.(\d*))?(?:e([+-]?\d+))?$/i.exec(value); assert(m);
  const n = BigInt(m[1] + m[2] + (m[3] || '')), power = Number(m[4] || 0) - (m[3] || '').length;
  return power >= 0 ? fraction(n * BigInt(10) ** BigInt(power)) : fraction(n, BigInt(10) ** BigInt(-power));
}
function endpoint(q: PartitionRational): Fraction {
  const n = BigInt(q.numerator.coefficient), d = BigInt(q.denominator.coefficient), power = q.numerator.exponent - q.denominator.exponent;
  return power >= 0 ? fraction(n << BigInt(power), d) : fraction(n, d << BigInt(-power));
}
function encloses(q: Q, value: string) {
  assert(q.bounds); const lo = endpoint(q.bounds.lower), hi = endpoint(q.bounds.upper), r = referenceFraction(value);
  const magnitude = r.n < BigInt(0) ? -r.n : r.n, uncertaintyNumerator = magnitude > r.d ? magnitude : r.d, uncertaintyDenominator = r.d * BigInt(10) ** BigInt(160);
  // Only the uncertainty of the independent stored180-digit reference.
  const above = fraction(r.n * uncertaintyDenominator + uncertaintyNumerator * r.d, r.d * uncertaintyDenominator), below = fraction(r.n * uncertaintyDenominator - uncertaintyNumerator * r.d, r.d * uncertaintyDenominator);
  assert(lessEqual(lo, hi) && lessEqual(lo, above) && lessEqual(below, hi), value);
}
test('all542 preceding units and guides and the543rd entropy unit preserve values and order as later units append', () => {
  assert.equal(learningUnits.slice(0, 543).length, 543); assert.equal(Object.keys(learningGuides).slice(0, 543).length, 543);
  assert.equal(learningUnits.slice(0, 543).reduce((n, u) => n + learningGuides[u.id].experiments.length, 0), 1629);
  assert.deepEqual(learningUnits.slice(0, 542).map(u => u.id), baseline.units.map((u: { id: string }) => u.id));
  assert.deepEqual(Object.keys(learningGuides).slice(0, 542), baseline.guideOrder); assert.equal(learningUnits[542].id, kind); assert.equal(Object.keys(learningGuides)[542], kind);
  for (let i = 0; i < 542; i++) { assert.equal(hash(learningUnits[i]), baseline.units[i].hash); assert.equal(hash(learningGuides[learningUnits[i].id]), baseline.units[i].guideHash); }
  for (const d of curriculumDomains) { const before = baseline.domains.find((x: { id: string }) => x.id === d.id); const historicalIds = new Set([...before.unitIds, kind]); assert.deepEqual({ ...d, unitIds: d.unitIds.filter(id => historicalIds.has(id)) }, { ...before, unitIds: d.id === 'nonequilibrium' ? [...before.unitIds, kind] : before.unitIds }); }
});
for (const [i, p] of entropyPresets[kind].entries()) test(`owned free preparation ${i}: ${p.label.en}`, () => {
  const input = entropyPresetInput(kind, i); assert(Object.isFrozen(input.parameters)); assert.notEqual(input, p.input);
  const reply = evaluate({ version: 1, requestId: 7, input }); assert(reply.ok); assert(isEntropySuccessForInput(reply, input, 7));
  const reference = references.cases.find(c => c.label === p.label.en)!; assert(reference);
  reply.result.stationaryLaw.forEach((q, i) => encloses(q, reference.stationary[i]));
  encloses(reply.result.stationaryCurrent, reference.current); encloses(reply.result.cycleAffinity, reference.affinity); encloses(reply.result.entropyRate, reference.entropyRate); encloses(reply.result.path.totalEntropy, reference.pathTotal);
  reply.result.history.forEach((row, i) => { encloses(row.time, reference.rows[i].time); encloses(row.meanEntropy, reference.rows[i].meanEntropy); encloses(row.logMoment, reference.rows[i].logMoment); });
  assert.equal(entropyFields(input).length, 15);
});
test('six independent concept comparisons, three guides and quizzes stay tied to their teaching content', () => {
  assert.equal(entropyConceptSteps[kind].length, 6);
  for (let i = 0; i < 6; i++) { const samples = entropyConceptInputs(kind, i); assert.equal(samples.length, 4); assert(samples.every(x => Object.isFrozen(x.parameters))); }
  assert.deepEqual([...entropyGuides[kind].flatMap(g => g.presetIndices)].sort((a, b) => a - b), Array.from({ length: 24 }, (_, i) => i));
  assert.deepEqual(learningGuides[kind].steps, entropyConceptSteps[kind].map(({ title, explanation, formula, diagram }) => ({ title, explanation, formula, diagram })));
  assert.deepEqual(learningUnits.find(x => x.id === kind)!.questions.map(q => q.question), [1, 2, 3].map(i => entropyConceptSteps[kind][i].predict));
});
for (const language of ['ko', 'en'] as const) test(`graduate-level entropy manuscript in ${language} renders its14 sections,8 equations and6 solutions`, async () => {
  const html = await getLearningContent(kind), source = readFileSync(`content/learning/physics-math-for-ai/${kind}.${language}.md`, 'utf8');
  assert.equal(html.outline[language].length, 15); assert.equal((source.match(/\$\$[\s\S]*?\$\$/g) || []).length, 17); assert.equal((source.match(/^\*\*\d\./gm) || []).length, 6); assert(!html[language].includes('katex-error'));
  assert(source.includes('π=(17,11,10)/38')); assert(source.includes('J=23/38/s'));
  if (language === 'en') assert(!/[가-힣]/.test(source));
});
test('edits reject lossy integer drafts and incompatible time or path changes without clipping', () => {
  const input = entropyPresetInput(kind, 2);
  for (const raw of ['1.00000000000000001', '1e-999', '1.5', '', 'Infinity', '0']) assert.equal(editEntropyDraft(input, 'a', raw), null);
  assert.equal(editEntropyDraft(input, 'jumpCount', '0')?.parameters.jumpCount, 0);
  assert.equal(editEntropyDraft(input, 'timeNumerator', '0'), null);
  assert.equal(editEntropyDraft(input, 'tiltNumerator', '9'), null);
  assert.equal(editEntropyDraft(entropyPresetInput(kind, 3), 'jumpCount', '1'), null);
  assert.equal(editEntropyDraft(input, 'unknown', '1'), null);
});
class Fake implements EntropyWorkerPort {
  onmessage: Worker['onmessage'] = null; onerror: Worker['onerror'] = null; onmessageerror: Worker['onmessageerror'] = null; sent: unknown; terminated = 0;
  postMessage(value: unknown) { this.sent = value; } terminate() { this.terminated++; }
  send(data: unknown) { this.onmessage?.call(this as unknown as Worker, { data } as MessageEvent); }
}
test('request replacement, obsolete result, malformed reply, asset error, cancellation and disposal own worker lifetimes', () => {
  const ports: Fake[] = [], controller = new EntropyCalculationController(() => { const p = new Fake(); ports.push(p); return p; }, () => {}), input = entropyPresetInput(kind, 2);
  controller.run(input); const obsolete = ports[0].onmessage;
  ports[0].send({ version: 1, requestId: 1, phase: 'started', kind }); assert.equal(controller.getSnapshot().status, 'pending');
  controller.run(entropyPresetInput(kind, 3)); assert.equal(ports[0].terminated, 1); obsolete?.call({} as Worker, { data: evaluate(ports[0].sent) } as MessageEvent); assert.equal(controller.getSnapshot().status, 'pending');
  ports[1].send(evaluate(ports[1].sent)); assert.equal(controller.getSnapshot().status, 'ready');
  controller.run(input); controller.cancel(); assert.equal(ports[2].terminated, 1);
  controller.run(input); ports[3].send({ version: 1, requestId: 4, ok: true }); assert.equal(controller.getSnapshot().status, 'error');
  controller.run(input); ports[4].onerror?.call({} as Worker, { preventDefault() {} } as ErrorEvent); assert.equal(controller.getSnapshot().status, 'error');
  controller.run(input); ports[5].send(evaluate(ports[5].sent)); assert.equal(controller.getSnapshot().status, 'ready');
  controller.run(input); controller.dispose(); assert(ports.every(p => p.terminated === 1)); assert.equal(controller.run(input), null);
});
test('fixed/free results render their certificates with distinct region names and no Worker access during SSR', async () => {
  (globalThis as typeof globalThis & { React?: typeof React }).React = React;
  const { EntropyResultView } = await import('../src/components/learning/entropy-production-view');
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'Worker'); let accesses = 0;
  Object.defineProperty(globalThis, 'Worker', { configurable: true, get() { accesses++; throw Error('Worker accessed during SSR'); } });
  try {
    for (const language of ['ko', 'en'] as const) {
      const reply = evaluate({ version: 1, requestId: 1, input: entropyPresetInput(kind, 2) }); assert(reply.ok);
      const html = renderToStaticMarkup(React.createElement(React.Fragment, null, ...['Free', 'Fixed'].map(contextLabel => React.createElement(EntropyResultView, { key: contextLabel, response: reply, language, contextLabel }))));
      const names = [...html.matchAll(/role="region"[^>]*aria-label="([^"]+)"/g)].map(m => m[1]); assert(names.length >= 20); assert.equal(new Set(names).size, names.length); assert(!/NaN|Infinity/.test(html));
      if (language === 'en') assert(!/[가-힣]/.test(html));
    }
    const reply = evaluate({ version: 1, requestId: 1, input: entropyPresetInput(kind, 18) }); assert(reply.ok);
    const html = renderToStaticMarkup(React.createElement(EntropyResultView, { response: reply as EntropySuccess, language: 'en' })); assert(html.includes('neither zero nor infinity')); assert(!html.includes('Unavailable by definition'));
    assert.equal(accesses, 0);
  } finally { if (descriptor) Object.defineProperty(globalThis, 'Worker', descriptor); else Reflect.deleteProperty(globalThis, 'Worker'); }
});
