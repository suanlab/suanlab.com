import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { entropyKind, entropyInputKey, isEntropyInput, isEntropySuccessForInput, snapshotEntropyInput, parseEntropyStoredInput, type EntropyInput, type EntropySuccess } from '../src/lib/learning-entropy-production-protocol';
import { evaluateEntropyRequest } from '../src/lib/learning-entropy-production-evaluator';
const cases = (JSON.parse(readFileSync('scripts/fixtures/entropy-production-preparation.json', 'utf8')) as { cases: { label: string; input: EntropyInput['parameters'] }[] }).cases;
const base: EntropyInput = { kind: entropyKind, parameters: { a: 2, b: 3, c: 4, d: 1, e: 1, f: 1, rateDenominator: 1, timeNumerator: 4, timeDenominator: 1, sampleSteps: 4, initialState: 0, pathCode: 0, jumpCount: 3, tiltNumerator: 1, tiltDenominator: 4 } };
for (const c of cases) test(`entropy message round-trip: ${c.label}`, () => {
  const input: EntropyInput = { kind: entropyKind, parameters: c.input };
  const response = evaluateEntropyRequest({ version: 1, requestId: 7, input });
  assert(response.ok); assert(isEntropySuccessForInput(structuredClone(response), input, 7));
  assert.equal(isEntropySuccessForInput(response, input, 8), false);
  assert.equal(entropyInputKey(parseEntropyStoredInput(JSON.stringify({ version: 1, value: input }))!), entropyInputKey(input));
});
test('reject stale preparation, missing certificates and malformed nested response data', () => {
  const response = evaluateEntropyRequest({ version: 1, requestId: 3, input: base }); assert(response.ok);
  const changed = { ...base, parameters: { ...base.parameters, pathCode: 7 } };
  assert.equal(isEntropySuccessForInput(response, changed, 3), false);
  assert.equal(isEntropySuccessForInput(response, base, NaN), false);
  type Mutable<T> = T extends ReadonlyArray<infer U> ? Mutable<U>[] : T extends object ? { -readonly [K in keyof T]: Mutable<T[K]> } : T;
  const mutations: ((x: Mutable<EntropySuccess>) => void)[] = [
    x => { x.inputKey = 'other'; }, x => { x.input.parameters.a++; }, x => { x.result.parameters.a++; },
    x => { x.result.history.pop(); }, x => { x.result.history.reverse(); },
    x => { x.result.current.step = 0; }, x => { x.result.stationaryLaw[0].value = Infinity; },
    x => { delete (x.result.entropyRate as unknown as Record<string, unknown>).bounds; },
    x => { x.result.entropyRate.bounds!.lower.denominator.coefficient = '0'; },
    x => { x.result.current.moment = null; }, x => { x.result.current.moment!.value = 0; },
    x => { x.result.current.momentProof.retainedTerms = 0; },
    x => { x.result.current.expectedLogMomentSign = 1; },
    x => { x.result.classification.sampledPaths = true; },
    x => { x.result.path.states[1] = 2; }, x => { x.result.path.jumps[0].index = 1; },
    x => { x.result.path.jumps[0].clockwise = false; }, x => { x.result.edges[0].to = 2; },
    x => { (x.result as unknown as Record<string, unknown>).extra = 1; },
    x => { Object.defineProperty(x.result, 'hidden', { value: 1 }); },
    x => { Object.defineProperty(x.result.history, 'hidden', { value: 1 }); },
  ];
  for (const change of mutations) { const copy: Mutable<EntropySuccess> = structuredClone(response); change(copy); assert.equal(isEntropySuccessForInput(copy, base, 3), false, change.toString()); }
});
test('worker evaluator separates bad envelope and bad input and owns canonical input', () => {
  for (const value of [null, {}, { version: 2, requestId: 1, input: base }, { version: 1, requestId: 0, input: base }, { version: 1, requestId: 1, input: base, extra: true }]) {
    const response = evaluateEntropyRequest(value); assert(!response.ok); assert.equal(response.error, 'invalid-request');
  }
  for (const input of [{ ...base, kind: 'unknown' }, { ...base, parameters: { ...base.parameters, a: 0 } }, { ...base, parameters: { ...base.parameters, sampleSteps: 1.5 } }]) {
    const response = evaluateEntropyRequest({ version: 1, requestId: 9, input }); assert(!response.ok); assert.equal(response.error, 'invalid-input'); assert.equal(response.requestId, 9);
  }
  const input = structuredClone(base), response = evaluateEntropyRequest({ version: 1, requestId: 9, input }); assert(response.ok);
  input.parameters.a++; assert.equal(response.input.parameters.a, base.parameters.a); assert(Object.isFrozen(response.input.parameters));
  const reordered = { parameters: Object.fromEntries(Object.entries(base.parameters).reverse()), kind: entropyKind } as EntropyInput;
  assert.equal(entropyInputKey(reordered), entropyInputKey(base)); assert(Object.isFrozen(snapshotEntropyInput(base)));
  assert.equal(isEntropyInput({ ...base, [Symbol('hidden')]: 1 }), false);
});
test('stored integer lexemes are checked before JSON rounding', () => {
  const raw = JSON.stringify({ version: 1, value: base });
  for (const bad of [raw.replace('"a":2', '"a":2.00000000000000001'), raw.replace('"initialState":0', '"initialState":1e-999'), raw.replace('"a":2', '"a":1e999'), raw.replace('"version":1', '"version":1.00000000000000001'), raw.replace('"pathCode":0', '"pathCode":0.000000000000000000000001'), 'null', 'x'.repeat(32769)]) assert.equal(parseEntropyStoredInput(bad), null);
  assert(isEntropyInput(parseEntropyStoredInput(raw)));
});
test('actual worker handler emits started only for accepted input', async () => {
  const original = Object.getOwnPropertyDescriptor(globalThis, 'self'), messages: unknown[] = [];
  const scope: { onmessage: ((event: { data: unknown }) => void) | null; postMessage(value: unknown): void } = { onmessage: null, postMessage: value => messages.push(structuredClone(value)) };
  Object.defineProperty(globalThis, 'self', { configurable: true, value: scope });
  try {
    await import('../src/lib/learning-entropy-production.worker'); assert(scope.onmessage);
    scope.onmessage({ data: { version: 1, requestId: 13, input: base } });
    assert.equal(messages.length, 2); assert.deepEqual(messages[0], { version: 1, requestId: 13, phase: 'started', kind: entropyKind }); assert(isEntropySuccessForInput(messages[1], base, 13));
    messages.length = 0; scope.onmessage({ data: { version: 1, requestId: 14, input: null } });
    assert.deepEqual(messages, [{ version: 1, requestId: 14, ok: false, error: 'invalid-input' }]);
  } finally { if (original) Object.defineProperty(globalThis, 'self', original); else Reflect.deleteProperty(globalThis, 'self'); }
});
