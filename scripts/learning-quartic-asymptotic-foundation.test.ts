import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { build } from 'esbuild';
import { calculateQuarticAsymptotic } from '../src/lib/learning-quartic-asymptotic';
import { quarticAsymptoticExamples, quarticAsymptoticDefault, readQuarticAsymptoticInput, type QuarticAsymptoticInput } from '../src/lib/learning-quartic-asymptotic-input';
import { Rational as Q } from '../src/lib/learning-graph-gauge-validation-math';
import { getLearningContent } from '../src/lib/learning-content';
import { snapshotQubitWire } from '../src/lib/learning-qubit-wire';
const zero = Q.read('0'), one = Q.read('1'), tolerance = new Q(BigInt(1), BigInt(10) ** BigInt(105));
const le = (a: Q, b: Q) => a.n * b.d <= b.n * a.d;
const abs = (a: Q) => a.n < BigInt(0) ? a.neg() : a;
const decimal = (text: string) => { const [whole, fraction = ''] = text.split('.'); return new Q(BigInt(whole + fraction), BigInt(10) ** BigInt(fraction.length)); };
const cases = JSON.parse(readFileSync('scripts/fixtures/quartic-asymptotic-reference.json', 'utf8')).cases as {
  id: string; input: QuarticAsymptoticInput; coefficients: string[]; referenceDecimal: string;
  bestBoundOrderWithinBudget: number; globallyBestBoundOrder: number;
  approximations: { order: number; coefficient: string; term: string; partial: string; firstOmittedMagnitude: string; position: string }[];
}[];
for (const c of cases) test(`independent Gaussian moments, truncation and numerical integral: ${c.id}`, () => {
  const result = calculateQuarticAsymptotic(c.input);
  assert.deepEqual(result.coefficients, c.coefficients);
  assert.deepEqual(result.approximations.map(row => Object.fromEntries(Object.entries(row).filter(([key]) => !['quadratureDistance', 'errorInterval'].includes(key)))), c.approximations);
  assert.equal(result.bestBoundOrderWithinBudget, c.bestBoundOrderWithinBudget);
  assert.equal(result.globallyBestBoundOrder, c.globallyBestBoundOrder);
  assert.equal(result.budgetReachesBestBound, c.globallyBestBoundOrder <= c.input.order);
  const reference = decimal(c.referenceDecimal), lo = Q.read(result.quadratureInterval.lower), hi = Q.read(result.quadratureInterval.upper);
  assert(le(lo, reference) && le(reference, hi));
  if (c.input.coupling !== '0') assert(le(lo, reference.sub(tolerance)) && le(reference.add(tolerance), hi));
  const combinedLo = Q.read(result.combinedInterval.lower), combinedHi = Q.read(result.combinedInterval.upper);
  assert(le(lo, combinedLo) && le(combinedLo, combinedHi) && le(combinedHi, hi));
  assert(le(combinedLo, reference.add(tolerance)) && le(reference.sub(tolerance), combinedHi));
  assert(le(zero, combinedLo) && le(combinedHi, one));
  for (const row of result.approximations) {
    const p = Q.read(row.partial), omitted = Q.read(row.firstOmittedMagnitude), actual = abs(p.sub(reference));
    assert(le(actual, omitted.add(tolerance)));
    if (row.position === 'above') assert(le(reference.sub(tolerance), p));
    if (row.position === 'below') assert(le(p, reference.add(tolerance)));
    const lower = Q.read(row.errorInterval.lower), upper = Q.read(row.errorInterval.upper);
    assert(le(zero, lower) && le(lower, upper) && le(upper, omitted));
    assert(le(lower, actual.add(tolerance)) && le(actual.sub(tolerance), upper));
  }
  assert.equal(result.quadrature.cells, 256); assert.equal(result.quadrature.precisionBits, 192);
});

test('fourteen frozen examples retain the independent input ledger', () => {
  assert.equal(quarticAsymptoticExamples.length, 14);
  for (const [i, example] of quarticAsymptoticExamples.entries()) { assert.deepEqual(example.input, cases[i].input); assert(Object.isFrozen(example.input)); assert.deepEqual(readQuarticAsymptoticInput(example.input), example.input); }
});

test('default hand calculation distinguishes coefficients, signed terms and certified errors', () => {
  const result = calculateQuarticAsymptotic(quarticAsymptoticDefault);
  assert.deepEqual(result.coefficients.slice(0, 4), ['1', '-3/4', '105/32', '-3465/128']);
  assert.equal(result.approximations[1].partial, '37/40');
  assert.equal(result.approximations[2].partial, '613/640');
  assert.equal(result.approximations[2].firstOmittedMagnitude, '693/25600');
  assert.equal(result.bestBoundOrderWithinBudget, 2); assert.equal(result.globallyBestBoundOrder, 2);
  assert.equal(result.series, 'divergent-asymptotic');
});

test('ties choose the earlier retained order; a limited budget is not called globally optimal', () => {
  const tie = calculateQuarticAsymptotic({ coupling: '8/35', order: 8 });
  assert.equal(tie.bestBoundOrderWithinBudget, 0); assert.equal(tie.globallyBestBoundOrder, 0);
  assert.equal(tie.approximations[0].firstOmittedMagnitude, tie.approximations[1].firstOmittedMagnitude);
  const tiny = calculateQuarticAsymptotic({ coupling: '1/999999999989', order: 32 });
  assert.equal(tiny.bestBoundOrderWithinBudget, 32); assert.equal(tiny.budgetReachesBestBound, false);
  assert(tiny.globallyBestBoundOrder > 100000000000);
});

test('the Gaussian identity is exact and raw coupling spelling remains attributable', () => {
  const gaussian = calculateQuarticAsymptotic({ coupling: '0', order: 32 });
  assert.deepEqual(gaussian.quadratureInterval, { lower: '1', upper: '1', width: '0' });
  assert(gaussian.approximations.every(row => row.position === 'exact' && row.partial === '1' && row.firstOmittedMagnitude === '0'));
  const a = calculateQuarticAsymptotic({ coupling: '1/10', order: 2 }), b = calculateQuarticAsymptotic({ coupling: '2/20', order: 2 });
  assert.deepEqual(a.approximations, b.approximations); assert.deepEqual(a.quadratureInterval, b.quadratureInterval);
  assert.notDeepEqual(a.input, b.input);
});

test('bounded ownership rejects accessors, symbols, invalid couplings and order coercion', () => {
  let called = false; const getter = { ...quarticAsymptoticDefault };
  Object.defineProperty(getter, 'coupling', { get() { called = true; return '1/10'; }, enumerable: true });
  assert.equal(readQuarticAsymptoticInput(getter), null); assert.equal(called, false);
  for (const coupling of ['-1', '0.1', '1e-2', '17', '1/0', '1/0002', '1000000000000', '999999999985/62499999999']) assert.equal(readQuarticAsymptoticInput({ coupling, order: 1 }), null);
  for (const order of [-1, 33, 1.5, '1', NaN, Infinity]) assert.equal(readQuarticAsymptoticInput({ coupling: '1', order }), null);
  assert.equal(readQuarticAsymptoticInput({ ...quarticAsymptoticDefault, extra: true }), null);
  assert.equal(readQuarticAsymptoticInput({ ...quarticAsymptoticDefault, [Symbol('hidden')]: true }), null);
  assert.throws(() => calculateQuarticAsymptotic({ coupling: '-1', order: 2 }), RangeError);
});

test('reader input and example bundle excludes quadrature, fractions and exact coefficient engines', async () => {
  const bundle = await build({ entryPoints: ['src/lib/learning-quartic-asymptotic-input.ts'], bundle: true, write: false, platform: 'browser', metafile: true, logLevel: 'silent' });
  assert.equal(Object.keys(bundle.metafile!.inputs).length, 1); assert(!bundle.outputFiles[0].text.includes('BigInt('));
});

test('certificate objects are owned per result and maximum outputs fit the existing wire budget', () => {
  const first = calculateQuarticAsymptotic({ coupling: '0', order: 0 });
  first.gaussianNormalization.cutoff = 99; first.quadrature.cells = 1;
  const fresh = calculateQuarticAsymptotic({ coupling: '0', order: 0 });
  assert.equal(fresh.gaussianNormalization.cutoff, 16); assert.equal(fresh.quadrature.cells, 256);
  const rows = [];
  for (const coupling of ['0', '16', '1/999999999989', '999999999983/62499999999']) {
    const result = calculateQuarticAsymptotic({ coupling, order: 32 }), owned = snapshotQubitWire(result);
    assert.deepEqual(owned, result); assert(Object.isFrozen(owned));
    rows.push({ coupling, order: 32, bytes: Buffer.byteLength(JSON.stringify(result)), existingWireBudgetUnchanged: true });
  }
  mkdirSync('.runtime/learning/quartic-asymptotic-foundation', { recursive: true });
  writeFileSync('.runtime/learning/quartic-asymptotic-foundation/wire-budget.json', JSON.stringify(rows, null, 2) + '\n');
});

test('bilingual unregistered manuscripts render substantive explanations, worked examples and strict mathematics', async () => {
  const id = 'quartic-integral-asymptotic-truncation', rendered = await getLearningContent(id), root = '.runtime/learning/quartic-asymptotic-foundation';
  mkdirSync(root, { recursive: true }); const rows = [];
  for (const language of ['ko', 'en'] as const) {
    const source = readFileSync(`content/learning/physics-math-for-ai/${id}.${language}.md`, 'utf8');
    assert.equal(rendered.outline[language].length, 18);
    assert.equal((source.match(/^\d\. \*\*/gm) ?? []).length, 8);
    assert(source.length > 8000 && !rendered[language].includes('katex-error') && !/<h1(?:\s|>)/.test(rendered[language]));
    assert(rendered[language].includes('<math')); if (language === 'en') assert(!/[가-힣]/.test(rendered.en));
    writeFileSync(`${root}/manuscript-${language}.html`, rendered[language]);
    rows.push({ language, sections: 18, displayEquations: (source.match(/^\$\$$/gm) ?? []).length / 2, explainedExercises: 8, sourceCharacters: source.length });
  }
  writeFileSync(`${root}/manuscripts.json`, JSON.stringify(rows, null, 2) + '\n');
});
