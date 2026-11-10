import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { build } from 'esbuild';
import { getLearningContent } from '../src/lib/learning-content';
import { calculateRegularPerturbation } from '../src/lib/learning-regular-perturbation';
import { readRegularPerturbationInput, regularPerturbationExamples, regularPerturbationDefault, type RegularPerturbationInput } from '../src/lib/learning-regular-perturbation-input';

// Independent rational checks: no production fraction/poly helper is imported.
type Q = { n: bigint; d: bigint };
const read = (s: string): Q => { const [n, d = '1'] = s.split('/'); return { n: BigInt(n), d: BigInt(d) }; };
const add = (a: Q, b: Q): Q => ({ n: a.n * b.d + b.n * a.d, d: a.d * b.d });
const neg = (a: Q): Q => ({ n: -a.n, d: a.d });
const sub = (a: Q, b: Q) => add(a, neg(b));
const mul = (a: Q, b: Q): Q => ({ n: a.n * b.n, d: a.d * b.d });
const le = (a: Q, b: Q) => a.n * b.d <= b.n * a.d;
const equal = (a: Q, b: Q) => a.n * b.d === b.n * a.d;
const zero = read('0');
const cube = (a: Q) => mul(mul(a, a), a);
const cases = JSON.parse(readFileSync('scripts/fixtures/regular-perturbation-reference.json', 'utf8')).cases as {
  id: string; input: RegularPerturbationInput; coefficients: string[]; residualPolynomial: string[];
  dimensionlessCoupling: string; independentRoot: { lower: string; upper: string };
  approximations: { order: number; coefficient: string; term: string; approximation: string; residual: string; errorBound: string; energyGapBound: string }[];
}[];

for (const reference of cases) test(`independent coefficient, residual and isolated root: ${reference.id}`, () => {
  const result = calculateRegularPerturbation(reference.input);
  assert.deepEqual(result.coefficients, reference.coefficients);
  assert.deepEqual(result.residualPolynomial, reference.residualPolynomial);
  assert.equal(result.dimensionlessCoupling, reference.dimensionlessCoupling);
  assert.deepEqual(result.approximations.map(row => Object.fromEntries(Object.entries(row).filter(([key]) => key !== 'rootDistance'))), reference.approximations);
  assert.equal(result.evaluatedResidual, result.approximations.at(-1)!.residual);
  assert(result.residualPolynomial.slice(0, reference.input.order + 1).every(c => c === '0'));
  const lower = read(result.root.lower), upper = read(result.root.upper);
  assert(le(lower, read(reference.independentRoot.upper)) && le(read(reference.independentRoot.lower), upper));
  const k = read(reference.input.stiffness), lambda = read(reference.input.coupling), h = read(reference.input.source);
  const F = (x: Q) => sub(add(mul(k, x), mul(lambda, cube(x))), h);
  assert(le(F(lower), zero) && le(zero, F(upper)));
  for (const row of result.approximations) {
    const a = read(row.approximation), bound = read(row.errorBound);
    // Prove the error enclosure through monotonic signs, independent of bisection.
    assert(le(F(sub(a, bound)), zero) && le(zero, F(add(a, bound))));
    const dl = read(row.rootDistance.lower), du = read(row.rootDistance.upper);
    assert(le(zero, dl) && le(dl, du));
    assert(le(F(sub(a, du)), zero) && le(zero, F(add(a, du))));
    if (!equal(dl, zero)) assert(le(zero, F(sub(a, dl))) || le(F(add(a, dl)), zero));
  }
  assert(le(zero, read(result.susceptibility.lower)));
  assert(le(read(result.susceptibility.lower), read(result.susceptibility.upper)));
});

test('fourteen teaching examples are frozen and match the independent ledger', () => {
  assert.equal(regularPerturbationExamples.length, 14);
  for (const [i, example] of regularPerturbationExamples.entries()) {
    assert.deepEqual(example.input, cases[i].input);
    assert(Object.isFrozen(example.input));
    assert.deepEqual(readRegularPerturbationInput(example.input), example.input);
  }
});

test('default expansion has the hand-derived first five coefficients and residual', () => {
  const r = calculateRegularPerturbation(regularPerturbationDefault);
  assert.deepEqual(r.coefficients, ['1', '-1', '3', '-12', '55']);
  assert.equal(r.approximations[2].approximation, '93/100');
  assert.equal(r.approximations[2].residual, '104357/10000000');
  assert.equal(r.approximations[2].errorBound, '104357/10000000');
  assert.equal(r.convergenceRegion, 'inside-radius');
});

test('real uniqueness does not imply convergence at the chosen coupling', () => {
  const outside = calculateRegularPerturbation(regularPerturbationExamples.find(x => x.id === 'outside-radius')!.input);
  assert.equal(outside.convergenceRegion, 'outside-radius');
  assert(le(read(outside.root.lower), read(outside.root.upper)));
  assert.equal(calculateRegularPerturbation(regularPerturbationExamples.find(x => x.id === 'critical-radius')!.input).convergenceRegion, 'radius-boundary');
  assert.equal(calculateRegularPerturbation(regularPerturbationExamples.find(x => x.id === 'zero-source')!.input).convergenceRegion, 'identically-zero');
});

test('source parity and common rescaling retain the expected physical relation', () => {
  const a = calculateRegularPerturbation(regularPerturbationDefault);
  const b = calculateRegularPerturbation({ ...regularPerturbationDefault, source: '-1' });
  const c = calculateRegularPerturbation({ ...regularPerturbationDefault, stiffness: '2', coupling: '1/5', source: '2' });
  assert.equal(a.dimensionlessCoupling, c.dimensionlessCoupling);
  for (let i = 0; i < a.approximations.length; i++) {
    assert(equal(read(a.approximations[i].approximation), neg(read(b.approximations[i].approximation))));
    assert.equal(a.approximations[i].approximation, c.approximations[i].approximation);
    assert.equal(a.approximations[i].errorBound, c.approximations[i].errorBound);
  }
});

test('nonlinear susceptibility is bounded by the strong convexity inverse', () => {
  for (const reference of cases) {
    const result = calculateRegularPerturbation(reference.input);
    const invK = { n: read(reference.input.stiffness).d, d: read(reference.input.stiffness).n };
    assert(le(read(result.susceptibility.upper), invK));
    if (reference.input.coupling === '0' || reference.input.source === '0') assert(equal(read(result.susceptibility.lower), invK));
  }
});

test('input ownership rejects getters, symbols, nonfinite orders and out-of-scope fractions', () => {
  let called = false;
  const getter = { ...regularPerturbationDefault };
  Object.defineProperty(getter, 'source', { get() { called = true; return '1'; }, enumerable: true });
  assert.equal(readRegularPerturbationInput(getter), null);
  assert.equal(called, false);
  for (const extra of [{ other: 1 }, { [Symbol('secret')]: 1 }]) assert.equal(readRegularPerturbationInput({ ...regularPerturbationDefault, ...extra }), null);
  for (const [key, values] of Object.entries({ stiffness: ['0', '-1', '1/1001', '17', '1000000000000', '1/0'], coupling: ['-1', '17', '1e-3', '0.1'], source: ['17', '-17', 'Infinity', '1/0002'], order: [-1, 17, 1.5, NaN, Infinity, '4'] })) {
    for (const value of values) assert.equal(readRegularPerturbationInput({ ...regularPerturbationDefault, [key]: value }), null, `${key}: ${value}`);
  }
  const input = { ...regularPerturbationDefault };
  const copied = readRegularPerturbationInput(input)!;
  input.source = '2';
  assert.equal(copied.source, '1');
  assert(Object.isFrozen(copied));
  assert.throws(() => calculateRegularPerturbation({ ...input, stiffness: '0' }), RangeError);
});

test('input rational boundaries use exact small-integer cross-products', () => {
  for (const input of [
    { ...regularPerturbationDefault, stiffness: '1/1000' },
    { ...regularPerturbationDefault, stiffness: '999999999/999999999000' },
    { ...regularPerturbationDefault, coupling: '999999999984/62499999999' },
    { ...regularPerturbationDefault, source: '-999999999984/62499999999' },
  ]) assert(readRegularPerturbationInput(input));
  for (const input of [
    { ...regularPerturbationDefault, stiffness: '999999999/999999999001' },
    { ...regularPerturbationDefault, coupling: '999999999985/62499999999' },
    { ...regularPerturbationDefault, source: '-999999999985/62499999999' },
  ]) assert.equal(readRegularPerturbationInput(input), null);
});

test('zero residual and linear limits have exact certificates', () => {
  const r = calculateRegularPerturbation(regularPerturbationExamples.find(x => x.id === 'linear-limit')!.input);
  assert.equal(r.root.lower, '3/2');
  assert.equal(r.root.upper, '3/2');
  assert(r.approximations.every(row => row.approximation === '3/2' && row.errorBound === '0' && row.energyGapBound === '0'));
  const z = calculateRegularPerturbation(regularPerturbationExamples.find(x => x.id === 'zero-source')!.input);
  assert.equal(z.root.width, '0');
  assert(z.approximations.every(row => row.rootDistance.upper === '0'));
  const exact = calculateRegularPerturbation(regularPerturbationExamples.find(x => x.id === 'exact-rational-root')!.input);
  assert.equal(exact.root.lower, '1');
  assert.equal(exact.root.upper, '1');
});

test('the main-thread input and example bundle excludes exact arithmetic and the solver', async () => {
  const bundle = await build({ entryPoints: ['src/lib/learning-regular-perturbation-input.ts'], bundle: true, write: false, platform: 'browser', format: 'esm', metafile: true, logLevel: 'silent' });
  assert.equal(Object.keys(bundle.metafile!.inputs).length, 1);
  assert(!bundle.outputFiles[0].text.includes('BigInt('));
});

test('both unregistered manuscripts render seventeen sections, eleven equations and seven explained problems', async () => {
  const id = 'regular-perturbation-residual-certificates';
  const rendered = await getLearningContent(id), root = '.runtime/learning/regular-perturbation-foundation';
  mkdirSync(root, { recursive: true });
  const evidence = [];
  for (const language of ['ko', 'en'] as const) {
    const source = readFileSync(`content/learning/physics-math-for-ai/${id}.${language}.md`, 'utf8');
    assert.equal(rendered.outline[language].length, 17);
    assert.equal((source.match(/^\d\. \*\*/gm) ?? []).length, 7);
    assert.equal((source.match(/^\$\$$/gm) ?? []).length / 2, 11);
    assert(source.length > 7500);
    assert(!rendered[language].includes('katex-error'));
    assert(!/<h1(?:\s|>)/.test(rendered[language]));
    assert(rendered[language].includes('<math'));
    if (language === 'en') assert(!/[가-힣]/.test(rendered[language]));
    writeFileSync(`${root}/manuscript-${language}.html`, rendered[language]);
    evidence.push({ language, sections: 17, equations: 11, explainedExercises: 7, sourceCharacters: source.length, strictMath: true });
  }
  writeFileSync(`${root}/manuscripts.json`, JSON.stringify(evidence, null, 2) + '\n');
});
