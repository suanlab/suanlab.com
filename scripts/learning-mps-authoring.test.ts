import { mkdirSync as ensureTestArtifactDirectory } from 'node:fs';
import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import remarkRehype from 'remark-rehype';
import rehypeSlug from 'rehype-slug';
import rehypeKatex from 'rehype-katex';
import rehypeStringify from 'rehype-stringify';
import { visit } from 'unist-util-visit';
import type { Root, Element } from 'hast';
import { mpsModel as model } from '../src/lib/learning-mps';
import { isMasterObservation } from '../src/lib/learning-master-equation-protocol';
import type { PartitionObservation as Q } from '../src/lib/learning-partition-exact-core';
ensureTestArtifactDirectory('.runtime/learning/mps-manuscripts', { recursive: true });

const id = 'matrix-product-state-contraction-gauge', base = 'docs/online-learning/physics-math-for-ai', runtime = '.runtime/learning/mps-manuscripts';
const sources = { ko: readFileSync(`${base}/${id}-chapter.ko.md`, 'utf8'), en: readFileSync(`${base}/${id}-chapter.en.md`, 'utf8') };
const references = JSON.parse(readFileSync('scripts/fixtures/mps-authoring-reference.json', 'utf8'));
const metrics: Record<string, unknown> = {};
for (const lang of ['ko', 'en'] as const) test(`MPS ${lang} manuscript: detailed explanations and valid mathematical/table rendering`, async () => {
  const source = sources[lang]; assert(!source.startsWith('---')); assert(!/^# /m.test(source)); assert(!/@E\d+@/.test(source));
  assert(!/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/.test(source));
  if (lang === 'en') { assert(!/[가-힣]/.test(source)); assert(source.trim().split(/\s+/).length > 3000); }
  else assert((source.match(/[가-힣]/g) ?? []).length > 5000);
  assert.equal((source.match(/^\*\*\d\./gm) ?? []).length, 8);
  assert.equal((source.match(lang === 'ko' ? /^\*\*해설\./gm : /^\*\*Explanation\./gm) ?? []).length, 8);
  const headings: string[] = []; let tables = 0, equations = 0;
  const result = await unified().use(remarkParse).use(remarkGfm).use(remarkMath).use(remarkRehype).use(rehypeSlug).use(rehypeKatex, { strict: 'error' }).use(() => (tree: Root) => {
    visit(tree, 'element', node => {
      if (node.tagName === 'h2') { assert.equal(typeof node.properties.id, 'string'); headings.push(node.properties.id as string); }
      if (node.tagName === 'table') {
        tables++; const rows: Element[] = []; visit(node, 'element', row => { if (row.tagName === 'tr') rows.push(row); });
        assert.equal(rows.length, [5, 5, 5, 4][tables - 1]);
        rows.forEach(row => assert.equal(row.children.filter(c => c.type === 'element' && ['th', 'td'].includes(c.tagName)).length, [3, 3, 3, 4][tables - 1]));
      }
      if (Array.isArray(node.properties.className) && node.properties.className.includes('katex-display')) equations++;
    });
  }).use(rehypeStringify).process(source);
  assert.equal(result.messages.length, 0); assert(!String(result).includes('katex-error')); assert.equal(headings.length, 16); assert.equal(new Set(headings).size, headings.length); assert.equal(tables, 4); assert.equal(equations, 14);
  const text = String(result); assert(text.includes('|000〉') && text.includes('|111〉'), 'ket notation survives rendering');
  metrics[lang] = { sections: headings.length, tables, displayedEquations: equations, explainedExercises: 8, characters: source.length, words: source.trim().split(/\s+/).length, hangulCharacters: (source.match(/[가-힣]/g) ?? []).length };
  writeFileSync(`${runtime}/${id}.${lang}.html`, text);
});
test('translations preserve all derivations, eight exercises and further-reading targets', () => {
  assert.deepEqual(sources.ko.match(/\$\$[\s\S]*?\$\$/g), sources.en.match(/\$\$[\s\S]*?\$\$/g));
  assert.deepEqual(sources.ko.match(/^\*\*\d\./gm), sources.en.match(/^\*\*\d\./gm));
  assert.deepEqual(sources.ko.match(/https:\/\/[^)]+/g), sources.en.match(/https:\/\/[^)]+/g));
});
type F = { n: bigint; d: bigint };
const frac = (n: bigint, d = BigInt(1)): F => ({ n, d });
const le = (a: F, b: F) => a.n * b.d <= b.n * a.d;
const equal = (a: F, b: F) => a.n * b.d === b.n * a.d;
const add = (a: F, b: F) => frac(a.n * b.d + b.n * a.d, a.d * b.d);
const multiply = (a: F, b: F) => frac(a.n * b.n, a.d * b.d);
function parse(raw: string): F {
  if (raw.includes('/')) { const [n, d] = raw.split('/'); return frac(BigInt(n), BigInt(d)); }
  const m = /^(-?)(\d+)(?:\.(\d*))?(?:e([+-]?\d+))?$/i.exec(raw); assert(m, raw);
  const n = BigInt(m[1] + m[2] + (m[3] ?? '')), exponent = Number(m[4] ?? 0) - (m[3] ?? '').length;
  return exponent >= 0 ? frac(n * BigInt(10) ** BigInt(exponent)) : frac(n, BigInt(10) ** BigInt(-exponent));
}
function endpoint(b: NonNullable<Q['bounds']>['lower']): F {
  const n = BigInt(b.numerator.coefficient), d = BigInt(b.denominator.coefficient), e = b.numerator.exponent - b.denominator.exponent;
  return e >= 0 ? frac(n * BigInt(10) ** BigInt(e), d) : frac(n, d * BigInt(10) ** BigInt(-e));
}
const exact = (q: Q) => { assert(q.bounds); const lo = endpoint(q.bounds.lower), hi = endpoint(q.bounds.upper); assert(equal(lo, hi)); return lo; };
let exactChecks = 0, directedChecks = 0;
function observation(actual: unknown, expected: string, path: string) {
  assert(isMasterObservation(actual), path); const q = actual as Q; assert(q.bounds, path);
  const lo = endpoint(q.bounds.lower), hi = endpoint(q.bounds.upper); assert(le(lo, hi), path);
  const directed = expected.startsWith('decimal:'), value = parse(directed ? expected.slice(8) : expected);
  const absolute = value.n < BigInt(0) ? -value.n : value.n, scale = absolute > value.d ? absolute : value.d;
  if (directed) {
    const eps = BigInt(10) ** BigInt(205);
    assert(le(lo, frac(value.n * eps + scale, value.d * eps)) && le(frac(value.n * eps - scale, value.d * eps), hi), path + ' reference overlap');
    directedChecks++;
  } else { assert(equal(lo, value) && equal(hi, value), path + ' exact equality'); exactChecks++; }
  const width = frac(hi.n * lo.d - lo.n * hi.d, hi.d * lo.d);
  assert(le(width, frac(scale, value.d * BigInt(10) ** BigInt(125))), path + ' directed width');
}
function compare(actual: unknown, expected: unknown, path = 'result') {
  if (typeof expected === 'string' && (/^-?\d/.test(expected) || expected.startsWith('decimal:'))) return observation(actual, expected, path);
  if (Array.isArray(expected)) { assert(Array.isArray(actual), path); assert.equal(actual.length, expected.length, path); expected.forEach((v, i) => compare(actual[i], v, `${path}[${i}]`)); return; }
  if (expected && typeof expected === 'object') { assert(actual && typeof actual === 'object', path); for (const [key, value] of Object.entries(expected)) compare((actual as Record<string, unknown>)[key], value, `${path}.${key}`); return; }
  assert.equal(actual, expected, path);
}
for (const reference of references.cases) test(`manuscript independent path/Gram example: ${reference.label}`, () => {
  const r = model(reference.input); compare(r, reference.expected);
  assert.deepEqual(r.original.state, r.gauged.state);
  assert.deepEqual(r.original.physical, r.gauged.physical);
  for (const q of Object.values(r.exactResiduals)) if (q !== null) observation(q, '0', 'gauge residual');
});
const named = (label: string) => { const ref = references.cases.find((r: { label: string }) => r.label === label); assert(ref); return model(ref.input); };
test('GHZ paths, gauge parameter lengths and transformed environments match the displayed values', () => {
  const r = named('GHZ3 real scale shear');
  observation(r.original.normSquared, '2', 'physical norm');
  observation(r.original.tensorParameterSquaredNorm, '6', 'raw coordinates');
  observation(r.gauged.tensorParameterSquaredNorm, '19/2', 'gauged coordinates');
  assert.equal(r.original.state.filter(v => v.real.sign !== 0 || v.imaginary.sign !== 0).length, 2);
  for (const cut of r.original.cuts) assert.equal(cut.schmidtRank, 2);
  assert.equal(r.cost.activeComplexTensorEntries, 16);
  for (const pair of r.original.physical!.correlations) { observation(pair.values[2][2].value, '1', 'GHZ pair ZZ'); observation(pair.values[2][2].connected, '1', 'GHZ connected ZZ'); }
  assert.equal(r.original.canonicality[0].leftIsometry, true);
  assert.equal(r.original.canonicality[1].leftIsometry, true);
  assert.equal(r.original.canonicality[2].leftIsometry, false);
  observation(r.original.canonicality[2].leftGram[0][0].real, '2', 'last-site normalization');
  compare(r.gauged.leftEnvironments[1], [[{real:'4',imaginary:'0'},{real:'2',imaginary:'0'}],[{real:'2',imaginary:'0'},{real:'2',imaginary:'0'}]]);
  compare(r.gauged.rightEnvironments[1], [[{real:'1/2',imaginary:'0'},{real:'-1/2',imaginary:'0'}],[{real:'-1/2',imaginary:'0'},{real:'1',imaginary:'0'}]]);
});
test('complex Gram closure rejects the wrong trace even though both metrics are positive', () => {
  const r = named('complex cancellation closure orientation'); const E = r.original.leftEnvironments[1], R = r.original.rightEnvironments[1];
  assert.deepEqual(E, R); observation(E[0][1].imaginary, '1', 'complex off-diagonal');
  observation(r.original.normSquared, '0', 'cancellation norm');
  assert.equal(r.original.physical, null); assert.equal(r.validity.physicalStateDefined, false);
  const zero = frac(BigInt(0));
  const complexMultiplyReal = (a: typeof E[0][0], b: typeof E[0][0]) => add(multiply(exact(a.real), exact(b.real)), multiply(frac(BigInt(-1)), multiply(exact(a.imaginary), exact(b.imaginary))));
  let correct = zero, wrong = zero;
  for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) { correct = add(correct, complexMultiplyReal(E[a][b], R[a][b])); wrong = add(wrong, complexMultiplyReal(E[a][b], R[b][a])); }
  assert(equal(correct, parse('0'))); assert(equal(wrong, parse('4')));
});
test('a redundant bond4 product uses 48 entries but only rank1 at either cut', () => {
  const r = named('bond4 redundant product');
  assert.equal(r.cost.activeComplexTensorEntries, references.analytic.redundantProductActiveEntries);
  assert.equal(r.cost.denseComplexStateAmplitudes, 8);
  assert(r.original.cuts.every(c => c.bondDimension === 4 && c.schmidtRank === 1));
  assert(r.original.physical!.correlations.every(pair => pair.values.every(row => row.every(v => v.connected.sign === 0))));
});
after(() => writeFileSync(`${runtime}/manuscript-metrics.json`, JSON.stringify({ ...metrics, exactChecks, directedChecks }, null, 2) + '\n'));
