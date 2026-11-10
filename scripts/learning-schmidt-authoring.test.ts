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
import { schmidtModel as model } from '../src/lib/learning-schmidt';
import { isMasterObservation } from '../src/lib/learning-master-equation-protocol';
import type { PartitionObservation as Q } from '../src/lib/learning-partition-exact-core';
ensureTestArtifactDirectory('.runtime/learning/schmidt-manuscripts', { recursive: true });

const id = 'schmidt-cuts-bond-rank-truncation', base = 'docs/online-learning/physics-math-for-ai', runtime = '.runtime/learning/schmidt-manuscripts';
const sources = { ko: readFileSync(`${base}/${id}-chapter.ko.md`, 'utf8'), en: readFileSync(`${base}/${id}-chapter.en.md`, 'utf8') };
const references = JSON.parse(readFileSync('scripts/fixtures/schmidt-authoring-reference.json', 'utf8'));
const metrics: Record<string, unknown> = {};
for (const lang of ['ko', 'en'] as const) test(`Schmidt ${lang} manuscript: detailed explanations and valid mathematical/table rendering`, async () => {
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
        assert.equal(rows.length, [4, 3, 5, 6][tables - 1]);
        rows.forEach(row => assert.equal(row.children.filter(c => c.type === 'element' && ['th', 'td'].includes(c.tagName)).length, [4, 4, 3, 6][tables - 1]));
      }
      if (Array.isArray(node.properties.className) && node.properties.className.includes('katex-display')) equations++;
    });
  }).use(rehypeStringify).process(source);
  assert.equal(result.messages.length, 0); assert(!String(result).includes('katex-error')); assert.equal(headings.length, 16); assert.equal(new Set(headings).size, headings.length); assert.equal(tables, 4); assert.equal(equations, 16);
  const text = String(result); assert(text.includes('|00〉') && text.includes('|11〉'), 'ket notation survives rendering');
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
let directedCertificateChecks = 0;
function directedZero(value: unknown) {
  if (value === null) return;
  if (isMasterObservation(value)) {
    const v = value as Q; assert(v.bounds);
    const lo = endpoint(v.bounds.lower), hi = endpoint(v.bounds.upper);
    assert(le(lo, frac(BigInt(0))) && le(frac(BigInt(0)), hi), 'directed certificate contains0');
    assert(le(frac(hi.n * lo.d - lo.n * hi.d, hi.d * lo.d), frac(BigInt(1), BigInt(10) ** BigInt(125))), 'directed certificate width');
    directedCertificateChecks++; return;
  }
  assert(value && typeof value === 'object'); Object.values(value).forEach(directedZero);
}
for (const ref of references.cases) test(`Schmidt manuscript independent eigenspace example: ${ref.label}`, () => {
  const r = model(ref.input); compare(r, ref.expected);
  for (const cut of r.cuts) {
    directedZero(cut.directedResiduals);
    if (cut.truncation.chosen) directedZero(cut.truncation.chosen.directedResiduals);
  }
});
const named = (label: string) => { const ref = references.cases.find((r: { label: string }) => r.label === label); assert(ref); return model(ref.input); };
test('unequal example distinguishes unnormalized error and normalized distance', () => {
  const c = named('unequal two-mode budget1').cuts[0];
  observation(c.purity, '17/25', 'purity');
  observation(c.truncation.discardedWeight, 'decimal:0.2', 'raw error');
  assert(c.truncation.chosen);
  observation(c.truncation.chosen.normalizedAmplitudes![0].real, 'decimal:1', 'renormalized amplitude');
  assert.notDeepEqual(c.truncation.optimalUnnormalizedSquaredError, c.truncation.optimalNormalizedDistanceSquared);
});
test('only splitting a positive group removes the chosen state in the five-row budget table', () => {
  for (let budget = 0; budget <= 4; budget++) {
    const c = named(`diagonal3-2-2-1 budget${budget}`).cuts[1];
    observation(c.purity, '19/54', 'four-weight purity');
    assert.equal(c.truncation.boundaryDegenerate, budget === 2);
    assert.equal(c.truncation.chosen === null, budget === 2);
    assert.equal(c.truncation.optimalUnnormalizedStateUnique, budget !== 2);
    assert.equal(c.truncation.normalizedStateDefinedForEveryOptimum, budget > 0);
    if (!budget) { assert.equal(c.truncation.optimalNormalizedFidelitySquared, null); assert.equal(c.truncation.chosen!.normalizedAmplitudes, null); }
  }
  const bell = named('Bell split positive group').cuts[0];
  assert.equal(bell.truncation.chosen, null); observation(bell.truncation.discardedWeight, 'decimal:0.5', 'Bell best error');
});
test('right projector preserves the leading complex ket and rejects the conjugated direction', () => {
  const c = named('right phase projector transpose').cuts[1];
  assert.equal(c.smallerSide, 'right');
  observation(c.rhoRight[0][1].imaginary, '-3/10', 'physical density phase');
  assert(c.truncation.chosen);
  observation(c.truncation.chosen.truncatedRawCoefficientMatrix[0][1].imaginary, 'decimal:2', 'kept phase');
  observation(c.truncation.chosen.directSquaredError, 'decimal:0.2', 'correct right action error');
  assert.equal(references.analytic.wrongRightProjectionSquaredError, '4/5');
});
test('adjacent and crossing Bell pairs have different middle-cut capacities and tiny weights stay positive', () => {
  const adjacent = named('adjacent Bell pairs'), crossing = named('crossing Bell pairs');
  assert.deepEqual(adjacent.cuts.map(c => c.schmidtRank), [2, 1, 2]);
  assert.deepEqual(crossing.cuts.map(c => c.schmidtRank), [2, 4, 2]);
  observation(adjacent.cuts[1].entropyBits, 'decimal:0', 'no middle entanglement');
  observation(crossing.cuts[1].entropyBits, 'decimal:2', 'two crossing pairs');
  const tiny = named('tiny positive second weight'); assert.equal(tiny.cuts[0].schmidtRank, 2);
  assert.equal(tiny.cuts[0].spectrum[1].weight.sign, 1);
  observation(tiny.rawNormSquared, references.analytic.tinyRawNorm, 'near singular raw norm');
});
after(() => writeFileSync(`${runtime}/manuscript-metrics.json`, JSON.stringify({ ...metrics, exactChecks, directedChecks, directedCertificateChecks }, null, 2) + '\n'));
