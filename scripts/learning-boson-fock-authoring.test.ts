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
import { bosonFockModel as model, createBosonFockParameters as create } from '../src/lib/learning-boson-fock';
import { bosonCoherentModel as coherent, createBosonCoherentParameters as prepare } from '../src/lib/learning-boson-coherent';
import { isMasterObservation } from '../src/lib/learning-master-equation-protocol';
import type { PartitionObservation as Q } from '../src/lib/learning-partition-exact-core';
const id = 'boson-fock-cutoff-boundary-algebra', base = 'docs/online-learning/physics-math-for-ai', runtime = '.runtime/learning/boson-fock-manuscripts';
const sources = { ko: readFileSync(`${base}/${id}-chapter.ko.md`, 'utf8'), en: readFileSync(`${base}/${id}-chapter.en.md`, 'utf8') };
const references = JSON.parse(readFileSync('scripts/fixtures/boson-fock-authoring-reference.json', 'utf8'));
const metrics: Record<string, unknown> = {};
for (const lang of ['ko', 'en'] as const) test(`boson Fock ${lang} manuscript: detailed explanations and valid mathematical/table rendering`, async () => {
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
        assert.equal(rows.length, [5, 3, 4, 4][tables - 1]);
        rows.forEach(row => assert.equal(row.children.filter(c => c.type === 'element' && ['th', 'td'].includes(c.tagName)).length, [4, 4, 3, 5][tables - 1]));
      }
      if (Array.isArray(node.properties.className) && node.properties.className.includes('katex-display')) equations++;
    });
  }).use(rehypeStringify).process(source);
  assert.equal(result.messages.length, 0); assert(!String(result).includes('katex-error')); assert.equal(headings.length, 18); assert.equal(new Set(headings).size, headings.length); assert.equal(tables, 4); assert.equal(equations, 17);
  const text = String(result); assert(text.includes('|3〉') && text.includes('|α〉'), 'ket bars stay inside rendered cells');
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
type Result = ReturnType<typeof model>;
function stats(r: Result | ReturnType<typeof coherent>['normalizedProjection'], expected: Record<string, unknown>) {
  const local = r.localStates[0];
  compare(r.rawNormSquared, expected.norm);
  compare(local.annihilationMean, expected.annihilationMean);
  compare(r.totalNumber.expectation, expected.numberMean);
  compare(r.totalNumber.variance, expected.numberVariance);
  compare(local.finiteCommutatorExpectation, expected.commutatorMean);
  for (const key of ['xMean', 'pMean', 'xVariance', 'pVariance', 'covariance', 'robertsonBound'] as const) compare(local.quadratures[key], expected[key]);
  compare(r.energy, expected.energy); compare(r.variance, expected.variance);
}
test('all four cutoffs match independent ladder matrices and vacuum/top observables', () => {
  for (const ref of references.cutoffs) {
    const p = { ...create(1, ref.cutoff), onsite0: 1 };
    const vacuum = model(p); stats(vacuum, ref.vacuum);
    compare(vacuum.operators[0].annihilation, ref.annihilation);
    compare(vacuum.commutation[0].mixed, ref.commutator);
    observation(vacuum.commutation[0].trace.real.observation, ref.commutatorTrace, 'finite trace');
    const top = model({ ...p, stateReal0: 0, [`stateReal${ref.cutoff}`]: 1 }); stats(top, ref.top);
    assert(ref.nilpotent);
  }
});
test('a boundary superposition can have a negative mean finite commutator', () => {
  const r = model({ ...create(), onsite0: 1, stateReal3: 1 }); stats(r, references.boundaryCat);
  observation(r.localStates[0].finiteCommutatorExpectation.coefficients[0], '-1', 'cat boundary mean');
});
test('projected oscillator Hamiltonian retains its zero-point energy at every cutoff', () => {
  for (const ref of references.cutoffs) {
    const r = model({ ...create(1, ref.cutoff), onsite0: 2, offset: 1, hDenominator: 2 });
    compare(r.hamiltonian, ref.projectedHamiltonian);
    observation(r.energy.coefficients[0], '1/2', 'vacuum oscillator energy');
    // These independently computed matrices compare two definitions; the second
    // is not a second Hamiltonian mode exposed by the current calculator.
    assert.deepEqual(ref.projectedXSquareCorrection, ref.projectedPSquareCorrection);
    assert.notDeepEqual(ref.projectedHamiltonian, ref.recombinedFiniteHamiltonian);
  }
});
test('boundary hopping loses one sqrt6 path and halves the energy variance', () => {
  for (const ref of references.hopping) {
    const [c0, c1] = ref.cutoffs;
    const r = model({ ...create(2, c0, c1), stateReal0: 0, [`stateReal${2 * (c1 + 1) + 2}`]: 1, hoppingReal: 1 });
    compare(r.hamiltonianAction, ref.hamiltonianAction); compare(r.energy, ref.energy); compare(r.variance, ref.variance);
  }
});
test('unit coherent preparation preserves factorial roots and all finite quadratures', () => {
  const r = coherent({ ...prepare(), onsite0: 1 }); const ref = references.coherentUnit;
  compare(r.exactRawCoefficients, ref.rawCoefficients); observation(r.rawSeriesNorm, ref.seriesNorm, 'raw series norm');
  stats(r.normalizedProjection, ref.finite);
  observation(r.perMode[0].normalizedProjection.topProbability, ref.topProbability, 'top probability');
  observation(r.perMode[0].normalizedProjection.annihilationEigenResidualSquared.coefficients[0], ref.annihilatorResidualSquared, 'finite eigen residual');
});
test('embedded original quadrature variances add the boundary correction', () => {
  const r = coherent(prepare()).normalizedProjection.localStates[0];
  const correction = multiply(exact(r.topProbability.coefficients[0]), frac(BigInt(2)));
  for (const key of ['xVariance', 'pVariance'] as const) {
    const finite = exact(r.quadratures[key].coefficients[0]);
    const original = parse(references.coherentUnit.embeddedOriginalOperators[key].coefficients[0]);
    assert(equal(add(finite, correction), original));
  }
  assert.equal(references.coherentUnit.embeddedOriginalOperators.commutatorMean.coefficients[0], '1');
});
test('six tail and fidelity examples match independent 250-digit incomplete-gamma calculations', () => {
  for (const ref of references.tails) {
    const [muN, muD = '1'] = ref.mu.split('/');
    const r = coherent({ ...prepare(1, ref.cutoff), alphaReal0: Math.sqrt(Number(muN)), alphaDenominator0: Math.sqrt(Number(muD)) });
    observation(r.rawSeriesNorm, ref.seriesNorm, 'series norm');
    observation(r.truncation.retainedProbability, ref.retained, 'retained');
    observation(r.truncation.discardedProbability, ref.tail, 'tail');
    observation(r.normalizedProjection.fidelitySquaredToInfinite, ref.fidelitySquared, 'fidelity squared');
    observation(r.normalizedProjection.distanceSquaredToInfinite, ref.normalizedDistanceSquared, 'normalized distance squared');
  }
});
test('rectangular two-mode retention is the product and not the sum of tails', () => {
  const r = coherent({ ...prepare(2, 3, 2), alphaReal1: 1 });
  observation(r.truncation.retainedProbability, references.twoMode.retained, 'joint retention');
  observation(r.truncation.discardedProbability, references.twoMode.tail, 'joint tail');
  const residual = r.truncation.jointMinusProductRetained; assert(residual.bounds);
  assert(le(endpoint(residual.bounds.lower), frac(BigInt(0))) && le(frac(BigInt(0)), endpoint(residual.bounds.upper)));
  observation(residual, 'decimal:0', 'product interval identity');
});
after(() => writeFileSync(`${runtime}/manuscript-metrics.json`, JSON.stringify({ ...metrics, exactChecks, directedChecks }, null, 2) + '\n'));
