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
import { spinChainModel as model, createSpinChainParameters as create, type SpinChainParameters as P } from '../src/lib/learning-spin-chain';
import type { PartitionObservation as Q } from '../src/lib/learning-partition-exact-core';
const id = 'finite-spin-chain-local-hamiltonians', base = 'docs/online-learning/physics-math-for-ai', runtime = '.runtime/learning/spin-chain-manuscripts';
const sources = { ko: readFileSync(`${base}/${id}-chapter.ko.md`, 'utf8'), en: readFileSync(`${base}/${id}-chapter.en.md`, 'utf8') };
const references = JSON.parse(readFileSync('scripts/fixtures/spin-chain-authoring-reference.json', 'utf8'));
const metrics: Record<string, unknown> = {};
for (const lang of ['ko', 'en'] as const) test(`spin-chain ${lang} manuscript: detailed explanations and valid mathematical/table rendering`, async () => {
  const source = sources[lang]; assert(!source.startsWith('---')); assert(!/^# /m.test(source)); assert(!/@E\d+@/.test(source));
  assert(!/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/.test(source));
  if (lang === 'en') { assert(!/[가-힣]/.test(source)); assert(source.trim().split(/\s+/).length > 2500); }
  else assert((source.match(/[가-힣]/g) ?? []).length > 4500);
  assert.equal((source.match(/^\*\*\d\./gm) ?? []).length, 6);
  assert.equal((source.match(lang === 'ko' ? /^\*\*해설\./gm : /^\*\*Explanation\./gm) ?? []).length, 6);
  const headings: string[] = []; let tables = 0, equations = 0;
  const result = await unified().use(remarkParse).use(remarkGfm).use(remarkMath).use(remarkRehype).use(rehypeSlug).use(rehypeKatex, { strict: 'error' }).use(() => (tree: Root) => {
    visit(tree, 'element', node => {
      if (node.tagName === 'h2') { assert.equal(typeof node.properties.id, 'string'); headings.push(node.properties.id as string); }
      if (node.tagName === 'table') {
        tables++; const rows: Element[] = []; visit(node, 'element', row => { if (row.tagName === 'tr') rows.push(row); });
        assert.equal(rows.length, tables === 3 ? 4 : 5);
        rows.forEach(row => assert.equal(row.children.filter(c => c.type === 'element' && ['th', 'td'].includes(c.tagName)).length, tables === 3 ? 3 : 4));
      }
      if (Array.isArray(node.properties.className) && node.properties.className.includes('katex-display')) equations++;
    });
  }).use(rehypeStringify).process(source);
  assert.equal(result.messages.length, 0); assert(!String(result).includes('katex-error')); assert.equal(headings.length, 14); assert.equal(new Set(headings).size, headings.length); assert.equal(tables, 3); assert.equal(equations, 13);
  const text = String(result); assert(text.includes('|00〉') && text.includes('|11〉'), 'ket bars stay inside rendered cells');
  metrics[lang] = { sections: headings.length, tables, displayedEquations: equations, explainedExercises: 6, characters: source.length, words: source.trim().split(/\s+/).length, hangulCharacters: (source.match(/[가-힣]/g) ?? []).length };
  writeFileSync(`${runtime}/${id}.${lang}.html`, text);
});
test('translations preserve all derivations, six exercises and further-reading targets', () => {
  assert.deepEqual(sources.ko.match(/\$\$[\s\S]*?\$\$/g), sources.en.match(/\$\$[\s\S]*?\$\$/g));
  assert.deepEqual(sources.ko.match(/^\*\*\d\./gm), sources.en.match(/^\*\*\d\./gm));
  assert.deepEqual(sources.ko.match(/https:\/\/[^)]+/g), sources.en.match(/https:\/\/[^)]+/g));
});
function rational(v: Q, raw: string) {
  assert(v.bounds); const [n, d = '1'] = raw.split('/');
  for (const bound of [v.bounds.lower, v.bounds.upper]) {
    const coefficient = BigInt(bound.numerator.coefficient), denominator = BigInt(bound.denominator.coefficient), exponent = bound.numerator.exponent - bound.denominator.exponent;
    const a = exponent >= 0 ? coefficient * BigInt(10) ** BigInt(exponent) : coefficient;
    const b = exponent >= 0 ? denominator : denominator * BigInt(10) ** BigInt(-exponent);
    assert.equal(a * BigInt(d), BigInt(n) * b);
  }
}
function complexMatrix(actual: { real: Q; imaginary: Q }[][], expected: { real: string; imaginary: string }[][]) {
  assert.equal(actual.length, expected.length); actual.forEach((row, i) => { assert.equal(row.length, expected[i].length); row.forEach((v, j) => { rational(v.real, expected[i][j].real); rational(v.imaginary, expected[i][j].imaginary); }); });
}
const ghz = (fieldNumerator: number, denominator = 1) => {
  const p: P = { ...create(4), hDenominator: denominator, stateReal15: 1, couplingZZ01: -denominator, couplingZZ12: -denominator, couplingZZ23: -denominator };
  for (let i = 0; i < 4; i++) p[`fieldX${i}`] = -fieldNumerator;
  return model(p);
};
test('basis-action table matches independent full Kronecker matrices including Y signs', () => {
  complexMatrix(model({ ...create(2), fieldX0: 1 }).hamiltonian, references.basisActions.X0);
  complexMatrix(model({ ...create(2), fieldY1: 1 }).hamiltonian, references.basisActions.Y1);
  complexMatrix(model({ ...create(2), couplingZZ01: 1 }).hamiltonian, references.basisActions.Z0Z1);
});
test('normalization exercise and diagonal gap bound retain the full ground space', () => {
  const r = model({ ...create(2), couplingZZ01: -1, stateReal0: 2, stateImag1: 2 });
  for (const [key, referenceKey] of [['rawNormSquared', 'norm'], ['energy', 'energy'], ['secondMoment', 'second'], ['variance', 'variance']] as const) rational(r[key], references.rawNormalization[referenceKey]);
  assert(r.diagonalSpectrum); rational(r.diagonalSpectrum.groundProbability, '1/2'); rational(r.diagonalSpectrum.excitationGap!, '2'); rational(r.diagonalSpectrum.certifiedLeakageUpperBound!, '1/2');
  const excited = model({ ...create(2), couplingZZ01: -1, stateReal0: 0, stateReal1: 1 });
  rational(excited.variance, '0'); assert.equal(excited.classification.selectedStateGround, false);
});
test('XX YY cancellation and Heisenberg singlet/triplet examples use Pauli energy units', () => {
  for (const [amplitude, expected] of [[1, '2'], [-1, '-2']] as const) {
    const p = { ...create(2), stateReal0: 0, stateReal1: 1, stateReal2: amplitude, couplingXX01: 1, couplingYY01: 1 };
    const xy = model(p); rational(xy.energy, expected); rational(xy.variance, '0'); assert.equal(xy.classification.selectedStateGround, null);
    const heis = model({ ...p, couplingZZ01: 1 }); rational(heis.energy, amplitude === 1 ? '1' : '-3'); rational(heis.variance, '0');
  }
  assert.deepEqual(references.xySpectrum, { '0': 2, '-2': 1, '2': 1 }); assert.deepEqual(references.heisenbergSpectrum, { '-3': 1, '1': 3 });
});
test('GHZ field example has hidden action in variance and exact local/correlation statistics', () => {
  for (const [numerator, denominator, referenceKey] of [[1, 2, 'halfField'], [2, 1, 'doubleField']] as const) {
    const r = ghz(numerator, denominator), ref = references.ghz[referenceKey];
    rational(r.rawNormSquared, ref.norm); rational(r.energy, ref.energy); rational(r.secondMoment, ref.second); rational(r.variance, ref.variance);
    r.localStates.forEach((v, i) => { complexMatrix(v.density, references.ghz.localDensities[i]); rational(v.purity, '1/2'); });
    r.correlations.forEach(pair => rational(pair.values[2][2].connected, '1'));
    assert.equal(r.classification.selectedStateGround, null); assert(!r.classification.exactEigenstate);
  }
  const noField = ghz(0); rational(noField.variance, '0'); assert.equal(noField.classification.selectedStateGround, true);
});
test('sharpness and operator conservation differ, including cancellation across terms', () => {
  const xx = model({ ...create(2), couplingXX01: 1 }); assert(!xx.totalZ.conserved); assert(xx.totalZ.selectedStateSharp); assert(xx.globalZParity.conserved);
  const sum = model({ ...create(2), couplingXX01: 1, couplingYY01: 1 }); assert(sum.totalZ.conserved);
  const transverse = model({ ...create(2), fieldX0: 1 }); assert(!transverse.totalZ.conserved); assert(!transverse.globalZParity.conserved); assert(transverse.totalZ.selectedStateSharp);
  const spread = model({ ...create(2), couplingZZ01: 1, stateReal3: 1 }); assert(spread.totalZ.conserved); assert(!spread.totalZ.selectedStateSharp);
  assert(references.ghz.xParityConserved && references.ghz.zParityBrokenForNonzeroField);
});
test('frustrated triangle example has six ground configurations and a gap4', () => {
  const r = model({ ...create(3, 2), couplingZZ01: 1, couplingZZ12: 1, couplingZZ02: 1 }); assert(r.diagonalSpectrum);
  r.diagonalSpectrum.basisEnergies.forEach((v, i) => rational(v, references.triangle.basisEnergies[i]));
  rational(r.diagonalSpectrum.groundEnergy, '-1'); assert.equal(r.diagonalSpectrum.groundMultiplicity, 6); rational(r.diagonalSpectrum.excitationGap!, '4');
});
test('positive local Y, mixed-state counterexample and graph convention remain separate', () => {
  const p = { ...create(2), stateImag1: 1 }, r = model(p); rational(r.localStates[1].bloch[1], '1'); complexMatrix(r.localStates[1].density, references.complexRightDensity);
  assert.equal(references.ghz.xProductPure, '1'); assert.equal(references.ghz.xProductMixture, '0'); assert.equal(references.ghz.globalPurityPure, '1'); assert.equal(references.ghz.globalPurityMixture, '1/2');
  assert.equal(references.ghz.pairZComparison.length, 6); for (const pair of references.ghz.pairZComparison) { assert.equal(pair.pure, '1'); assert.equal(pair.mixture, '1'); }
  const open = model({ ...create(2), couplingZZ01: -1 }), ring = model({ ...create(2, 1), couplingZZ01: -1 }); assert.deepEqual(open.hamiltonian, ring.hamiltonian); assert.equal(ring.graphEdges.length, 1);
});
after(() => writeFileSync(`${runtime}/manuscript-metrics.json`, JSON.stringify(metrics, null, 2) + '\n'));
