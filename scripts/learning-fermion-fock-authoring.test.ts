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
import { fermionFockModel as model, createFermionFockParameters as create, type FermionFockParameters as P } from '../src/lib/learning-fermion-fock';
import type { PartitionObservation as Q } from '../src/lib/learning-partition-exact-core';
const id = 'fermion-fock-parity-operator-algebra', base = 'docs/online-learning/physics-math-for-ai', runtime = '.runtime/learning/fermion-fock-manuscripts';
const sources = { ko: readFileSync(`${base}/${id}-chapter.ko.md`, 'utf8'), en: readFileSync(`${base}/${id}-chapter.en.md`, 'utf8') };
const references = JSON.parse(readFileSync('scripts/fixtures/fermion-fock-authoring-reference.json', 'utf8'));
const metrics: Record<string, unknown> = {};
for (const lang of ['ko', 'en'] as const) test(`fermion Fock ${lang} manuscript: detailed explanations and valid mathematical/table rendering`, async () => {
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
        assert.equal(rows.length, [5, 4, 3, 4][tables - 1]);
        rows.forEach(row => assert.equal(row.children.filter(c => c.type === 'element' && ['th', 'td'].includes(c.tagName)).length, tables === 2 || tables === 3 ? 4 : 3));
      }
      if (Array.isArray(node.properties.className) && node.properties.className.includes('katex-display')) equations++;
    });
  }).use(rehypeStringify).process(source);
  assert.equal(result.messages.length, 0); assert(!String(result).includes('katex-error')); assert.equal(headings.length, 16); assert.equal(new Set(headings).size, headings.length); assert.equal(tables, 4); assert.equal(equations, 17);
  const text = String(result); assert(text.includes('|00〉') && text.includes('|11〉'), 'ket bars stay inside rendered cells');
  metrics[lang] = { sections: headings.length, tables, displayedEquations: equations, explainedExercises: 8, characters: source.length, words: source.trim().split(/\s+/).length, hangulCharacters: (source.match(/[가-힣]/g) ?? []).length };
  writeFileSync(`${runtime}/${id}.${lang}.html`, text);
});
test('translations preserve all derivations, eight exercises and further-reading targets', () => {
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
function vector(actual: { real: Q; imaginary: Q }[], expected: { real: string; imaginary: string }[]) {
  assert.equal(actual.length, expected.length); actual.forEach((v, i) => { rational(v.real, expected[i].real); rational(v.imaginary, expected[i].imaginary); });
}
function stats(p: P, ref: Record<string, any>) {
  const r = model(p); rational(r.rawNormSquared, ref.norm); rational(r.energy, ref.energy); rational(r.variance, ref.variance);
  rational(r.totalNumber.expectation, ref.numberMean); rational(r.totalNumber.variance, ref.numberVariance);
  rational(r.fermionParity.expectation, ref.parityMean); assert.equal(r.totalNumber.conserved, ref.numberConserved); assert.equal(r.fermionParity.conserved, ref.parityConserved);
  complexMatrix(r.gamma, ref.gamma); complexMatrix(r.anomalous, ref.anomalous); vector(r.oddMeans, ref.oddMeans);
  rational(r.parityAssumption.coherenceFrobeniusSquared, ref.coherenceSquared); rational(r.parityDephased.purity, ref.twirlPurity); rational(r.parityDephased.energy, ref.twirlEnergy); vector(r.parityDephased.oddMeans, ref.twirlOddMeans);
  return r;
}
test('annihilation table and creation order match independent full JW matrices', () => {
  const r = model(create(2));
  r.operators.forEach((op, mode) => op.annihilationColumns.forEach((value, column) => {
    const rows = references.annihilators[mode], row = rows.findIndex((r: { real: string }[]) => r[column].real !== '0');
    assert.deepEqual(value, { row: row < 0 ? null : row, sign: row < 0 ? 0 : Number(rows[row][column].real) });
  }));
  const creation = model({ ...create(2), leftMode: 0, rightMode: 1, leftCreation: 1, rightCreation: 1 }); vector(creation.probe.forward, references.creationOrder.forward); vector(creation.probe.reverse, references.creationOrder.reverse);
  const exclusion = model({ ...create(2), leftMode: 0, rightMode: 0, leftCreation: 1, rightCreation: 1 }); vector(exclusion.probe.forward, references.creationOrder.excluded);
  rational(exclusion.rawNormSquared, '1'); rational(exclusion.probe.forwardSquaredNormOnNormalizedInput, '0');
});
test('actual mixed CAR paths cancel whereas independent bit ladders give2 on the selected state', () => {
  const p = { ...create(2, 2), stateReal1: 0, stateReal2: 1, leftMode: 0, rightMode: 1, leftCreation: 0, rightCreation: 1 };
  const r = model(p); vector(r.probe.forward, references.mixedCar.forward); vector(r.probe.reverse, references.mixedCar.reverse); rational(r.probe.carResidualFrobeniusSquared, '0'); rational(r.probe.naiveCarResidualFrobeniusSquared, '4');
  assert.equal(references.mixedCar.naiveSum[1].real, '2'); assert(references.exchangeAntisymmetry);
});
test('imaginary hopping and long-range occupied-middle hopping keep phase and parity strings', () => {
  const imaginary = stats({ ...create(2, 2), stateImag2: 1, hoppingImag01: 1 }, references.imaginaryHopping);
  rational(imaginary.gamma[0][1].imaginary, '-1/2'); rational(imaginary.energy, '1');
  const long = stats({ ...create(3), stateReal0: 0, stateReal3: 1, stateReal6: 1, hoppingReal02: 1 }, references.longRange);
  rational(long.energy, '-1'); rational(long.naiveComparison.energy, references.longRangeNaiveEnergy); rational(long.totalNumber.variance, '0'); assert.equal(long.classification.selectedStateGround, null);
});
test('real and complex pairing use reversed adjoints and even number superpositions', () => {
  const plus = stats({ ...create(2), stateReal3: 1, pairingReal01: 1 }, references.pairPlus);
  const minus = stats({ ...create(2), stateReal3: -1, pairingReal01: 1 }, references.pairMinus);
  rational(plus.energy, '1'); rational(minus.energy, '-1'); assert(plus.classification.exactEigenstate); assert.equal(plus.classification.selectedStateGround, null);
  const complex = stats({ ...create(2), stateImag3: 1, pairingImag01: 1 }, references.complexPair);
  const mean = complex.terms.find(t => t.type === 'pairing')!.expectation; rational(mean.real, references.complexCreationPairMean.real); rational(mean.imaginary, references.complexCreationPairMean.imaginary);
  assert.equal(references.wrongPairHermiticityResidualSquared, '8'); rational(complex.energy, '1');
});
test('number sharpness does not supply number conservation or replace parity conservation', () => {
  const vacuum = stats({ ...create(2), pairingReal01: 1 }, references.pairVacuum);
  assert(vacuum.totalNumber.selectedStateSharp); assert(!vacuum.totalNumber.conserved); assert(vacuum.fermionParity.conserved);
  const spread = model({ ...create(2), stateReal3: 1, onsite0: 1 }); assert(spread.totalNumber.conserved); assert(!spread.totalNumber.selectedStateSharp); rational(spread.totalNumber.variance, '1');
});
test('formal one-mode and three-mode states separate odd means from parity compatibility', () => {
  const formal = stats({ ...create(1, 0), stateReal1: 1, onsite0: 1 }, references.formalOneMode);
  assert(!formal.parityAssumption.pureStateCompatibleWithParitySuperselection); rational(formal.oddMeans[0].real, '1/2'); rational(formal.parityDephased.oddMeans[0].real, '0'); rational(formal.parityDephased.purity, '1/2');
  const cat = stats({ ...create(3, 0), stateReal7: 1 }, references.formalThreeCat); assert(cat.oddMeans.every(v => v.real.sign === 0 && v.imaginary.sign === 0)); assert(!cat.parityAssumption.pureStateCompatibleWithParitySuperselection); rational(cat.parityAssumption.coherenceFrobeniusSquared, '1/2');
  assert(references.symbolicPureCoherenceIdentity);
});
test('the same gamma and anomalous matrix do not fix the full number distribution', () => {
  const u = stats({ ...create(4), stateReal0: 0, stateReal3: 1, stateReal12: 1 }, references.sameGammaFixed);
  const v = stats({ ...create(4), stateReal15: 1 }, references.sameGammaCat);
  assert.deepEqual(u.gamma, v.gamma); assert.deepEqual(u.anomalous, v.anomalous); rational(u.totalNumber.variance, '0'); rational(v.totalNumber.variance, '4'); assert.notDeepEqual(u.numberDistribution, v.numberDistribution);
  assert.equal(references.sameGammaFixed.numberSecond, '4'); assert.equal(references.sameGammaCat.numberSecond, '8');
});
test('one-mode diagonal minima distinguish full Fock and selected parity sectors', () => {
  const r = stats({ ...create(1), onsite0: -1 }, references.vacuumSector); assert(r.diagonalSpectrum);
  rational(r.diagonalSpectrum.groundEnergy, references.sectorMinima.full); rational(r.diagonalSpectrum.parityMinima[0].energy, references.sectorMinima.even); rational(r.diagonalSpectrum.parityMinima[1].energy, references.sectorMinima.odd);
  assert.equal(r.classification.selectedStateGround, references.sectorMinima.vacuumFullGround); assert.equal(r.classification.selectedStateParitySectorGround, references.sectorMinima.vacuumEvenGround); assert.equal(r.diagonalSpectrum.parityMinima[0].excitationGap, null);
});
after(() => writeFileSync(`${runtime}/manuscript-metrics.json`, JSON.stringify(metrics, null, 2) + '\n'));
