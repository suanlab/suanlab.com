import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { getLearningContent } from '../src/lib/learning-content';
import { finiteBoundaryExtensivityModel as model, type BoundaryExtensivityParameters as Params } from '../src/lib/learning-boundary-extensivity';
import type { PartitionObservation as Q } from '../src/lib/learning-partition-exact-core';
const id = 'boundary-growth-extensivity-additivity';
const raw = (language: 'ko' | 'en') => readFileSync(`content/learning/physics-math-for-ai/${id}.${language}.md`, 'utf8');
const base: Params = { dimension: 2, lengthOne: 2, lengthTwo: 2, width: 3, growthMode: 'uniform-dilation', scale: 1, spinOne: 1, spinTwo: -1, couplingNumerator: 1, couplingDenominator: 1 };
function ratio(q: Q) { assert.ok(q.bounds); assert.deepEqual(q.bounds.lower, q.bounds.upper); return { n: BigInt(q.bounds.lower.numerator.coefficient), d: BigInt(q.bounds.lower.denominator.coefficient) }; }
function equal(q: Q, n: bigint | number, d: bigint | number = 1) { const a = ratio(q); assert.equal(a.n * BigInt(d), BigInt(n) * a.d); }
const printed = (q: Q) => { const { n, d } = ratio(q); return `${n < BigInt(0) ? '−' : ''}${n < BigInt(0) ? -n : n}${d === BigInt(1) ? '' : `/${d}`}`; };
function table(source: string, index: number) { const groups = source.match(/(?:^\|.*\|\n)+/gm) ?? []; assert.equal(groups.length, 4); return groups[index].trim().split('\n').slice(2).map(line => line.split('|').slice(1, -1).map(cell => cell.trim())); }
for (const language of ['ko', 'en'] as const) test(`${language}: full boundary manuscript renders accessible equations, tables, grid and six explained exercises`, async () => {
 const source = raw(language), content = await getLearningContent(id), html = content[language];
 assert.equal(content.outline[language].length, 17); assert.equal(new Set(content.outline[language].map(h => h.id)).size, 17);
 assert.equal((source.match(/^### [1-6]\. /gm) ?? []).length, 6); assert.equal((source.match(/\*\*(?:Explanation\.|해설\.)\*\*/g) ?? []).length, 6);
 assert.equal((html.match(/class="katex-display"/g) ?? []).length, 18); assert.equal((html.match(/aria-label="(?:Equation|수식) /g) ?? []).length, 18); assert.equal((html.match(/<table[^>]*tabindex="0"/g) ?? []).length, 4);
 assert.match(html, /<pre[^>]*tabindex="0"[^>]*role="region"/); assert.doesNotMatch(html, /katex-error|\$\$|@@E\d/); assert.doesNotMatch(source, /^# |^---/m);
 if (language === 'en') { assert.ok(source.split(/\s+/).length > 3000); assert.doesNotMatch(html, /[가-힣]/); } else assert.ok((source.match(/[가-힣]/g) ?? []).length > 6500);
});
test('translations share all displayed derivations and preserve the physical and numerical distinctions', () => {
 const equations = (source: string) => [...source.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(match => match[1].replace(/\s+/g, ''));
 assert.deepEqual(equations(raw('ko')), equations(raw('en')));
 assert.doesNotMatch(raw('en'), /not an already registered interactive reader/);
 for (const phrase of [/fixed lattice spacing/, /deterministic microstate/, /van Hove/, /does not imply that every energy density/, /over all spin configurations/, /opposite sign/, /configuration-independent shifts/, /J\/N/, /counterterm/, /not floating-point underflow/, /the current reader lets you manipulate/]) assert.match(raw('en'), phrase);
});
test('both ledger translations match the actual restricted, cross, combined and worst energies', () => {
 const r = model(base).current;
 const expected = (['nearest-neighbor', 'complete-unscaled', 'complete-kac'] as const).map(name => { const m = r.modes[name]; return [m.restrictedEnergyOne, m.restrictedEnergyTwo, m.crossEnergy, m.combinedEnergy, m.energyPerSite, m.maximumAbsoluteEnergy].map(printed); });
 for (const language of ['ko', 'en'] as const) assert.deepEqual(table(raw(language), 1).map(row => row.slice(1)), expected);
 equal(r.separatelyNormalizedKac.energySum, -5); equal(r.separatelyNormalizedKac.normalizationCounterterm, 5, 2); equal(r.separatelyNormalizedKac.combinedMinusSeparate, 11, 2);
});
test('both growth tables use actual site counts and exact densities on the stated spatial path', () => {
 const expected = (['uniform-dilation', 'fixed-thickness'] as const).flatMap(growthMode => [1, 2, 4].map(scale => { const r = model({ ...base, growthMode, scale }).current; return [String(scale), r.geometry.siteCount, printed(r.modes['nearest-neighbor'].energyPerSite), printed(r.modes['nearest-neighbor'].crossMagnitudePerSite)]; }));
 for (const language of ['ko', 'en'] as const) assert.deepEqual(table(raw(language), 2).map(row => row.slice(1)), expected);
});
test('the parallel-block exercise preserves the normalization counterterm rather than replacing it by the cut', () => {
 for (const scale of [1, 2, 4, 1000]) { const r = model({ ...base, spinTwo: 1, scale }).current; equal(r.separatelyNormalizedKac.combinedMinusSeparate, -1, 2); equal(r.modes['complete-kac'].crossEnergyPerSite, -1, 4); assert.notEqual(printed(r.modes['complete-kac'].crossEnergy), printed(r.separatelyNormalizedKac.combinedMinusSeparate)); }
});
test('the selected positive energy and the Hamiltonian growth conclusion remain different', () => {
 const first = model({ ...base, lengthTwo: 3 }), second = model({ ...base, lengthTwo: 3, scale: 2 }); equal(first.current.modes['complete-unscaled'].energyPerSite, 1, 5); equal(second.current.modes['complete-unscaled'].energyPerSite, -7, 10); assert.equal(second.limits['complete-unscaled'].energyPerSite.kind, 'negative-infinity');
 const balanced = model({ ...base, scale: 4 }); equal(balanced.current.modes['complete-unscaled'].energyPerSite, 1, 2); assert.equal(balanced.classification.maximumEnergyGrowth['complete-unscaled'], 'quadratic-in-site-count');
});
test('the two-site and zero-coupling exercises do not add self pairs or change the geometry classification', () => {
 const tiny = model({ ...base, dimension: 1, lengthOne: 1, lengthTwo: 1, width: 1 }).current; assert.equal(tiny.modes['complete-kac'].internalPairCountOne, '0'); equal(tiny.modes['complete-unscaled'].combinedEnergy, 1); equal(tiny.modes['complete-kac'].combinedEnergy, 1, 2);
 const zero = model({ ...base, growthMode: 'fixed-thickness', couplingNumerator: 0 }); assert.equal(zero.classification.vanHoveGeometryAlongPath, false); for (const m of Object.values(zero.current.modes)) { equal(m.combinedEnergy, 0); equal(m.crossEnergy, 0); equal(m.maximumAbsoluteEnergy, 0); }
});
test('the huge-count exercise keeps exact integer counts and nonzero representable corrections', () => {
 const r = model({ ...base, dimension: 3, lengthOne: 1000000, lengthTwo: 1000000, width: 1000000, scale: 1000000, couplingDenominator: 1000000 }).current;
 assert.equal(r.geometry.siteCount, '2000000000000000000000000000000000000'); equal(r.modes['nearest-neighbor'].crossMagnitudePerSite, BigInt(1), BigInt(2) * BigInt(10) ** BigInt(18)); equal(r.modes['complete-kac'].crossMagnitudePerSite, 1, 4000000); equal(r.modes['complete-kac'].energyPerSite, BigInt(1), BigInt(4) * BigInt(10) ** BigInt(42));
 assert.ok(r.modes['complete-kac'].energyPerSite.value! > 0); assert.equal(r.modes['complete-kac'].energyPerSite.sign, 1); assert.ok(BigInt(r.geometry.siteCount) > BigInt(Number.MAX_SAFE_INTEGER));
});
