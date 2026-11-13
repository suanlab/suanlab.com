import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { getLearningContent } from '../src/lib/learning-content';
import { twoPhaseRoundingModel as model, type TwoPhaseRoundingParameters as Params } from '../src/lib/learning-two-phase-rounding';
import type { PartitionObservation as Q } from '../src/lib/learning-partition-exact-core';
const id = 'two-phase-partition-zeros-rounding';
const raw = (language: 'ko' | 'en') => readFileSync(`content/learning/physics-math-for-ai/${id}.${language}.md`, 'utf8');
const base: Params = { siteCount: 4, activityNumerator: 2, activityDenominator: 1, sizePath: 'fixed', approachSide: 1, offsetNumerator: 1, offsetDenominator: 2 };
function fraction(q: Q) { assert.ok(q.bounds); assert.deepEqual(q.bounds.lower, q.bounds.upper); assert.equal(q.bounds.lower.numerator.exponent, 0); assert.equal(q.bounds.lower.denominator.exponent, 0); return { n: BigInt(q.bounds.lower.numerator.coefficient), d: BigInt(q.bounds.lower.denominator.coefficient) }; }
function equal(q: Q, n: bigint | number, d: bigint | number = 1) { const f = fraction(q); assert.equal(f.n * BigInt(d), BigInt(n) * f.d); }
const printed = (q: Q) => { const { n, d } = fraction(q); return `${n}${d === BigInt(1) ? '' : `/${d}`}`; };
function tables(source: string) { return (source.match(/(?:^\|.*\|\n)+/gm) ?? []).map(group => group.trim().split('\n').slice(2).map(line => line.split('|').slice(1, -1).map(cell => cell.trim()))); }
for (const language of ['ko', 'en'] as const) test(`${language}: two-phase manuscript renders complete math, ledgers, two diagrams and explained exercises`, async () => {
 const source = raw(language), content = await getLearningContent(id), html = content[language];
 assert.equal(content.outline[language].length, 17); assert.equal(new Set(content.outline[language].map(h => h.id)).size, 17);
 assert.equal((source.match(/^### [1-6]\. /gm) ?? []).length, 6); assert.equal((source.match(/\*\*(?:Explanation\.|해설\.)\*\*/g) ?? []).length, 6);
 assert.equal((html.match(/class="katex-display"/g) ?? []).length, 22); assert.equal((html.match(/aria-label="(?:Equation|수식) /g) ?? []).length, 22); assert.equal((html.match(/<table[^>]*tabindex="0"/g) ?? []).length, 5); assert.equal((html.match(/<pre[^>]*tabindex="0"[^>]*role="region"/g) ?? []).length, 3);
 assert.match(html, /No intermediate configurations are allowed/); assert.match(html, /The four roots lie on the unit circle/); assert.doesNotMatch(html, /katex-error|\$\$|@@E\d/); assert.doesNotMatch(source, /^# |^---/m);
 if (language === 'en') { assert.ok(source.split(/\s+/).length > 2400); assert.doesNotMatch(html, /[가-힣]/); } else assert.ok((source.match(/[가-힣]/g) ?? []).length > 4200);
});
test('both translations preserve every displayed equation and numerical limit-table entry', () => {
 const equations = (source: string) => [...source.matchAll(/\$\$([\s\S]*?)\$\$/g)].map(m => m[1].replace(/\s+/g, ''));
 assert.deepEqual(equations(raw('ko')), equations(raw('en')));
 for (const language of ['ko', 'en'] as const) assert.deepEqual(tables(raw(language))[1].map(row => row.slice(1)), [['−∞', '0', '0'], ['0', '1/2', '1/4'], ['+∞', '1', '0'], ['sc', 'L(sc)', 'L(sc)[1−L(sc)]'], ['0', '1/2', '1/4']]);
});
test('all printed finite ledger cells match the stated two-state law rather than an IID partition', () => {
 const expected = [[1, 2], [1, 1], [2, 1]].map(([activityNumerator, activityDenominator]) => {
  const r = model({ ...base, activityNumerator, activityDenominator }).current;
  return [r.activity, r.partitionFunction, r.occupiedProbability, r.densityVariance, r.densityResponseToField, r.exponentialPressureCorrection].map(printed);
 });
 for (const language of ['ko', 'en'] as const) assert.deepEqual(tables(raw(language))[0], expected);
 const r = model(base).current; equal(r.countVariance, 256, 289); equal(r.meanCount, 64, 17); equal(r.partitionFunction, 17); assert.notEqual(r.partitionFunction.value, 81);
});
test('both root ledgers preserve the exact turn fractions and distinguish the odd-size negative real root', () => {
 for (const language of ['ko', 'en'] as const) {
  const rows = tables(raw(language))[2]; assert.equal(rows.length, 4);
  for (const row of rows) { const siteCount = Number(row[0]), r = model({ ...base, siteCount }); assert.equal(row[1], r.roots.map(root => printed(root.turns)).join(', ')); assert.equal(row[2] === (language === 'ko' ? '예' : 'yes'), r.rootGeometry.negativeRealRootExists); equal(r.rootGeometry.nearestPositiveAxisTurns, 1, 2 * siteCount); }
 }
});
test('the two approach-path exercises use exact finite activities and different limiting densities', () => {
 const linear = model({ ...base, sizePath: 'inverse-size' }), square = model({ ...base, sizePath: 'inverse-size-squared' });
 equal(linear.current.activity, 9, 8); equal(linear.current.activityPower, 6561, 4096); equal(linear.current.occupiedProbability, 6561, 10657);
 equal(square.current.activity, 33, 32); equal(square.current.activityPower, 1185921, 1048576); equal(square.current.occupiedProbability, 1185921, 2234497);
 assert.equal(linear.limits.scaledField.kind, 'finite'); if (linear.limits.scaledField.kind === 'finite') equal(linear.limits.scaledField.value, 1, 2); equal(square.limits.meanDensity, 1, 2); assert.ok(linear.limits.meanDensity.value! > .62 && linear.limits.meanDensity.value! < .63);
 const fixed = model(base); equal(fixed.limits.meanDensity, 1); assert.equal(fixed.classification.selfAveragesAlongPath, true); assert.equal(linear.classification.selfAveragesAlongPath, false);
});
test('doubling coexistence size halves pressure correction and field width but doubles response', () => {
 const first = model({ ...base, activityNumerator: 1 }), second = model({ ...base, activityNumerator: 1, siteCount: 8 });
 equal(first.current.densityVariance, 1, 4); equal(second.current.densityVariance, 1, 4); equal(first.current.densityResponseToField, 1); equal(second.current.densityResponseToField, 2);
 assert.equal(first.current.pressureCorrection.value, 2 * second.current.pressureCorrection.value!); assert.equal(first.current.tenToNinetyFieldWidth.value, 2 * second.current.tenToNinetyFieldWidth.value!); assert.equal(second.limits.densityResponseToField.kind, 'positive-infinity');
});
test('the extreme logarithm exercise keeps a positive certified interval despite a rounded exponential', () => {
 const r = model({ ...base, siteCount: 512, activityNumerator: 1000000 }).current;
 equal(r.minorityToMajorityWeight, BigInt(1), BigInt(10) ** BigInt(3072)); assert.equal(r.exponentialPressureCorrection.value, 1); assert.equal(r.pressureCorrection.value, null); assert.equal(r.pressureCorrection.sign, 1); assert.equal(r.pressureCorrection.resolution, 'interval-resolved'); assert.ok(BigInt(r.pressureCorrection.bounds!.lower.numerator.coefficient) > BigInt(0)); assert.ok(r.pressureCorrection.logMagnitude! < -7000);
});
