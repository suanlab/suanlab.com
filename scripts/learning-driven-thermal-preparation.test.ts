import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { drivenTwoStateThermodynamics as run, isDrivenThermalParameters as valid, type DrivenThermalParameters as Parameters } from '../src/lib/learning-driven-two-state-thermodynamics';
import type { PartitionObservation as Q, PartitionRational } from '../src/lib/learning-partition-exact-core';
const fixtures = JSON.parse(readFileSync('scripts/fixtures/driven-two-state-thermal-reference.json', 'utf8')) as { cases: { label: string; input: Parameters; canonicalInitial: boolean; expected: unknown }[] };
type Fraction = { n: bigint; d: bigint };
const f = (n: bigint, d = BigInt(1)): Fraction => ({ n, d });
const le = (a: Fraction, b: Fraction) => a.n * b.d <= b.n * a.d;
function parse(value: string): Fraction {
  if (value.includes('/')) { const [n, d] = value.split('/'); return f(BigInt(n), BigInt(d)); }
  const m = /^(-?)(\d+)(?:\.(\d*))?(?:e([+-]?\d+))?$/i.exec(value); assert(m, value);
  const n = BigInt(m[1] + m[2] + (m[3] || '')), power = Number(m[4] || 0) - (m[3] || '').length;
  return power >= 0 ? f(n * BigInt(10) ** BigInt(power)) : f(n, BigInt(10) ** BigInt(-power));
}
function endpoint(q: PartitionRational): Fraction {
  const n = BigInt(q.numerator.coefficient), d = BigInt(q.denominator.coefficient), e = q.numerator.exponent - q.denominator.exponent;
  return e >= 0 ? f(n << BigInt(e), d) : f(n, d << BigInt(-e));
}
function encloses(q: Q, source: string, label: string) {
  assert(q.bounds, label); const lo = endpoint(q.bounds.lower), hi = endpoint(q.bounds.upper);
  const scientific = /^(-?)(\d+)(?:\.(\d*))?e([+-]?\d+)$/i.exec(source);
  // A containing reference interval avoids materializing10^434294482 for
  // exp(-1e9). This is not an assertion that the positive decay is zero.
  const tiny = scientific && scientific[2].length - 1 + Number(scientific[4]) < -4096;
  const tinyBound = f(BigInt(1), BigInt(10) ** BigInt(4096));
  const lower = tiny ? (scientific[1] ? f(-tinyBound.n, tinyBound.d) : f(BigInt(0))) : parse(source);
  const upper = tiny ? (scientific[1] ? f(BigInt(0)) : tinyBound) : lower;
  const magnitude = [lower.n < BigInt(0) ? -lower.n : lower.n, upper.n < BigInt(0) ? -upper.n : upper.n].reduce((a, b) => a > b ? a : b);
  const un = magnitude > lower.d ? magnitude : lower.d, ud = lower.d * BigInt(10) ** BigInt(160);
  // Only the uncertainty of the stored180-digit independent reference.
  const below = f(lower.n * ud - un * lower.d, lower.d * ud), above = f(upper.n * ud + un * upper.d, upper.d * ud);
  assert(le(lo, hi) && le(lo, above) && le(below, hi), label + ': ' + source);
  assert(q.value === null || Number.isFinite(q.value), label);
  if (q.value === 0) assert.equal(q.resolution, 'structural-zero', label);
  if (q.resolution === 'unresolved-at-precision-ceiling') assert.equal(q.value, null, label);
}
let observations = 0;
function compare(actual: unknown, expected: unknown, path: string) {
  if (typeof expected === 'string' && actual !== null && typeof actual === 'object' && 'bounds' in actual) { encloses(actual as Q, expected, path); observations++; return; }
  if (Array.isArray(expected)) { assert(Array.isArray(actual)); assert.equal(actual.length, expected.length, path); expected.forEach((v, i) => compare(actual[i], v, path + '.' + i)); return; }
  if (expected !== null && typeof expected === 'object') { assert(actual !== null && typeof actual === 'object'); for (const [k, v] of Object.entries(expected)) compare((actual as Record<string, unknown>)[k], v, path + '.' + k); return; }
  assert.deepEqual(actual, expected, path);
}
for (const c of fixtures.cases) test(`independent driven-thermal reference: ${c.label}`, () => {
  const result = run(c.input); compare(result, c.expected, c.label);
  assert.equal(result.jarzynskiApplicable, c.canonicalInitial); assert.equal(result.initialDistributionIsCanonical, c.canonicalInitial);
  assert(result.classification.endpointBinsIntegrateAllJumpCounts); assert.equal(result.classification.normalizedOrTruncatedPathSum, false);
  for (let i = 0; i < 2; i++) if (c.input[i === 0 ? 'time1Numerator' : 'time2Numerator'] > 0) assert.notEqual(result.holds[i].decay.resolution, 'structural-zero');
});
const find = (label: string) => fixtures.cases.find(c => c.label === label)!.input;
test('exact equilibrium path zeros, zero-time support and cyclic zero work are structural', () => {
  const equilibrium = run(find('no quenches equilibrium')); assert.equal(equilibrium.totalEntropyProduction.resolution, 'structural-zero');
  assert(equilibrium.endpointAtoms.filter(a => a.reachable).every(a => a.totalEntropy?.resolution === 'structural-zero'));
  const cyclic = run(find('instantaneous cyclic quenches')); assert(cyclic.classification.exactlyZeroWork); assert.equal(cyclic.meanWork.value, 0); assert.equal(cyclic.meanWorkVariance.value, 0);
  assert.equal(cyclic.endpointAtoms.filter(a => a.reachable).length, 2);
  assert(cyclic.endpointAtoms.filter(a => !a.reachable).every(a => a.probability.value === 0 && a.totalEntropy === null && a.firstLawResidual === null));
  const fast = run(find('fast mixing with tiny later deviation'));
  assert.equal(fast.holds[1].exactlyZeroProduction, false);
  assert.equal(fast.holds[1].totalEntropyProduction.resolution, 'unresolved-at-precision-ceiling');
  assert.notEqual(fast.stages.at(-1)!.relativeEntropyToCanonical.resolution, 'structural-zero');
});
test('a noncanonical start does not inherit canonical JE or uncorrected Crooks', () => {
  const eq = run(find('equilibrium start')), neq = run(find('nonequilibrium start'));
  assert(eq.jarzynskiApplicable); assert.equal(eq.jarzynskiIdentity?.value, .75);
  assert.equal(neq.jarzynskiApplicable, false); assert.equal(neq.jarzynskiIdentity, null);
  assert(Math.abs(neq.workExponentialMoment.value! - .75) > .01);
  assert(neq.workAtoms.some(a => a.reachable && a.initialDensityCorrection.value !== 1));
  assert(eq.endpointAtoms.some(a => a.totalEntropy?.value !== null && a.totalEntropy?.value !== undefined && a.totalEntropy.value < 0));
  assert(eq.totalEntropyProduction.value! > 0);
});
test('final relaxation changes heat and entropy while preserving the complete work law', () => {
  const p = find('equilibrium start'), a = run(p), b = run({ ...p, time2Numerator: 0 });
  assert.deepEqual(a.workAtoms, b.workAtoms); assert.deepEqual(a.workExponentialMoment, b.workExponentialMoment); assert.deepEqual(a.meanWorkVariance, b.meanWorkVariance);
  assert.notDeepEqual(a.stages.at(-1)!.stateOneProbability, b.stages.at(-1)!.stateOneProbability);
});
test('extractable noncanonical free energy and zero-mean fluctuating work are different', () => {
  const extraction = run(find('extract from nonequilibrium')), zeroMean = run(find('zero mean with fluctuating work'));
  assert(extraction.meanWork.value! < 0); assert(extraction.totalEntropyProduction.value! > 0);
  assert.equal(zeroMean.meanWork.resolution, 'structural-zero'); assert(zeroMean.meanWorkVariance.value! > 0);
  assert.equal(zeroMean.classification.exactlyZeroWork, false); assert(zeroMean.classification.zeroMeanWorkIdentityApplies);
  assert(zeroMean.workExponentialMoment.value! > 1); assert.equal(zeroMean.jarzynskiApplicable, false);
});
test('temperature rescales SI energies while dimensionless dynamics and work stay fixed', () => {
  const a = run(find('equilibrium start')), b = run(find('hot SI rescaling'));
  assert.deepEqual(a.meanWork, b.meanWork); assert.deepEqual(a.workAtoms, b.workAtoms);
  assert(Math.abs(b.stages.at(-1)!.meanWorkJ.value! / a.stages.at(-1)!.meanWorkJ.value! - 2) < 1e-14);
  assert.equal(a.boltzmannConstantJPerK.value, 1.380649e-23);
});
test('eighteen-field input ownership and full initial support are enforced', () => {
  const p = find('equilibrium start'), copy = JSON.stringify(p); assert(valid(p)); const result = run(p); assert.equal(JSON.stringify(p), copy); assert(Object.isFrozen(result.parameters));
  for (const value of [null, { ...p, extra: 1 }, { ...p, r1Numerator: 0 }, { ...p, clock1Numerator: 0 }, { ...p, temperatureNumerator: 0 }, { ...p, initialOneNumerator: 0 }, { ...p, initialOneNumerator: p.initialDenominator }, { ...p, time1Numerator: -1 }, { ...p, time1Numerator: 5 * p.time1Denominator }, { ...p, r0Denominator: Infinity }, { ...p, time2Denominator: 1.5 }]) { assert.equal(valid(value), false); assert.throws(() => run(value as Parameters), RangeError); }
});
test('independent reference coverage visits every stored scalar certificate', () => {
  function count(value: unknown): number {
    if (typeof value === 'string') return /^-?\d+(?:\.\d*)?(?:e[+-]?\d+)?(?:\/\d+)?$/i.test(value) ? 1 : 0;
    return value && typeof value === 'object' ? Object.values(value).reduce<number>((n, v) => n + count(v), 0) : 0;
  }
  const expected = fixtures.cases.reduce((n, c) => n + count(c.expected), 0); assert.equal(observations, expected);
  console.log({ independentPreparations: fixtures.cases.length, scalarCertificates: observations });
});
