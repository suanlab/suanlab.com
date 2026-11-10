import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { coupledPartitionModel as model, type CoupledPartitionParameters } from '../src/lib/learning-coupled-partition';
import type { PartitionObservation, PartitionRational } from '../src/lib/learning-partition-exact-core';
import { canonicalBoltzmannConstant as kB } from '../src/lib/learning-canonical-spectrum';
type Fraction = { n: bigint; d: bigint };
const Z = BigInt(0), O = BigInt(1);
function rational(v: PartitionRational): Fraction {
  let n = BigInt(v.numerator.coefficient), d = BigInt(v.denominator.coefficient);
  const e = v.numerator.exponent - v.denominator.exponent;
  if (e >= 0) n <<= BigInt(e); else d <<= BigInt(-e);
  return { n, d };
}
function decimal(text: string): Fraction {
  const [mantissa, power = '0'] = text.toLowerCase().split('e');
  const dot = mantissa.indexOf('.'), digits = mantissa.replace('.', '');
  const exponent = Number(power) - (dot < 0 ? 0 : mantissa.length - dot - 1);
  return exponent >= 0 ? { n: BigInt(digits) * BigInt(10) ** BigInt(exponent), d: O }
    : { n: BigInt(digits), d: BigInt(10) ** BigInt(-exponent) };
}
const sub = (a: Fraction, b: Fraction): Fraction => ({ n: a.n * b.d - b.n * a.d, d: a.d * b.d });
const compare = (a: Fraction, b: Fraction) => a.n * b.d - b.n * a.d;
function check(q: PartitionObservation, reference: string, digits: number) {
  const actual = decimal(reference), value = Number(reference);
  assert.notEqual(q.resolution, 'unresolved-at-precision-ceiling');
  assert.equal(q.sign, actual.n === Z ? 0 : actual.n > Z ? 1 : -1);
  if (value === 0 && actual.n !== Z || !Number.isFinite(value)) assert.equal(q.value, null);
  else assert.ok(q.value !== null && Math.abs(q.value - value) <= Math.max(2 * Number.MIN_VALUE, Math.abs(value) * 1e-12), `${q.value} versus ${reference}`);
  assert.ok(q.bounds, 'Reference cases require directed rational bounds');
  const lower = rational(q.bounds.lower), upper = rational(q.bounds.upper);
  // Independent Decimal convergence is checked by an extra 80 digits.
  // A relative guard at reference precision minus 100 digits covers its rounding.
  const slack = actual.n === Z ? decimal(`1e-${digits - 100}`)
    : { n: actual.n < Z ? -actual.n : actual.n, d: actual.d * BigInt(10) ** BigInt(digits - 100) };
  assert.ok(compare(sub(actual, lower), { n: -slack.n, d: slack.d }) >= Z, 'Reference below lower bound');
  assert.ok(compare(sub(upper, actual), { n: -slack.n, d: slack.d }) >= Z, 'Reference above upper bound');
}
interface Case { id: string; parameters: CoupledPartitionParameters; confirmationPrecisionDigits: number; reference: Record<string, string | string[] | string[][]> }
const fixture = JSON.parse(readFileSync('scripts/fixtures/coupled-partition-oracle.json', 'utf8')) as { cases: Case[] };
for (const row of fixture.cases) test(`${row.id}: independent Decimal means, covariance, information and normalization`, () => {
  const r = model(row.parameters); assert.equal(r.fullyResolved, true);
  for (const [key, truth] of Object.entries(row.reference)) {
    if (key === 'probabilities') {
      const states = r.states.filter(s => s.allowed);
      (truth as string[]).forEach((p, i) => check(states[i].probability, p, row.confirmationPrecisionDigits));
    } else if (key === 'sourceHessian') (truth as string[][]).forEach((v, i) => v.forEach((p, j) => check(r.sourceHessian[i][j], p, row.confirmationPrecisionDigits)));
    else check((r as unknown as Record<string, PartitionObservation>)[key], truth as string, row.confirmationPrecisionDigits);
  }
});
test('zero mean spins do not imply independence; partition defect is not mutual information', () => {
  const r = model({ couplingEnergy: kB * Math.log(2), fieldEnergy: 0, temperature: 1, support: 'full-product' });
  assert.deepEqual(r.meanSpins.map(v => v.value), [0, 0]);
  assert.ok(Math.abs(r.meanAlignment.value! - 3 / 5) < 1e-14);
  assert.ok(Math.abs(r.mutualInformation.value! - (.8 * Math.log(1.6) + .2 * Math.log(.4))) < 1e-14);
  assert.ok(Math.abs(r.logFactorizationDefect.value! - Math.log(1.25)) < 1e-14);
});
test('support constraint creates dependence even at zero coupling, with genuine forbidden zeros', () => {
  const r = model({ couplingEnergy: 0, fieldEnergy: 0, temperature: 1, support: 'fixed-zero-sum' });
  assert.equal(r.factorizationIdentity, false);
  assert.equal(r.spinCovariance.value, -1);
  assert.ok(Math.abs(r.mutualInformation.value! - Math.log(2)) < 1e-14);
  assert.equal(r.logFactorizationDefect.sign, -1);
  assert.deepEqual(r.states.map(s => s.probability.value), [0, .5, .5, 0]);
  assert.deepEqual(r.sourceHessian.map(row => row.map(q => q.value)), [[0, 0], [0, 0]]);
});
test('zero-temperature crossings separate dimensionless covariance from physical response', () => {
  const r = model({ couplingEnergy: -1, fieldEnergy: 1, temperature: 0, support: 'full-product' });
  assert.equal(r.joint.anchorMultiplicity, 3);
  for (const q of r.sourceHessian.flat()) assert.ok(Math.abs(q.value! - 8 / 9) < 1e-14);
  assert.equal(r.responses.magnetizationToField.status, 'undefined-at-zero-temperature');
  const anti = model({ couplingEnergy: -1, fieldEnergy: 0, temperature: 0, support: 'full-product' });
  assert.equal(anti.responses.magnetizationToField.status, 'locally-constant-zero');
  assert.equal(anti.logFactorizationDefect.resolution, 'mathematical-infinity');
  assert.equal(anti.freeEnergyDefect.value, -1);
});
test('negative temperature reverses physical susceptibility without negating source covariance', () => {
  const r = model({ couplingEnergy: kB, fieldEnergy: .3 * kB, temperature: -1, support: 'full-product' });
  assert.equal(r.sourceHessian[1][1].sign, 1);
  assert.equal(r.responses.magnetizationToField.observation!.sign, -1);
  assert.equal(r.spinCovariance.sign, -1);
});
test('zero cross covariance does not establish a zero-temperature derivative', () => {
  const r = model({ couplingEnergy: 0, fieldEnergy: 0, temperature: 0, support: 'full-product' });
  assert.equal(r.sourceHessian[0][1].value, 0);
  assert.equal(r.responses.alignmentToField.status, 'undefined-at-zero-temperature');
});
test('Hamiltonian combines finite fields exactly even when 2h and energy overflow display range', () => {
  const r = model({ couplingEnergy: Number.MAX_VALUE, fieldEnergy: Number.MAX_VALUE, temperature: 0, support: 'full-product' });
  assert.equal(r.fullyResolved, true);
  assert.equal(r.joint.meanEnergy.value, null);
  assert.equal(r.joint.meanEnergy.sign, -1);
  assert.equal(r.states[3].probability.value, 1);
  assert.equal(r.freeEnergyDefect.value, -Number.MAX_VALUE);
});
test('weak dependence survives when mutual information is smaller than binary64 display range', () => {
  const r = model({ couplingEnergy: 1e-200, fieldEnergy: 0, temperature: 1, support: 'full-product' });
  assert.equal(r.mutualInformation.value, null);
  assert.equal(r.mutualInformation.sign, 1);
  assert.notEqual(r.mutualInformation.resolution, 'structural-zero');
  assert.ok(r.attempts.length > 1);
});
test('invalid inputs fail before evaluation', () => {
  for (const patch of [{ couplingEnergy: Infinity }, { temperature: NaN }, { fieldEnergy: -Infinity }, { support: 'anything' }])
    assert.throws(() => model({ couplingEnergy: 0, fieldEnergy: 0, temperature: 1, support: 'full-product', ...patch } as CoupledPartitionParameters), RangeError);
});
