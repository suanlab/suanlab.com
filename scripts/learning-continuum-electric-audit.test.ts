import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { thermoelectricAt, type ThermoelectricParameters, type MultiphysicsObservation } from '../src/lib/learning-continuum-multiphysics';
interface Reference { value: number | null; sign: number; logMagnitude: string | null }
const fixture = JSON.parse(readFileSync('scripts/fixtures/continuum-electric-audit-independent.json', 'utf8')) as {
  cases: { label: string; parameters: ThermoelectricParameters; xi: number; expected: Record<string, Reference>; leftCoolingCOP: Reference | null; electricalRegime: string }[];
};
function compare(actual: MultiphysicsObservation, expected: Reference, label: string) {
  assert.equal(actual.sign, expected.sign, label + ' sign');
  assert.equal(actual.value, expected.value, label + ' correctly rounded rational observation');
  if (expected.logMagnitude === null) assert.equal(actual.logMagnitude, null, label);
  else {
    assert.notEqual(actual.logMagnitude, null, label);
    assert.ok(Math.abs(actual.logMagnitude! - Number(expected.logMagnitude)) < 2e-10, label + ' log');
  }
}
assert.equal(fixture.cases.length, 36);
for (const row of fixture.cases) test('independent thermoelectric polynomial: ' + row.label, () => {
  const result = thermoelectricAt(row.parameters, row.xi);
  for (const [name, expected] of Object.entries(row.expected)) compare(result[name as keyof typeof result] as MultiphysicsObservation, expected, name);
  assert.equal(result.electricalRegime, row.electricalRegime);
  if (row.leftCoolingCOP === null) assert.equal(result.leftCoolingCOP, null);
  else { assert.notEqual(result.leftCoolingCOP, null); compare(result.leftCoolingCOP!, row.leftCoolingCOP, 'COP'); }
  assert.doesNotThrow(() => JSON.stringify(result), 'internal exact arithmetic stays out of public results');
});
