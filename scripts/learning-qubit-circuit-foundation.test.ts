import { test } from 'node:test';
import assert from 'node:assert/strict';
import reference from './fixtures/qubit-circuit-independent.json';
import { f, zero, one } from '../src/lib/learning-quantum-hilbert-core';
import { quadratic, circuitScalar, circuitZero, circuitOne, circuitScalarWire, runQubitCircuit, inverseCircuit, circuitEqual, type CircuitGate } from '../src/lib/learning-qubit-circuit-exact';

for (const example of reference.circuitCases) {
  test(`exact dense reference: ${example.name}`, () => {
    const input = example.input.map(([re, im]) => circuitScalar(quadratic(f(BigInt(re))), quadratic(f(BigInt(im)))));
    const gates = example.gates as CircuitGate[];
    const result = runQubitCircuit(example.sites, gates, input);
    assert.deepEqual(result.prefixes.map(prefix => prefix.map(circuitScalarWire)), example.prefixes);
    assert.equal(result.normSquared[0].d === BigInt(1) ? String(result.normSquared[0].n) : `${result.normSquared[0].n}/${result.normSquared[0].d}`, example.normSquared);
    assert.equal(result.normSquared[1].n, BigInt(0));
    const restored = runQubitCircuit(example.sites, inverseCircuit(gates), result.state).state;
    assert.ok(restored.every((z, j) => circuitEqual(z, input[j])));
  });
}

test('chronological HZH and ZHH differ despite identical gate multisets', () => {
  const run = (names: ('H'|'Z')[]) => runQubitCircuit(1, names.map(gate=>({gate,target:0})), [circuitOne,circuitZero]).state;
  assert.ok(circuitEqual(run(['H','Z','H'])[1],circuitOne));
  assert.ok(circuitEqual(run(['Z','H','H'])[0],circuitOne));
});

test('algebraic input retains irrational norm terms exactly', () => {
  const input=[circuitScalar(quadratic(one,one)),circuitScalar(quadratic(zero),quadratic(one))];
  const result=runQubitCircuit(1,[{gate:'H',target:0},{gate:'T',target:0}],input);
  assert.deepEqual(result.normSquared,[f(BigInt(4)),f(BigInt(2))]);
});

test('input and gates are not mutated', () => {
  const input=[circuitOne,circuitZero];const gates: CircuitGate[]=[{gate:'H',target:0}];
  const before=JSON.stringify({input,gates},(_k,v)=>typeof v==='bigint'?String(v):v);
  runQubitCircuit(1,gates,input);
  assert.equal(JSON.stringify({input,gates},(_k,v)=>typeof v==='bigint'?String(v):v),before);
});

test('reject dimension, zero vector, unsupported gates, bad sites and excess budget', () => {
  for(const sites of [0,4,1.5,NaN])assert.throws(()=>runQubitCircuit(sites,[],[circuitOne,circuitZero]),RangeError);
  assert.throws(()=>runQubitCircuit(1,[],[circuitOne]),RangeError);
  assert.throws(()=>runQubitCircuit(1,[],[circuitZero,circuitZero]),RangeError);
  for(const op of [{gate:'H',target:1},{gate:'H',target:-1},{gate:'H',target:0.5},{gate:'CX',control:0,target:0},{gate:'SWAP',other:0,target:0},{gate:'bogus',target:0},{gate:'toString',target:0}])assert.throws(()=>runQubitCircuit(1,[op as CircuitGate],[circuitOne,circuitZero]),RangeError);
  assert.throws(()=>runQubitCircuit(1,Array.from({length:17},()=>({gate:'I',target:0})),[circuitOne,circuitZero]),RangeError);
});
