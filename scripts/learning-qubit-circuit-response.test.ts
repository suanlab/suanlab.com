import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { qubitCircuitReport } from '../src/lib/learning-qubit-circuit-report';
import { verifyQubitCircuitReport } from '../src/lib/learning-qubit-circuit-validator';
import { f,zero,one,quadratic,complex,Z,O,times,qAdd,qEqual,circuitScalarWire,mWire,partialTrace,type C } from '../src/lib/learning-qubit-channel-algebra';
import { channelFamily,channelReport,applyWithReference } from '../src/lib/learning-qubit-channel';
import type { CircuitGate } from '../src/lib/learning-qubit-circuit-exact';

type WireScalar={real:string[];imaginary:string[]};
type Example={name:string;sites:number;gates:CircuitGate[];input:WireScalar[];expected:ReturnType<typeof qubitCircuitReport>};
const references=JSON.parse(readFileSync('scripts/fixtures/qubit-circuit-response-independent.json','utf8')) as {cases:Example[]};
const fraction=(s:string)=>{const [n,d='1']=s.split('/');return f(BigInt(n),BigInt(d));};
const scalar=(w:WireScalar):C=>complex([fraction(w.real[0]),fraction(w.real[1])],[fraction(w.imaginary[0]),fraction(w.imaginary[1])]);
const real=(n:number,d=1)=>circuitScalarWire(complex(quadratic(f(BigInt(n),BigInt(d)))));
for(const example of references.cases)test(`circuit prefix observables and dense independent check: ${example.name}`,()=>{
  const input=example.input.map(scalar),actual=qubitCircuitReport(example.sites,example.gates,input);
  assert.deepEqual(actual,example.expected);
  assert.deepEqual(verifyQubitCircuitReport(example.sites,example.gates,input,actual),{valid:true,errors:[]});
  assert.deepEqual(JSON.parse(JSON.stringify(actual)),actual);
  for(const prefix of actual.prefixes){
    assert.ok(qEqual(prefix.probabilities.reduce((s,w)=>qAdd(s,scalar(w).re),quadratic(zero)),quadratic(one)));
    assert.deepEqual(prefix.normResidual,real(0));assert.deepEqual(prefix.purity,real(1));
  }
  const altered=structuredClone(actual),entry=altered.prefixes[altered.prefixes.length-1].singleQubits[0].density[0][0];
  entry.real[0]=entry.real[0]==='0'?'1':'0';
  assert.equal(verifyQubitCircuitReport(example.sites,example.gates,input,altered).valid,false);
});

test('reference inventory includes every gate, all wire positions, both controls and maximum depth',()=>{
  assert.equal(references.cases.length,91);assert.equal(references.cases.reduce((n,c)=>n+c.expected.prefixes.length,0),216);
  const gates=new Set(references.cases.flatMap(c=>c.gates.map(g=>g.gate)));
  assert.equal(gates.size,12);
  for(const n of [1,2,3])assert.ok(references.cases.some(c=>c.sites===n&&c.gates.length===0));
  assert.ok(references.cases.some(c=>c.sites===3&&c.gates.length===16));
});
test('squared fidelity differs from root overlap and is explicitly labeled',()=>{
  const r=qubitCircuitReport(1,[{gate:'H',target:0}],[O,Z]);
  assert.equal(r.fidelityConvention,'squared-pure-state-overlap');assert.deepEqual(r.fidelityWithInput,real(1,2));
  assert.deepEqual(r.overlapWithInput,{real:['0','1/2'],imaginary:['0','0']});
});
test('global phase changes amplitudes while preserving density and squared fidelity',()=>{
  const r=qubitCircuitReport(1,[{gate:'T',target:0}],[Z,O]);
  assert.equal(r.sameAmplitudesAsInput,false);assert.equal(r.samePhysicalStateAsInput,true);
  assert.deepEqual(r.fidelityWithInput,real(1));assert.deepEqual(r.prefixes[0].density,r.prefixes[1].density);
});
test('identical populations can hide opposite X and Y expectations',()=>{
  const p=qubitCircuitReport(1,[],[O,O]).prefixes[0],m=qubitCircuitReport(1,[],[O,complex(quadratic(f(BigInt(-1))))]).prefixes[0];
  assert.deepEqual(p.probabilities,m.probabilities);assert.deepEqual(p.singleQubits[0].bloch[0],real(1));assert.deepEqual(m.singleQubits[0].bloch[0],real(-1));
  const a=qubitCircuitReport(1,[],[O,complex(quadratic(zero),quadratic(one))]).prefixes[0];
  assert.deepEqual(a.singleQubits[0].bloch[1],real(1));
});
test('Bell local mixedness coexists with a pure joint state and nonzero correlations',()=>{
  const r=qubitCircuitReport(2,[{gate:'H',target:0},{gate:'CX',control:0,target:1}],[O,Z,Z,Z]).prefixes.at(-1)!;
  assert.deepEqual(r.purity,real(1));
  for(const q of r.singleQubits){assert.deepEqual(q.purity,real(1,2));assert.deepEqual(q.bloch,[real(0),real(0),real(0)]);}
  assert.deepEqual(r.pauliPairs[0].expectations,[[real(1),real(0),real(0)],[real(0),real(-1),real(0)],[real(0),real(0),real(1)]]);
  assert.deepEqual(r.pauliPairs[0].connected,r.pauliPairs[0].expectations);
});
test('product inputs have zero connected correlations',()=>{
  const first=[O,complex(quadratic(zero),quadratic(one))],second=[O,O];
  const input=first.flatMap(a=>second.map(b=>times(a,b))),r=qubitCircuitReport(2,[],input).prefixes[0];
  assert.ok(r.pauliPairs[0].connected.flat().every(v=>JSON.stringify(v)===JSON.stringify(real(0))));
  assert.deepEqual(r.singleQubits.map(q=>q.purity),[real(1),real(1)]);
});
test('a circuit density connects to channel and reference calculations without dropping coherence',()=>{
  const circuit=qubitCircuitReport(2,[{gate:'H',target:0},{gate:'CX',control:0,target:1}],[O,Z,Z,Z]).prefixes.at(-1)!;
  const rho=circuit.density.map(row=>row.map(scalar)),local=circuit.singleQubits[1].density.map(row=>row.map(scalar));
  const damping=channelFamily('damping'),noisy=applyWithReference(damping,rho),channel=channelReport(damping,local);
  assert.deepEqual(mWire(noisy),channel.bellOutput);
  assert.deepEqual(mWire(partialTrace(noisy,2,2,'second')),channel.output);
  assert.deepEqual(mWire(partialTrace(noisy,2,2,'first')),circuit.singleQubits[0].density);
});
test('strict checking binds observables to the computation even when probabilities match',()=>{
  const gates:CircuitGate[]=[{gate:'H',target:0}],input=[O,Z],r=qubitCircuitReport(1,gates,input);
  const other=qubitCircuitReport(1,[{gate:'H',target:0},{gate:'T',target:0}],input);
  assert.deepEqual(other.prefixes.at(-1)!.probabilities,r.prefixes.at(-1)!.probabilities);
  const changed=structuredClone(r);changed.prefixes[1].singleQubits[0].bloch[0]=real(0);
  assert.equal(verifyQubitCircuitReport(1,gates,input,changed).valid,false);
  for(const key of Object.keys(r)){
    const missing={...r} as Record<string,unknown>;delete missing[key];
    assert.equal(verifyQubitCircuitReport(1,gates,input,missing).valid,false,key);
  }
  assert.equal(verifyQubitCircuitReport(1,gates,input,{...r,extra:true}).valid,false);
});
test('exact report does not mutate the raw input or chronological gate list',()=>{
  const input=[complex([one,one]),complex(quadratic(zero),quadratic(one))],gates:CircuitGate[]=[{gate:'Tdg',target:0},{gate:'H',target:0}];
  const snapshot=(x:unknown)=>JSON.stringify(x,(_key,v)=>typeof v==='bigint'?String(v):v),before=snapshot({input,gates});
  qubitCircuitReport(1,gates,input);assert.equal(snapshot({input,gates}),before);
});
test('invalid circuit inputs cannot produce a verified report',()=>{
  assert.throws(()=>qubitCircuitReport(1,[],[Z,Z]),RangeError);
  assert.throws(()=>qubitCircuitReport(4,[],Array.from({length:16},()=>O)),RangeError);
  assert.throws(()=>qubitCircuitReport(1,[{gate:'CX',control:0,target:0}],[O,Z]),RangeError);
  assert.throws(()=>qubitCircuitReport(1,Array.from({length:17},()=>({gate:'H',target:0})),[O,Z]),RangeError);
  assert.equal(verifyQubitCircuitReport(1,[],[Z,Z],{}).valid,false);
  assert.equal(verifyQubitCircuitReport(4,[],[O,Z],{}).valid,false);
  assert.equal(verifyQubitCircuitReport(1,[{gate:'unknown',target:0} as unknown as CircuitGate],[O,Z],{}).valid,false);
});
