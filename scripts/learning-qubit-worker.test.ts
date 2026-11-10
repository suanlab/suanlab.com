import { test } from 'node:test';
import assert from 'node:assert/strict';
import { qubitWorkerCases as cases,qubitWorkerRepresentatives as reps } from './learning/qubit-worker-cases';
import { readQubitInput,parseQubitInputDraft as draft,parseQubitStoredInput as stored,qubitInputKey,qubitData } from '../src/lib/learning-qubit-input';
import { evaluateQubitRequest as calculate } from '../src/lib/learning-qubit-evaluator';
import { evaluateQubitValidationRequest as validate } from '../src/lib/learning-qubit-validation-evaluator';
import { snapshotQubitWire,snapshotQubitWireInTasks,QubitWireLimit,QubitCopyCancelled } from '../src/lib/learning-qubit-wire';
function alterFirstScalar(value:unknown):boolean {
  if(!value||typeof value!=='object')return false;
  const record=value as Record<string,unknown>;
  if(Array.isArray(record.real)&&Array.isArray(record.imaginary)){record.real[0]=record.real[0]==='0'?'1':'0';return true;}
  return Object.values(value).some(alterFirstScalar);
}

for(const [index,c] of cases.entries())test(`owned Worker protocol and independent mathematics: ${c.label}`,()=>{
  const input=readQubitInput(c.input);assert(input);assert(Object.isFrozen(input));
  assert.deepEqual(draft(JSON.stringify(input)),input);assert.deepEqual(stored(JSON.stringify({version:1,value:input})),input);
  const id=index+1,started:unknown[]=[],response=calculate({version:1,requestId:id,input},m=>started.push(m));
  assert(response.ok);assert.deepEqual(response.result,c.expected);assert.equal(response.inputKey,qubitInputKey(input));
  assert.deepEqual(started,[{version:1,requestId:id,phase:'started',kind:input.kind}]);
  const receipt=validate({version:1,requestId:id,input,response});assert.equal(receipt.phase,'validated');
  assert(Object.isFrozen(response)&&Object.isFrozen(response.result));
  const changed=structuredClone(response);assert(alterFirstScalar(changed.result));
  assert.equal(validate({version:1,requestId:id,input,response:changed}).phase,'rejected');
});
test('inventory retains all independently authored preparations',()=>assert.equal(cases.length,595));
test('rational strings reject approximate numbers, zero denominators, exponent syntax and excess digits',()=>{
  const initial=structuredClone(reps[0].input);assert(initial.kind==='circuit');
  for(const value of ['01','-0','1/0','1/-2','0.5','1e-4','1'.repeat(33),1,NaN,null]){
    const input=structuredClone(initial) as unknown as {input:{real:unknown[]}[]};input.input[0].real[0]=value;
    assert.equal(readQubitInput(input),null,String(value));
  }
});
test('JSON drafts reject duplicate keys, lossy integer tokens, excessive size and wrong versions',()=>{
  const raw=JSON.stringify(reps[0].input);
  assert.equal(draft(raw.replace('"sites":2','"sites":2.0000000000000001')),null);
  assert.equal(draft(raw.replace('"sites":2','"sites":2e0')),null);
  assert.equal(draft(raw.replace('"kind":','"kind":"circuit","kind":')),null);
  assert.equal(draft(raw.replace('"kind":','"k\\u0069nd":"circuit","kind":')),null);
  assert.equal(draft(' '.repeat(65537)+raw),null);assert.equal(stored(JSON.stringify({version:2,value:reps[0].input})),null);
});
test('owned structural readers never invoke getters or get traps',()=>{
  let calls=0;const hostile={get kind(){calls++;return 'circuit';}};
  assert.equal(readQubitInput(hostile),null);
  const proxy=new Proxy(reps[0].input,{get(){calls++;throw Error('get trap');}});
  assert(readQubitInput(proxy));assert.equal(calls,0);
  assert.equal(qubitData(Object.assign(Object.create({}),{a:1}),['a']),null);
  const a:unknown[]=[];a.length=2;assert.equal(readQubitInput({kind:'circuit',sites:1,gates:[],input:a}),null);
  const prototypeKey=JSON.parse('{"__proto__":{"polluted":true}}');const owned=qubitData(prototypeKey,['__proto__']);assert(owned);
  assert.equal(Object.getPrototypeOf(owned),Object.prototype);assert(Object.hasOwn(owned,'__proto__'));
});
test('request attribution cannot be replaced by another valid ID, kind, key or preparation',()=>{
  const input=reps[0].input,response=calculate({version:1,requestId:1,input});assert(response.ok);
  for(const altered of [{...response,requestId:2},{...response,kind:'channel'},{...response,inputKey:'other'},{...response,input:reps[1].input}])assert.equal(validate({version:1,requestId:1,input,response:altered}).phase,'rejected');
  assert.equal(calculate({version:1,requestId:0,input}).ok,false);
});
test('mathematically invalid states fail inside the Worker, including zero vectors',()=>{
  const input=structuredClone(reps[0].input);assert(input.kind==='circuit');
  const zero={real:['0','0'],imaginary:['0','0']};
  const result=calculate({version:1,requestId:1,input:{...input,input:input.input.map(()=>zero)}});
  assert.deepEqual(result,{version:1,requestId:1,ok:false,error:'invalid-input'});
  const env=structuredClone(reps[2].input);assert(env.kind==='environment');
  assert.equal(readQubitInput({...env,parameter:'4/3'}),null);
  const invalid=calculate({version:1,requestId:2,input:{...env,state:[[zero,zero],[zero,zero]]}});assert(!invalid.ok&&invalid.error==='invalid-input');
});
test('bounded snapshots reject getters, cycles, special prototypes, holes and resource exhaustion',()=>{
  let calls=0;assert.throws(()=>snapshotQubitWire({get value(){calls++;return 1;}}));assert.equal(calls,0);
  const cyclic:Record<string,unknown>={};cyclic.next=cyclic;assert.throws(()=>snapshotQubitWire(cyclic));
  for(const value of [new Date(),new Map(),Array(2),undefined,Infinity,BigInt(1)])assert.throws(()=>snapshotQubitWire(value));
  assert.throws(()=>snapshotQubitWire('x'.repeat(65537)),QubitWireLimit);
  assert.throws(()=>snapshotQubitWire(Array.from({length:32},()=> 'x'.repeat(65536))),QubitWireLimit);
});
test('asynchronous ownership yields and stops when a request is cancelled',async()=>{
  let active=true,pauses=0;
  await assert.rejects(snapshotQubitWireInTasks({values:Array.from({length:100},(_,i)=>i)},{nodesPerTask:1,active:()=>active,pause:async()=>{pauses++;active=false;}}),QubitCopyCancelled);
  assert.equal(pauses,1);
});
