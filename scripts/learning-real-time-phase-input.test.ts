import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {readRealTimePhaseInput as read, realTimeChainDefault as chain, realTimePacketDefault as packet,
  realTimeFoldDefault as fold, realTimePhaseExamples} from '../src/lib/learning-real-time-phase-input';

test('all three modes retain independent physical controls, full12digit inputs and32intervals',()=>{
  assert.equal(realTimePhaseExamples.length,32);
  for(const {id,input} of realTimePhaseExamples) {
    const copy=read(input);assert.deepEqual(copy,input,id);assert.notEqual(copy,input);assert(Object.isFrozen(copy));
  }
  for(const field of ['time','mass','hbar']) {
    assert(read({...chain,[field]:'1/999999999999'}));assert(read({...chain,[field]:'999999999999/62500000000'}));
    for(const bad of ['0','-1','16/0','17','999999999999/62499999999','1000000000000/999999999999'])assert.equal(read({...chain,[field]:bad}),null);
  }
  assert(read({...chain,intervals:32,probe:32,omegaSquared:'-16',damping:'16'}));
  assert(read({...fold,control:'16',hbar:'1/999999999999'}));
  assert(read({...fold,control:'-16',hbar:'1/999999999999'}));
  for(const widthMode of ['fixed-variance','hbar-scaled-variance'])assert(read({...packet,widthMode,width:'1/999999999999'}));
});

test('invalid fields and coercible inputs cannot cross mode or numeric boundaries',()=>{
  for(const bad of ['', '01','+1','1.0','1e0',' 1','1 ','1/01','1/0','-0','-0/7','0/0','1/2/3',1,NaN,Infinity,null,{},new String('1')])assert.equal(read({...fold,hbar:bad}),null);
  for(const invalid of [1,33,-0,2.5,NaN,Infinity,'4'])assert.equal(read({...chain,intervals:invalid}),null);
  for(const invalid of [-1,5,-0,1.5,NaN,Infinity,'2'])assert.equal(read({...chain,probe:invalid}),null);
  for(const invalid of ['0','-1'])assert.equal(read({...packet,width:invalid}),null);
  assert.equal(read({...packet,widthMode:'coherent'}),null);
  assert.equal(read({...chain,damping:'-1'}),null);
  assert.equal(read({...fold,control:'17'}),null);
  assert.equal(read({...chain,mode:'packet'}),null);
  assert.equal(read({...packet,mode:'chain'}),null);
  assert.equal(read({...fold,mode:'chain'}),null);
  assert.equal(read({...fold,extra:true}),null);
  assert.equal(read({...fold,[Symbol('hidden')]:true}),null);
  for(const key of Object.keys(chain)){const value={...chain};delete (value as Record<string,unknown>)[key];assert.equal(read(value),null);}
});

test('owned snapshots reject accessors, inherited fields and traps without invoking field getters',()=>{
  let calls=0;
  for(const base of [chain,packet,fold])for(const key of Object.keys(base)) {
    const value={...base};Object.defineProperty(value,key,{get(){calls++;throw Error('getter');},enumerable:true});
    assert.equal(read(value),null);
  }
  assert.equal(calls,0);
  assert.equal(read(Object.create(fold)),null);
  assert.equal(read(Object.assign(Object.create(null),fold)),null);
  assert.equal(read(new Proxy({...fold},{ownKeys(){throw Error('trap');}})),null);
  const revoked=Proxy.revocable({...fold},{});revoked.revoke();assert.equal(read(revoked.proxy),null);
  const value={...chain},owned=read(value);value.start='16';assert.equal(owned?.mode,'chain');assert.equal(owned&&'start'in owned?owned.start:null,'0');
  for(const invalid of [null,undefined,[],true,()=>fold,new Date()])assert.equal(read(invalid),null);
});

test('input module does not import exact algebra into the reader',()=>{
  const source=readFileSync('src/lib/learning-real-time-phase-input.ts','utf8');
  assert(!/^import /m.test(source));assert(!source.includes('BigInt('));
});
