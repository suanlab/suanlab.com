import test from 'node:test';
import assert from 'node:assert/strict';
import {readRealTimePhaseRequest,realTimePhaseInputKey,realTimePhaseUnverifiedHeader} from '../src/lib/learning-real-time-phase-protocol';
import {realTimeChainDefault as input} from '../src/lib/learning-real-time-phase-input';

test('request metadata owns inputs andpreserves raw fraction spellings',()=>{
  const raw={version:1,requestId:7,input:{...input,time:'2/2'}},r=readRealTimePhaseRequest(raw);assert('request'in r);assert(Object.isFrozen(r.request.input));
  raw.input.time='1';assert.equal(r.request.input.mode,'chain');assert.equal('time'in r.request.input?r.request.input.time:null,'2/2');
  assert.notEqual(realTimePhaseInputKey(r.request.input),realTimePhaseInputKey(input));
});
test('request fields,IDs,accessors andprototype data fail closed',()=>{
  for(const id of [0,-0,-1,1.5,NaN,Infinity,Number.MAX_SAFE_INTEGER+1,'7'])assert('failure'in readRealTimePhaseRequest({version:1,requestId:id,input}));
  assert('failure'in readRealTimePhaseRequest({version:1,requestId:7,input,extra:true}));
  assert('failure'in readRealTimePhaseRequest(Object.create({version:1,requestId:7,input})));
  let calls=0;const getter={version:1,requestId:7,get input(){calls++;throw Error('getter');}};assert('failure'in readRealTimePhaseRequest(getter));assert.equal(calls,0);
  const invalid=readRealTimePhaseRequest({version:1,requestId:7,input:{...input,mass:'0'}});assert('failure'in invalid);assert.equal(invalid.failure.requestId,7);assert.equal(invalid.failure.error,'invalid-input');
});
test('unverified success header checks all attribution butdoes not certify mathematics',()=>{
  const response={version:1,requestId:7,ok:true,kind:'real-time-phase',input,inputKey:realTimePhaseInputKey(input),result:{input,forgedMath:true}};
  assert(realTimePhaseUnverifiedHeader(response,input,7),'this isonly a transport header guard');
  for(const patch of [{requestId:8},{version:2},{ok:false},{kind:'other'},{inputKey:'wrong'},{input:{...input,hbar:'2'}},{result:{input:{...input,hbar:'2'}}},{extra:1}])assert(!realTimePhaseUnverifiedHeader({...response,...patch},input,7));
  assert(!realTimePhaseUnverifiedHeader({...response,requestId:0},input,0));
  const revoked=Proxy.revocable({input},{});revoked.revoke();assert(!realTimePhaseUnverifiedHeader({...response,result:revoked.proxy},input,7));
  let calls=0;const result={get input(){calls++;throw Error('getter');}};assert(!realTimePhaseUnverifiedHeader({...response,result},input,7));assert.equal(calls,0);
});
