import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdirSync,readFileSync,writeFileSync} from 'node:fs';
import path from 'node:path';
import ts from 'typescript';
import {verifyRealTimePhaseResponse} from '../src/lib/learning-real-time-phase-validation';
import {calculateRealTimePhase} from '../src/lib/learning-real-time-phase';
import {realTimePhaseExamples,realTimeChainDefault,realTimePacketDefault,realTimeFoldDefault,type RealTimePhaseInput} from '../src/lib/learning-real-time-phase-input';
import {realTimePhaseInputKey,realTimePhaseUnverifiedHeader,type RealTimePhaseUnverifiedSuccess} from '../src/lib/learning-real-time-phase-protocol';
import {snapshotQubitWire,QubitWireLimit} from '../src/lib/learning-qubit-wire';
import {CheckerRational as Q} from '../src/lib/learning-real-time-validation-rational';

const root='.runtime/learning/real-time-whole-response';mkdirSync(root,{recursive:true});
function candidate(input:RealTimePhaseInput,id=1) {
  const request={version:1,requestId:id,input},response:RealTimePhaseUnverifiedSuccess={version:1,requestId:id,ok:true,kind:'real-time-phase',input,inputKey:realTimePhaseInputKey(input),result:calculateRealTimePhase(input)};
  return{request,response};
}
const cache=new Map<string,ReturnType<typeof candidate>>();
const measurements:{id:string;input:RealTimePhaseInput;kind:string;calculationMilliseconds:number;validationMilliseconds:number;bytes:number;characters:number;maxString:number;nodes:number}[]=[];
for(const [index,row]of realTimePhaseExamples.entries())test('whole producer envelope andallmathematics: '+row.id,()=>{
  const started=performance.now(),data=candidate(row.input,index+1),calculationMilliseconds=performance.now()-started;cache.set(row.id,data);
  const validationStarted=performance.now(),receipt=verifyRealTimePhaseResponse(data.request,data.response),validationMilliseconds=performance.now()-validationStarted;
  assert(receipt,row.id);assert.equal(receipt.fullResponseValidated,true);assert.equal(receipt.phase,'validated');assert.equal(receipt.requestId,index+1);
  assert.equal(receipt.inputKey,realTimePhaseInputKey(row.input));assert(Object.isFrozen(receipt)&&Object.isFrozen(receipt.response)&&Object.isFrozen(receipt.response.result));
  assert.notEqual(receipt.response,data.response);assert.notEqual(receipt.response.result,data.response.result);
  assert.deepEqual(receipt.response,data.response);assert.deepEqual(snapshotQubitWire(receipt),receipt);
  let characters=0,maxString=0,nodes=0;const walk=(x:unknown):void=>{nodes++;if(typeof x==='string'){characters+=x.length;maxString=Math.max(maxString,x.length);}else if(Array.isArray(x))x.forEach(walk);else if(x&&typeof x==='object')Object.values(x).forEach(walk);};walk(receipt.response);
  assert(characters<=2000000&&maxString<=65536&&nodes<=100000);
  measurements.push({id:row.id,input:row.input,kind:receipt.response.result.kind,calculationMilliseconds,validationMilliseconds,bytes:Buffer.byteLength(JSON.stringify(receipt.response)),characters,maxString,nodes});
  writeFileSync(root+`/profiles-node${process.versions.node.split('.')[0]}.json`,JSON.stringify(measurements,null,2)+'\n');
});

type Segment=string|number;
function leaves(x:unknown,base:Segment[]=[]):{path:Segment[];value:unknown}[] {
  if(x===null||typeof x!=='object')return[{path:base,value:x}];
  return Object.entries(x).flatMap(([key,value])=>leaves(value,[...base,Array.isArray(x)?Number(key):key]));
}
function mutate<T>(value:T,segments:Segment[],next:unknown):T {
  const copy=structuredClone(value);let at=copy as Record<Segment,unknown>;
  for(const key of segments.slice(0,-1))at=at[key]as Record<Segment,unknown>;at[segments.at(-1)!]=next;return copy;
}
function wrong(leaf:{path:Segment[];value:unknown}):unknown {
  if(leaf.path.at(-1)==='terms')return 0;
  if(typeof leaf.value==='string') {
    if(/^(?:0|-?[1-9]\d*)(?:\/[1-9]\d*)?$/.test(leaf.value)) {
      const packed=leaf.path.includes('upperRows');
      return Q.input(leaf.value).add(Q.of(packed?BigInt(1)<<BigInt(520):BigInt(1024))).text();
    }
    return'wrong-'+leaf.value;
  }
  return typeof leaf.value==='number'?leaf.value+1:typeof leaf.value==='boolean'?!leaf.value:'not-null';
}
for(const id of ['negative-mode-hand','caustic-compatible','fixed-width-free','merging-stationary-points'])test('reject every primitive whole-envelope corruption: '+id,()=>{
  const data=cache.get(id)!;let rejected=0;
  for(const leaf of leaves(data.response)) {
    assert.equal(verifyRealTimePhaseResponse(data.request,mutate(data.response,leaf.path,wrong(leaf))),null,leaf.path.join('.'));rejected++;
  }
  writeFileSync(root+`/mutations-${id}-node${process.versions.node.split('.')[0]}.json`,JSON.stringify({id,rejected},null,2)+'\n');
});

test('request attribution covers every full example before mathematical acceptance',()=>{
  let rejected=0;
  for(const row of realTimePhaseExamples) {
    const data=cache.get(row.id)!;
    const alternatives=[{...data.response,requestId:data.request.requestId+1000},{...data.response,inputKey:'foreign'},
      {...data.response,input:row.input.mode==='fold'?realTimeChainDefault:realTimeFoldDefault}];
    for(const value of alternatives){assert.equal(verifyRealTimePhaseResponse(data.request,value),null,row.id);rejected++;}
  }
  assert.equal(rejected,96);
});
test('an attribution header orpartial certificate cannot substitute for whole mathematics',()=>{
  const data=cache.get('harmonic-four')!,forged=mutate(data.response,['result','wave','unregulated','unitPhase','real','lower'],'1');
  assert(realTimePhaseUnverifiedHeader(forged,data.request.input,data.request.requestId));
  assert.equal(verifyRealTimePhaseResponse(data.request,forged),null);
  const partial={...data.response,result:{input:data.request.input,scope:'finite-exact-core-only',fullResponseValidated:false,wire:(data.response.result as{exact:unknown}).exact}};
  assert(realTimePhaseUnverifiedHeader(partial,data.request.input,data.request.requestId));assert.equal(verifyRealTimePhaseResponse(data.request,partial),null);
  const receipt=verifyRealTimePhaseResponse(data.request,data.response);assert(receipt);
  assert.equal(verifyRealTimePhaseResponse(data.request,receipt),null,'receipt is not a producer envelope');
  assert.equal(verifyRealTimePhaseResponse(data.request,{...data.response,fullResponseValidated:true}),null);
});
test('top-level mode,shape andchain wave precision metadata are checked',()=>{
  for(const id of ['harmonic-four','fixed-width-free','merging-stationary-points']) {
    const data=cache.get(id)!;
    assert.equal(verifyRealTimePhaseResponse(data.request,mutate(data.response,['result','kind'],'other-mode')),null);
    const extra=structuredClone(data.response);(extra.result as Record<string,unknown>).extra=true;
    assert.equal(verifyRealTimePhaseResponse(data.request,extra),null);
  }
  const data=cache.get('harmonic-four')!;
  for(const key of ['workingBits','observationBits'])assert.equal(verifyRealTimePhaseResponse(data.request,mutate(data.response,['result','wave',key],0)),null);
  const missing=structuredClone(data.response);delete(missing.result as Record<string,unknown>).formalConnectedResponse;
  assert.equal(verifyRealTimePhaseResponse(data.request,missing),null);
});
test('self-consistent foreign models cannot be rebound by patching input labels',()=>{
  for(const [input,foreign]of [
    [realTimeChainDefault,{...realTimeChainDefault,mass:'2'}],
    [realTimePacketDefault,{...realTimePacketDefault,hbar:'1'}],
    [realTimeFoldDefault,{...realTimeFoldDefault,control:'2'}],
  ]as[RealTimePhaseInput,RealTimePhaseInput][]) {
    const data=candidate(foreign,701),result=data.response.result as Record<string,unknown>;
    data.request.input=input;data.response.input=input;data.response.inputKey=realTimePhaseInputKey(input);result.input=input;
    if(input.mode!=='fold')(result.exact as Record<string,unknown>).input=input;
    assert(realTimePhaseUnverifiedHeader(data.response,input,701));assert.equal(verifyRealTimePhaseResponse(data.request,data.response),null);
  }
});
test('maximum32interval models reject late fields andcoordinated phase sign corruption',()=>{
  const targets:{id:string;path:Segment[]}[]=[
    {id:'maximum-mixed-coprime',path:['result','exact','regulator','mean',30,'imaginary']},
    {id:'maximum-mixed-coprime',path:['result','inverse','upperRows',30,0,'lower']},
    {id:'maximum-mixed-coprime',path:['result','regulatedConnectedResponse','upperRows',30,0,'imaginary','upper']},
    {id:'inverted-packet',path:['result','wave','samples',64,'density','upper']},
    {id:'minimum-width',path:['result','wave','samples',64,'value','real','lower']},
    {id:'maximum-negative-physical-control',path:['result','complexStationaryPoints',1,'contributesToRealContour']},
  ];
  for(const item of targets){const data=cache.get(item.id)!;let value:unknown=data.response;
    for(const key of item.path)value=(value as Record<Segment,unknown>)[key];
    assert.equal(verifyRealTimePhaseResponse(data.request,mutate(data.response,item.path,wrong({path:item.path,value}))),null,item.id+' '+item.path.join('.'));
  }
  const data=cache.get('regulated-free-branch')!,forged=structuredClone(data.response);
  const regulated=(forged.result as{wave:{regulated:{unitPhase:{real:{lower:string;upper:string};imaginary:{lower:string;upper:string}};scaledValue:{real:{lower:string;upper:string};imaginary:{lower:string;upper:string}};value:{real:{lower:string;upper:string};imaginary:{lower:string;upper:string}}}}}).wave.regulated;
  for(const c of [regulated.unitPhase,regulated.scaledValue,regulated.value])for(const key of ['real','imaginary']as const){const x=c[key];c[key]={lower:Q.input(x.upper).neg().text(),upper:Q.input(x.lower).neg().text()};}
  assert.equal(verifyRealTimePhaseResponse(data.request,forged),null);
});
test('owned immutable receipts survive caller mutation without trusting serialized claims',()=>{
  const data=cache.get('fixed-width-free')!,copy=structuredClone(data.response),receipt=verifyRealTimePhaseResponse(data.request,copy);assert(receipt);
  const before=JSON.stringify(receipt.response);
  (copy.result as Record<string,unknown>).wave={forged:true};copy.inputKey='tampered';
  assert.equal(JSON.stringify(receipt.response),before);assert.equal(Reflect.set(receipt.response.result,'kind','forged'),false);
  assert.equal(verifyRealTimePhaseResponse(data.request,JSON.parse(JSON.stringify(receipt))),null);
  assert(verifyRealTimePhaseResponse(data.request,receipt.response),'rechecking actual full mathematics is allowed');
});
test('invalid requests,raw fraction spellings andstale request IDs fail closed',()=>{
  const data=cache.get('equivalent-spelling')!;
  assert.equal(verifyRealTimePhaseResponse({...data.request,input:{...realTimeChainDefault}},data.response),null);
  for(const id of [0,-0,-1,NaN,Infinity,0.5,Number.MAX_SAFE_INTEGER+1])assert.equal(verifyRealTimePhaseResponse({...data.request,requestId:id},data.response),null);
  assert.equal(verifyRealTimePhaseResponse({...data.request,version:2},data.response),null);
  assert.equal(verifyRealTimePhaseResponse({...data.request,extra:true},data.response),null);
  assert.equal(verifyRealTimePhaseResponse(data.request,{version:1,requestId:data.request.requestId,ok:false,error:'calculation-failed'}),null);
});
test('getters,prototypes,symbols,cycles andcopy budgets cannot bypass whole validation',()=>{
  const data=cache.get('harmonic-four')!;let calls=0;
  const getter={...data.response};Object.defineProperty(getter,'result',{get(){calls++;throw Error('getter');}});
  assert.equal(verifyRealTimePhaseResponse(data.request,getter),null);assert.equal(calls,0);
  const requestGetter={...data.request};Object.defineProperty(requestGetter,'input',{get(){calls++;throw Error('getter');}});
  assert.equal(verifyRealTimePhaseResponse(requestGetter,data.response),null);assert.equal(calls,0);
  assert.equal(verifyRealTimePhaseResponse(data.request,Object.create(data.response)),null);
  assert.equal(verifyRealTimePhaseResponse(data.request,{...data.response,[Symbol('hidden')]:true}),null);
  const cycle=structuredClone(data.response);(cycle.result as Record<string,unknown>).extra=cycle;
  assert.equal(verifyRealTimePhaseResponse(data.request,cycle),null);
  const revoked=Proxy.revocable({},{});revoked.revoke();assert.equal(verifyRealTimePhaseResponse(data.request,revoked.proxy),null);
  for(const probe of ['x'.repeat(65537),Array.from({length:257},()=>0),Array.from({length:40},()=> 'x'.repeat(60000))]) {
    const value={...data.response,probe};assert.throws(()=>snapshotQubitWire(value),QubitWireLimit);
    assert.equal(verifyRealTimePhaseResponse(data.request,value),null);
  }
});
function dependencyGraph(start:string,allowed:Set<string>) {
  const queue=[start],visited=new Set<string>();
  while(queue.length){const file=queue.shift()!;if(visited.has(file))continue;visited.add(file);assert(allowed.has(file),file);
    const source=ts.createSourceFile(file,readFileSync(file,'utf8'),ts.ScriptTarget.Latest,true);
    for(const statement of source.statements)if(ts.isImportDeclaration(statement)&&!statement.importClause?.isTypeOnly&&ts.isStringLiteral(statement.moduleSpecifier)) {
      const spec=statement.moduleSpecifier.text;if(spec.startsWith('.'))queue.push(path.normalize(path.join(path.dirname(file),spec+'.ts')));
    }
  }return visited;
}
test('whole checker runtime graph contains only independent gates andreader-safe protocol',()=>{
  const names=['learning-real-time-phase-validation','learning-real-time-phase-protocol','learning-real-time-phase-input','learning-qubit-wire',
    'learning-real-time-matrix-check','learning-real-time-unregulated-wave-check','learning-real-time-regulated-wave-check','learning-real-time-packet-wave-check',
    'learning-real-time-fold-check','learning-real-time-exact-core-check','learning-real-time-validation-rational','learning-real-time-checker-interval',
    'learning-real-time-observation-check','learning-real-time-checker-airy','learning-real-time-checker-airy-origin'];
  const allowed=new Set(names.map(x=>'src/lib/'+x+'.ts'));assert.equal(dependencyGraph('src/lib/learning-real-time-phase-validation.ts',allowed).size,15);
});
test('reader protocol imports neither the checker norproducer computation',()=>{
  const allowed=new Set(['src/lib/learning-real-time-phase-protocol.ts','src/lib/learning-real-time-phase-input.ts']);
  assert.equal(dependencyGraph('src/lib/learning-real-time-phase-protocol.ts',allowed).size,2);
});
