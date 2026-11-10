import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync,mkdirSync,writeFileSync } from 'node:fs';
import { calculateSaddleFluctuation,type SaddleFluctuationResult } from '../src/lib/learning-saddle-fluctuation';
import { validateSaddleFluctuationResult } from '../src/lib/learning-saddle-fluctuation-validation';
import { saddleFluctuationDefault,saddleFluctuationExamples,type SaddleFluctuationInput } from '../src/lib/learning-saddle-fluctuation-input';
import { Q,compare,exact,integer,interval,root,rational,green,verifyObservation,readBounded } from '../src/lib/learning-saddle-fluctuation-validation-math';
type Chain=Extract<SaddleFluctuationResult,{kind:'chain'}>;
const fixture=JSON.parse(readFileSync('scripts/fixtures/saddle-fluctuation-reference-inputs.json','utf8')) as {cases:{id:string;input:SaddleFluctuationInput}[]};
const directory='.runtime/learning/saddle-fluctuation-validation';mkdirSync(directory,{recursive:true});
const measurements:{id:string;milliseconds:number}[]=[];
for(const item of fixture.cases)test('independent whole response accepted: '+item.id,()=>{
  const result=calculateSaddleFluctuation(item.input),started=performance.now(),checked=validateSaddleFluctuationResult(item.input,result);
  measurements.push({id:item.id,milliseconds:performance.now()-started});
  writeFileSync(directory+'/measurements-node'+process.versions.node.split('.')[0]+'.json',JSON.stringify(measurements,null,2)+'\n');
  assert(checked,item.id);assert.deepEqual(checked,result);assert.notEqual(checked,result);assert(Object.isFrozen(checked));
  assert(Object.isFrozen(checked.kind==='chain'?checked.gaussian.covariance:checked.stationaryPoints));
});
function leaves(value:unknown,path:(string|number)[]=[]):{path:(string|number)[];value:unknown}[]{
  return value&&typeof value==='object'?Object.entries(value).flatMap(([k,x])=>leaves(x,[...path,Array.isArray(value)?Number(k):k])):[{path,value}];
}
function changed(source:unknown,path:(string|number)[],value:unknown){
  const result=structuredClone(source) as Record<string,unknown>;let cursor=result;
  for(const key of path.slice(0,-1))cursor=cursor[String(key)] as Record<string,unknown>;cursor[String(path.at(-1)!)]=value;return result;
}
let mutationCount=0;
for(const example of [saddleFluctuationExamples[0],saddleFluctuationExamples[1],saddleFluctuationExamples[8],...saddleFluctuationExamples.slice(24,27)])test('every primitive response field rejects a substantive mutation: '+example.id,()=>{
  const result=calculateSaddleFluctuation(example.input);assert(validateSaddleFluctuationResult(example.input,result));
  for(const leaf of leaves(result)){
    let next:unknown;
    if(typeof leaf.value==='boolean')next=!leaf.value;
    else if(typeof leaf.value==='number')next=leaf.value+1;
    else if(typeof leaf.value==='string'){
      try{next=readBounded(leaf.value).add(integer(1)).text();}catch{next=leaf.value+'-invalid';}
    }else next='unexpected';
    assert.equal(validateSaddleFluctuationResult(example.input,changed(result,leaf.path,next)),null,example.id+': '+leaf.path.join('.'));mutationCount++;
  }
});
test('maximum full matrices and determinant reject off-diagonal, boundary and large-value mutations',()=>{
  const input=saddleFluctuationExamples.find(x=>x.id==='tiny-mass-weak-quartic')?.input??fixture.cases.find(x=>x.id==='tiny-mass-weak-quartic')!.input;
  const result=calculateSaddleFluctuation(input);assert(result.kind==='chain');assert(validateSaddleFluctuationResult(input,result));
  for(const [i,j] of [[0,0],[0,32],[1,31],[16,16],[31,1],[32,32]])for(const matrix of ['inverseCurvature','covariance'] as const){
    const mutation:Chain=structuredClone(result);mutation.gaussian[matrix][i][j]={lower:'1',upper:'1'};
    assert.equal(validateSaddleFluctuationResult(input,mutation),null);mutationCount++;
  }
  const mutation:Chain=structuredClone(result);mutation.gaussian.determinant.normalized={lower:'0',upper:'1'};assert.equal(validateSaddleFluctuationResult(input,mutation),null);mutationCount++;
});
test('raw spelling, model, probe and mathematical input attribution are checked',()=>{
  const result=calculateSaddleFluctuation(saddleFluctuationDefault);
  for(const patch of [{time:'2/2'},{probe:0},{end:'0'},{quartic:'0'},{mass:'2'}])assert.equal(validateSaddleFluctuationResult({...saddleFluctuationDefault,...patch},result),null);
  assert.equal(validateSaddleFluctuationResult({mode:'well',hbar:'1',kappa:'1'},result),null);
});
test('invalid input or untrusted witnesses cannot replace the convexity certificate',()=>{
  const result=calculateSaddleFluctuation(saddleFluctuationDefault);assert(result.kind==='chain');
  for(const value of ['1/3','17','1/'+String(BigInt(1)<<BigInt(769)),'1'.repeat(4097),'0/1','-0','1e-300']){
    const mutation:Chain=structuredClone(result);mutation.stationary.candidate[1]=value;assert.equal(validateSaddleFluctuationResult(saddleFluctuationDefault,mutation),null);
  }
  assert.equal(validateSaddleFluctuationResult({...saddleFluctuationDefault,quartic:'-1'},result),null);
});
test('all record schemas and matrix dimensions are exact',()=>{
  const result=calculateSaddleFluctuation(saddleFluctuationDefault);assert(result.kind==='chain');
  for(const parent of [[],['stationary'],['gaussian'],['gaussian','determinant'],['correction'],['ratio'],['scope']]){
    const missing:Chain=structuredClone(result),extra:Chain=structuredClone(result);let a=missing as unknown as Record<string,unknown>,b=extra as unknown as Record<string,unknown>;
    for(const key of parent){a=a[key] as Record<string,unknown>;b=b[key] as Record<string,unknown>;}
    delete a[Object.keys(a)[0]];b.unexpected=true;
    assert.equal(validateSaddleFluctuationResult(saddleFluctuationDefault,missing),null);assert.equal(validateSaddleFluctuationResult(saddleFluctuationDefault,extra),null);
  }
  for(const matrix of ['hessian','inverseCurvature','covariance'] as const){const mutation:Chain=structuredClone(result);mutation.gaussian[matrix].pop();assert.equal(validateSaddleFluctuationResult(saddleFluctuationDefault,mutation),null);}
});
test('descriptor snapshots reject accessors, prototypes, symbols, cycles and excessive copies without executing getters',()=>{
  const result=calculateSaddleFluctuation(saddleFluctuationDefault);let calls=0;
  const getter={...result};Object.defineProperty(getter,'stationary',{get(){calls++;throw Error('getter');},enumerable:true});
  const symbol={...result,[Symbol('secret')]:true},inherited=Object.assign(Object.create({}),result),cycle={...result} as Record<string,unknown>;cycle.scope=cycle;
  for(const value of [getter,symbol,inherited,cycle,{...result,scope:{unbounded:'1'.repeat(65537)}}])assert.equal(validateSaddleFluctuationResult(saddleFluctuationDefault,value),null);
  assert.equal(calls,0);
});
test('enclosures require ordered, narrow, canonical dyadic bounds and contain the independent reference',()=>{
  const x=exact(integer(1));
  for(const range of [{lower:'0',upper:'2'},{lower:'2',upper:'1'},{lower:'1/3',upper:'1'},{lower:'0',upper:'0'}])assert.equal(verifyObservation(range,x),false);
  assert.throws(()=>verifyObservation({lower:'1/1',upper:'1'},x),/Noncanonical/);
  assert.equal(verifyObservation({lower:'0',upper:'0'},{lo:BigInt(1),hi:BigInt(-1)}),false);
  assert(verifyObservation({lower:'1',upper:'1'},x));
});
test('integer square/fourth roots enclose exact rational powers without floating arithmetic',()=>{
  for(const value of ['0','1','2','1/999999999989','999999999983/62499999999'])for(const power of [2,4] as const){
    const x=Q.read(value),bounds=root(x,power),lo=rational(bounds.lo),hi=rational(bounds.hi);
    let a=integer(1),b=integer(1);for(let i=0;i<power;i++){a=a.mul(lo);b=b.mul(hi);}
    assert(compare(a,x)<=0&&compare(b,x)>=0);
  }
});
test('continuant Green inverse encloses the free pinned chain closed form',()=>{
  const n=32,result=green(Array.from({length:n-1},()=>interval(integer(2),integer(2))));
  for(let i=0;i<n-1;i++)for(let j=0;j<n-1;j++){
    const expected=new Q(BigInt((Math.min(i,j)+1)*(n-Math.max(i,j)-1)),BigInt(n));
    assert(compare(rational(result.inverse[i][j].lo),expected)<=0);assert(compare(rational(result.inverse[i][j].hi),expected)>=0);
  }
});
test('validator bundle contains no producer engine, solver or shared producer arithmetic',async()=>{
  const {build}=await import('esbuild');const bundle=await build({entryPoints:['src/lib/learning-saddle-fluctuation-validation.ts'],bundle:true,write:false,metafile:true,platform:'browser',format:'esm'});
  const imports=Object.keys(bundle.metafile!.inputs);
  for(const forbidden of ['learning-saddle-fluctuation.ts','learning-quantum-hilbert-core.ts','learning-quantum-evolution-interval.ts'])assert(!imports.some(p=>p.endsWith('/'+forbidden)),forbidden);
});
test('record final substantive mutation count',()=>{
  assert(mutationCount>=200);
  writeFileSync(directory+'/mutation-count-node'+process.versions.node.split('.')[0]+'.json',JSON.stringify({rejected:mutationCount},null,2)+'\n');
});
