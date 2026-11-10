import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdirSync,writeFileSync} from 'node:fs';
import {build} from 'esbuild';
import {evaluateRealTimePhaseRequest as calculate} from '../src/lib/learning-real-time-phase-evaluator';
import {evaluateRealTimePhaseValidationRequest as validate} from '../src/lib/learning-real-time-phase-validation-evaluator';
import {realTimeChainDefault,realTimePacketDefault,realTimeFoldDefault} from '../src/lib/learning-real-time-phase-input';
import {realTimePhaseInputKey} from '../src/lib/learning-real-time-phase-protocol';
const root='.runtime/learning/real-time-phase-worker';mkdirSync(root,{recursive:true});
for(const input of [realTimeChainDefault,realTimePacketDefault,{...realTimeFoldDefault,control:'0'}])test('producer remains unverified andindependent validation admits '+input.mode,()=>{
 const started:unknown[]=[],verification:unknown[]=[],response=calculate({version:1,requestId:7,input},e=>started.push(e));assert(response.ok);
 assert.deepEqual(started,[{version:1,requestId:7,phase:'started',kind:'real-time-phase'}]);
 assert(Object.isFrozen(response)&&Object.isFrozen(response.result));
 const receipt=validate({version:1,requestId:7,input,response},e=>verification.push(e));assert(receipt.phase==='validated');
 assert.deepEqual(verification,[{version:1,requestId:7,phase:'verification-started',kind:'real-time-phase'}]);
 assert(receipt.fullResponseValidated);assert.deepEqual(receipt.response,response);assert.notEqual(receipt.response,response);assert(Object.isFrozen(receipt.response.result));
});
test('malformed requests andinvalid physical inputs produce no started event',()=>{
 for(const request of [null,{}, {version:2,requestId:1,input:realTimeChainDefault},
  ...[0,-0,-1,0.5,Infinity,Number.MAX_SAFE_INTEGER+1].map(requestId=>({version:1,requestId,input:realTimeChainDefault})),
  {version:1,requestId:1,input:{...realTimeChainDefault,hbar:'0'}}]) {
  const events:unknown[]=[];assert(!calculate(request,e=>events.push(e)).ok);assert.equal(events.length,0);
  assert.equal(validate(request,e=>events.push(e)).phase,'rejected');assert.equal(events.length,0);
 }
});
test('validation rejects foreign IDs,original fraction spellings andmathematical corruption',()=>{
 const input={...realTimeChainDefault,time:'2/2'},response=calculate({version:1,requestId:3,input});assert(response.ok);
 assert.notEqual(realTimePhaseInputKey(input),realTimePhaseInputKey(realTimeChainDefault));
 for(const value of [{version:1,requestId:4,input,response},{version:1,requestId:3,input:realTimeChainDefault,response},
  {version:1,requestId:3,input,response:{...response,result:{...(response.result as object),extra:true}}}]) {
  const receipt=validate(value);assert(receipt.phase==='rejected');assert.equal(receipt.error,'invalid-response');
 }
});
test('accessors are never evaluated andcopy resource failures remain distinct',()=>{
 let reads=0;const getter={get version(){reads++;return 1;},requestId:1,input:realTimeChainDefault};
 assert(!calculate(getter).ok);assert.equal(validate(getter).phase,'rejected');assert.equal(reads,0);
 const response=calculate({version:1,requestId:1,input:realTimeChainDefault});assert(response.ok);
 const tooLarge={...response,result:{...(response.result as object),extra:'x'.repeat(65537)}};
 assert.deepEqual(validate({version:1,requestId:1,input:realTimeChainDefault,response:tooLarge}),{version:1,requestId:1,phase:'rejected',error:'resource-limit'});
 const proxy=Proxy.revocable({},{});proxy.revoke();assert(!calculate(proxy.proxy).ok);assert.equal(validate(proxy.proxy).phase,'rejected');
});
test('bundled hook excludes allmathematics andvalidation Worker excludes producer engines',async()=>{
 for(const[label,entry]of [['hook','src/components/learning/use-real-time-phase-calculation.ts'],['calculate','src/lib/learning-real-time-phase.worker.ts'],['validate','src/lib/learning-real-time-phase-validation.worker.ts']]as const) {
  const bundle=await build({entryPoints:[entry],bundle:true,write:false,platform:'browser',format:'esm',target:'es2020',metafile:true}),inputs=Object.keys(bundle.metafile!.inputs);
  if(label!=='calculate')for(const file of ['learning-real-time-phase.ts','learning-real-time-phase-exact.ts','learning-real-time-airy.ts','learning-real-time-wave-observations.ts','learning-quantum-hilbert-core.ts'])assert(!inputs.includes('src/lib/'+file),label+' '+file);
  if(label==='hook'){assert(!inputs.includes('src/lib/learning-real-time-phase-validation.ts'));assert(!bundle.outputFiles[0].text.includes('BigInt('));}
  if(label==='validate')assert(inputs.includes('src/lib/learning-real-time-phase-validation.ts'));
  writeFileSync(root+'/'+label+'-dependency-graph.json',JSON.stringify(bundle.metafile,null,2)+'\n');
 }
});
test('server rendering never reads document orWorker globals',async()=>{
 const React=await import('react'),{renderToStaticMarkup}=await import('react-dom/server'),{useRealTimePhaseCalculation}=await import('../src/components/learning/use-real-time-phase-calculation');
 const saved=Object.getOwnPropertyDescriptor(globalThis,'Worker');let reads=0;
 Object.defineProperty(globalThis,'Worker',{configurable:true,get(){reads++;throw Error('SSR Worker');}});
 try{function Probe(){return React.createElement('span',null,useRealTimePhaseCalculation(realTimeChainDefault,true).status);}assert.equal(renderToStaticMarkup(React.createElement(Probe)),'<span>idle</span>');assert.equal(reads,0);}
 finally{if(saved)Object.defineProperty(globalThis,'Worker',saved);else Reflect.deleteProperty(globalThis,'Worker');}
});
