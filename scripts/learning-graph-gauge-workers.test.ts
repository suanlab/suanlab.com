import assert from 'node:assert/strict';
import test from 'node:test';
import {readFileSync} from 'node:fs';
import {build} from 'esbuild';
import {validateGraphGaugeResult} from '../src/lib/learning-graph-gauge-validation';
import {evaluateGraphGaugeRequest as calculate} from '../src/lib/learning-graph-gauge-evaluator';
import {evaluateGraphGaugeValidationRequest as validate} from '../src/lib/learning-graph-gauge-validation-evaluator';
import {graphGaugeExamples} from '../src/lib/learning-graph-gauge-examples';
import type {GraphGaugeSuccess} from '../src/lib/learning-graph-gauge-protocol';
import type {GraphGaugeInput} from '../src/lib/learning-graph-gauge-input';
const cases=JSON.parse(readFileSync('scripts/fixtures/graph-gauge-independent.json','utf8')).cases as {label:string;input:GraphGaugeInput}[];
for(const x of cases)test(`independent scalar energy validator accepts fixture input: ${x.label}`,()=>{
 const started:unknown[]=[],response=calculate({version:1,requestId:1,input:x.input},s=>started.push(s));assert(response.ok);assert.equal(started.length,1);const receipt=validate({version:1,requestId:1,input:x.input,response});assert.equal(receipt.phase,'validated');if(receipt.phase==='validated'){assert.deepEqual(receipt.response,response);assert(Object.isFrozen(receipt.response.result));assert(Object.isFrozen(receipt.response.result.operator.original[0]));}
});
test('every numerical leaf,flag,index and declared convention rejects a changed claim',()=>{
 const input=graphGaugeExamples[0].input,response=calculate({version:1,requestId:2,input});assert(response.ok);
 const paths:(string|number)[][]=[];const visit=(v:unknown,path:(string|number)[])=>{if(v&&typeof v==='object'){for(const[k,x]of Object.entries(v))if(k!=='input')visit(x,[...path,Array.isArray(v)?Number(k):k]);}else paths.push(path);};visit(response.result,[]);
 for(const path of paths){const altered:GraphGaugeSuccess=structuredClone(response);let parent=altered.result as unknown as Record<string|number,unknown>;for(const k of path.slice(0,-1))parent=parent[k]as Record<string|number,unknown>;const k=path.at(-1)!,old=parent[k];parent[k]=typeof old==='boolean'?!old:typeof old==='number'?old+1:old==='0'?'1':'0';assert.equal(validate({version:1,requestId:2,input,response:altered}).phase,'rejected',path.join('.'));}
 assert.equal(paths.length,231);
});
test('energy summaries cannot hide local changes or incomplete invariant claims',()=>{
 const input=graphGaugeExamples.find(x=>x.id==='positive-flux')!.input,response=calculate({version:1,requestId:3,input});assert(response.ok);
 const altered:GraphGaugeSuccess=structuredClone(response);altered.result.edges[0].energy='1';altered.result.edges[1].energy='1';assert.equal(altered.result.energy.hopping,'2');assert.equal(validate({version:1,requestId:3,input,response:altered}).phase,'rejected');
 for(const edit of [(r:GraphGaugeSuccess)=>{r.result.walks[0].holonomy.imaginary='-1';},(r:GraphGaugeSuccess)=>{r.result.classification.completeLoopBasisClaimed=true;},(r:GraphGaugeSuccess)=>{r.result.classification.gaussConstraintImplemented=true;},(r:GraphGaugeSuccess)=>{r.result.nodes[0].gradient.real='4';}]){const bad:GraphGaugeSuccess=structuredClone(response);edit(bad);assert.equal(validate({version:1,requestId:3,input,response:bad}).phase,'rejected');}
});
test('ownership,chart spelling,loop orientation and request attribution are preserved',()=>{
 const input=graphGaugeExamples[0].input,response=calculate({version:1,requestId:4,input});assert(response.ok);
 for(const other of [{...input,quadratic:'0/3'},{...input,walks:[{start:0,steps:[2,1,0].map(edge=>({edge,forward:false}))}]},{...input,frames:input.frames.map(g=>({...g,cayley:'0',minusOne:false}))}])assert.equal(validate({version:1,requestId:4,input:other,response}).phase,'rejected');
 assert.equal(validate({version:1,requestId:5,input,response}).phase,'rejected');const bad:GraphGaugeSuccess=structuredClone(response);bad.result.input={...input,quadratic:'1'};assert.equal(validate({version:1,requestId:4,input,response:bad}).phase,'rejected');
 const extra=structuredClone(response)as GraphGaugeSuccess&{additional?:boolean};extra.additional=true;assert.equal(validate({version:1,requestId:4,input,response:extra}).phase,'rejected');
 let reads=0;const getter={...response.result};Object.defineProperty(getter,'energy',{get(){reads++;throw Error('Accessor');}});assert.equal(validateGraphGaugeResult(input,getter),null);assert.equal(reads,0);
});
test('protocol rejects malformed requests and enforces explicit wire budgets',()=>{
 const input=graphGaugeExamples[0].input;for(const request of [{version:2,requestId:1,input},{version:1,requestId:0,input},{version:1,requestId:1.5,input},{version:1,requestId:1,input,extra:true},{version:1,requestId:1,input:{...input,loopCoupling:'-1'}}])assert.equal(calculate(request).ok,false);
 let reads=0;assert.equal(calculate({get version(){reads++;return 1;},requestId:1,input}).ok,false);assert.equal(reads,0);
 const response=calculate({version:1,requestId:5,input});assert(response.ok);const huge=structuredClone(response);huge.result.energy.total='1'.repeat(65537);const receipt=validate({version:1,requestId:5,input,response:huge});assert.equal(receipt.phase,'rejected');if(receipt.phase==='rejected')assert.equal(receipt.error,'resource-limit');
});
test('reader hook excludes mathematics; validator excludes producer arithmetic and evaluator',async()=>{
 for(const entry of ['src/components/learning/use-graph-gauge-calculation.ts','src/lib/learning-graph-gauge-validation.worker.ts']){
  const r=await build({entryPoints:[entry],bundle:true,write:false,platform:'browser',format:'esm',metafile:true,logLevel:'silent'}),inputs=Object.keys(r.metafile!.inputs);
  assert(!inputs.includes('src/lib/learning-graph-gauge.ts'));assert(!inputs.includes('src/lib/learning-quantum-hilbert-core.ts'));assert(!inputs.includes('src/lib/learning-graph-gauge-evaluator.ts'));
  if(!entry.includes('validation.worker')){assert(!inputs.includes('src/lib/learning-graph-gauge-validation.ts'));assert(!r.outputFiles[0].text.includes('BigInt('));}
 }
});
test('server rendering the hook does not access Worker',async()=>{
 const{createElement}=await import('react'),{renderToString}=await import('react-dom/server'),{useGraphGaugeCalculation}=await import('../src/components/learning/use-graph-gauge-calculation');const before=Object.getOwnPropertyDescriptor(globalThis,'Worker');let calls=0;Object.defineProperty(globalThis,'Worker',{configurable:true,get(){calls++;throw Error('SSR Worker');}});
 try{function Probe(){return createElement('span',null,useGraphGaugeCalculation(graphGaugeExamples[0].input,true).status);}assert.equal(renderToString(createElement(Probe)),'<span>idle</span>');assert.equal(calls,0);}finally{if(before)Object.defineProperty(globalThis,'Worker',before);else Reflect.deleteProperty(globalThis,'Worker');}
});
