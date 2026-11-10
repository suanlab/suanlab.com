import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {build} from 'esbuild';
import {validateBoundaryLayerResult} from '../src/lib/learning-boundary-layer-validation';
import {evaluateBoundaryLayerRequest as calculate} from '../src/lib/learning-boundary-layer-evaluator';
import {evaluateBoundaryLayerValidationRequest as validate} from '../src/lib/learning-boundary-layer-validation-evaluator';
import {boundaryLayerDefault,boundaryLayerExamples,type BoundaryLayerInput} from '../src/lib/learning-boundary-layer-input';
import type {BoundaryLayerSuccess} from '../src/lib/learning-boundary-layer-protocol';
const root='.runtime/learning/boundary-layer-worker';mkdirSync(root,{recursive:true});
const fixture=JSON.parse(readFileSync('scripts/fixtures/boundary-layer-reference.json','utf8'))as{cases:{id:string;input:BoundaryLayerInput}[]};
for(const c of fixture.cases)test(`independent full boundary-layer response: ${c.id}`,()=>{
 const events:unknown[]=[],response=calculate({version:1,requestId:1,input:c.input},m=>events.push(m));assert(response.ok);assert.deepEqual(events,[{version:1,requestId:1,phase:'started',kind:'boundary-layer'}]);
 const accepted=validate({version:1,requestId:1,input:c.input,response});assert.equal(accepted.phase,'validated');if(accepted.phase==='validated'){assert.deepEqual(accepted.response,response);assert.notEqual(accepted.response,response);assert(Object.isFrozen(accepted.response.result));}
});
test('every leaf of sampled profiles,derivatives,intervals and metadata rejects individual alteration',()=>{
 const response=calculate({version:1,requestId:2,input:boundaryLayerDefault});assert(response.ok);
 const leaves:{path:(string|number)[];value:unknown}[]=[];
 const walk=(v:unknown,path:(string|number)[])=>{if(v===null||typeof v!=='object'){leaves.push({path,value:v});return;}for(const[k,x]of Object.entries(v))walk(x,[...path,Array.isArray(v)?Number(k):k]);};walk(response.result,[]);
 for(const leaf of leaves){const changed:BoundaryLayerSuccess['result']=structuredClone(response.result);let node:unknown=changed;for(const key of leaf.path.slice(0,-1))node=(node as Record<string|number,unknown>)[key];(node as Record<string|number,unknown>)[leaf.path.at(-1)!]=typeof leaf.value==='string'?leaf.value+'!':typeof leaf.value==='number'?leaf.value+1:typeof leaf.value==='boolean'?!leaf.value:'forged';assert.equal(validateBoundaryLayerResult(boundaryLayerDefault,changed),null,leaf.path.join('.'));}
 writeFileSync(root+'/leaf-mutations.json',JSON.stringify({rejected:leaves.length,paths:leaves.map(x=>x.path.join('.'))},null,2)+'\n');for(const root of ['input','layer','probe','physicalSamples','layerSamples','grid','contract'])assert(leaves.some(leaf=>leaf.path[0]===root),root+' mutation coverage');
});
test('raw spellings,request identity and invalid-input boundaries retain attribution',()=>{
 const input=boundaryLayerDefault,response=calculate({version:1,requestId:3,input});assert(response.ok);
 for(const changed of [{...response,requestId:99},{...response,inputKey:'wrong'},{...response,kind:'wrong'},{...response,input:{...input,epsilon:'2/20'}}])assert.equal(validate({version:1,requestId:3,input,response:changed}).phase,'rejected');
 assert.equal(calculate({version:1,requestId:0,input}).ok,false);assert.equal(calculate({version:1,requestId:1,input:{...input,epsilon:'0'}}).ok,false);assert.equal(validate({version:1,requestId:3,input:{...input,panels:Infinity},response}).phase,'rejected');
});
test('accessors,extra fields,cycles,symbols and oversized strings are rejected without getters',()=>{
 const response=calculate({version:1,requestId:4,input:boundaryLayerDefault});assert(response.ok);let reads=0;
 const getter={...response.result};Object.defineProperty(getter,'precisionBits',{get(){reads++;return 192;}});assert.equal(validateBoundaryLayerResult(boundaryLayerDefault,getter),null);assert.equal(reads,0);
 const cycle:Record<string,unknown>={...response.result};cycle.self=cycle;for(const v of [cycle,{...response.result,extra:1},{...response.result,[Symbol('bad')]:1},{...response.result,contract:'x'.repeat(65537)}])assert.equal(validateBoundaryLayerResult(boundaryLayerDefault,v),null);
});
test('zero drift,compatible data and tiny positive diffusion cannot be relabelled into a different regime',()=>{
 for(const id of ['zero-drift-forced','compatible-boundaries','tiny-diffusion','tiny-nonzero-drift']){const input=boundaryLayerExamples.find(c=>c.id===id)!.input,response=calculate({version:1,requestId:5,input});assert(response.ok);const changed=structuredClone(response)as BoundaryLayerSuccess;
  if(changed.result.layer)changed.result.layer.hasLayerAmplitude=!changed.result.layer.hasLayerAmplitude;else (changed.result as unknown as Record<string,unknown>).branch='nonzero-drift';assert.equal(validate({version:1,requestId:5,input,response:changed}).phase,'rejected');}
});
test('untrusted output fractions never become independent arithmetic operands',async()=>{
 const{Rational}=await import('../src/lib/learning-graph-gauge-validation-math');const original=Rational.read,input={...boundaryLayerDefault,probe:'3/17'},response=calculate({version:1,requestId:6,input});assert(response.ok);const forged=structuredClone(response.result);forged.probe.exact.lower='untrusted-fraction';let parsed=false;
 Rational.read=function(value:string){if(value==='untrusted-fraction'){parsed=true;throw Error('untrusted arithmetic');}return original.call(this,value);};try{assert.equal(validateBoundaryLayerResult(input,forged),null);assert.equal(parsed,false);}finally{Rational.read=original;}
});
test('reader hook excludes arithmetic and independent verifier excludes all producer engines',async()=>{
 for(const entry of ['src/components/learning/use-boundary-layer-calculation.ts','src/lib/learning-boundary-layer-validation.worker.ts']){const bundle=await build({entryPoints:[entry],bundle:true,write:false,platform:'browser',format:'esm',metafile:true});const inputs=Object.keys(bundle.metafile!.inputs);
  for(const forbidden of ['learning-boundary-layer.ts','learning-boundary-layer-evaluator.ts','learning-quantum-hilbert-core.ts','learning-interval-transcendentals.ts'])assert(!inputs.includes('src/lib/'+forbidden));
  if(entry.includes('use-boundary')){assert(!inputs.includes('src/lib/learning-boundary-layer-validation.ts'));assert(!bundle.outputFiles[0].text.includes('BigInt('));}
  writeFileSync(root+'/'+(entry.includes('use-boundary')?'hook':'independent')+'-dependency-graph.json',JSON.stringify(bundle.metafile,null,2)+'\n');
 }
});
test('hook server-renders idle without creating a Worker',async()=>{
 const React=await import('react'),{renderToStaticMarkup}=await import('react-dom/server'),{useBoundaryLayerCalculation}=await import('../src/components/learning/use-boundary-layer-calculation');const saved=Object.getOwnPropertyDescriptor(globalThis,'Worker');let calls=0;Object.defineProperty(globalThis,'Worker',{configurable:true,get(){calls++;throw Error('SSR Worker');}});
 try{function Probe(){return React.createElement('span',null,useBoundaryLayerCalculation(boundaryLayerDefault,true).status);}assert.equal(renderToStaticMarkup(React.createElement(Probe)),'<span>idle</span>');assert.equal(calls,0);}finally{if(saved)Object.defineProperty(globalThis,'Worker',saved);else Reflect.deleteProperty(globalThis,'Worker');}
});
