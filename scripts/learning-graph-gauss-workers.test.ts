import assert from 'node:assert/strict';
import test from 'node:test';
import {readFileSync,mkdirSync,writeFileSync} from 'node:fs';
import {build} from 'esbuild';
import {validateGraphGaussResult} from '../src/lib/learning-graph-gauss-validation';
import {evaluateGraphGaussRequest as calculate} from '../src/lib/learning-graph-gauss-evaluator';
import {evaluateGraphGaussValidationRequest as validate} from '../src/lib/learning-graph-gauss-validation-evaluator';
import {graphGaussExamples} from '../src/lib/learning-graph-gauss-examples';
import type {GraphGaussSuccess} from '../src/lib/learning-graph-gauss-protocol';
import type {GraphGaussInput} from '../src/lib/learning-graph-gauss-input';
const cases=JSON.parse(readFileSync('scripts/fixtures/graph-gauss-independent.json','utf8')).cases as {label:string;input:GraphGaussInput;expected:{chargesCompatible:boolean;inputAdmissible:boolean;physicalPhaseSpaceDimension:number|null}}[];
const preparation=(id:string)=>{const x=graphGaussExamples.find(x=>x.id===id);assert(x);return x.input;};
for(const x of cases)test(`independent full canonical response: ${x.label}`,()=>{
 const started:unknown[]=[],response=calculate({version:1,requestId:1,input:x.input},s=>started.push(s));assert(response.ok,x.label);assert.equal(started.length,1);assert.equal(response.result.chargesCompatible,x.expected.chargesCompatible);assert.equal(response.result.inputAdmissible,x.expected.inputAdmissible);assert.equal(response.result.counts.physicalPhaseSpaceDimension,x.expected.physicalPhaseSpaceDimension);
 const receipt=validate({version:1,requestId:1,input:x.input,response});assert.equal(receipt.phase,'validated',x.label);if(receipt.phase==='validated'){assert.deepEqual(receipt.response,response);assert(Object.isFrozen(receipt.response.result.fixing.nodeTree[0]));}
});
test('every individual output scalar,path,index,flag,null and contract rejects mutation',()=>{
 const input=preparation('physical-triangle'),response=calculate({version:1,requestId:2,input});assert(response.ok);const paths:(string|number)[][]=[];
 const visit=(v:unknown,path:(string|number)[])=>{if(v&&typeof v==='object'){for(const[k,x]of Object.entries(v))if(k!=='input')visit(x,[...path,Array.isArray(v)?Number(k):k]);}else paths.push(path);};visit(response.result,[]);
 for(const path of paths){const altered=JSON.parse(JSON.stringify(response))as GraphGaussSuccess;let parent=altered.result as unknown as Record<string|number,unknown>;for(const k of path.slice(0,-1))parent=parent[k]as Record<string|number,unknown>;const key=path.at(-1)!,old=parent[key];parent[key]=typeof old==='boolean'?!old:typeof old==='number'?old+1:old===null?0:old==='0'?'1':'0';assert.equal(validate({version:1,requestId:2,input,response:altered}).phase,'rejected',path.join('.'));}
 mkdirSync('.runtime/learning/graph-gauss-worker',{recursive:true});writeFileSync('.runtime/learning/graph-gauss-worker/leaf-mutations.json',JSON.stringify({mutationsRejected:paths.length,oneLeafPerClone:true},null,2)+'\n');assert.equal(paths.length,213);
});
test('empty levels and invalid inputs cannot be forged into physical reduced states',()=>{
 for(const id of ['incompatible-total-charge','global-neutrality-insufficient','constraint-violating-triangle']){const input=preparation(id),response=calculate({version:1,requestId:3,input});assert(response.ok);const altered:GraphGaussSuccess=structuredClone(response);altered.result.chargesCompatible=true;altered.result.inputAdmissible=true;altered.result.counts.physicalPhaseSpaceDimension=2;altered.result.canonical.currentStateReduced=true;altered.result.canonical.currentHamiltonianResidual='0';assert.equal(validate({version:1,requestId:3,input,response:altered}).phase,'rejected',id);}
});
test('boundary flux,canonical brackets,particular fields and exact force signs are independently checked',()=>{
 for(const id of ['boundary-charge-flux','every-site-anchored','parallel-links-self-loop','alternative-tree-root']){const input=preparation(id),response=calculate({version:1,requestId:4,input});assert(response.ok);const altered:GraphGaussSuccess=structuredClone(response);
  if(id==='boundary-charge-flux')altered.result.flow.anchorChargeRates[0].value='0';
  if(id==='every-site-anchored')altered.result.canonical.coordinateMomentumBrackets[0][0]=0;
  if(id==='parallel-links-self-loop')altered.result.flow.electricRates[0]='-1';
  if(id==='alternative-tree-root')altered.result.canonical.affineElectric=['1','0','0'];
  assert.equal(validate({version:1,requestId:4,input,response:altered}).phase,'rejected',id);
 }
});
test('tree order,root,anchors,raw fractional spelling and request IDs retain attribution',()=>{
 const input=preparation('physical-triangle'),response=calculate({version:1,requestId:5,input});assert(response.ok);
 for(const other of [{...input,forestEdges:[1,0]},{...input,roots:[1]},{...input,electric:['2/2','1','1']},{...input,anchors:[0],charges:[null,'0','0']}])assert.equal(validate({version:1,requestId:5,input:other,response}).phase,'rejected');
 assert.equal(validate({version:1,requestId:6,input,response}).phase,'rejected');const altered:GraphGaussSuccess=structuredClone(response);altered.kind='gauge-orbits'as'graph-gauss';assert.equal(validate({version:1,requestId:5,input,response:altered}).phase,'rejected');
});
test('wire ownership rejects accessors,extra keys and unchanged wire size limits',()=>{
 const input=preparation('physical-triangle');for(const request of [{version:2,requestId:1,input},{version:1,requestId:0,input},{version:1,requestId:1.5,input},{version:1,requestId:1,input,extra:true},{version:1,requestId:1,input:{...input,forestEdges:[]}}])assert.equal(calculate(request).ok,false);
 let calls=0;assert.equal(calculate({get version(){calls++;return 1;},requestId:1,input}).ok,false);const response=calculate({version:1,requestId:7,input});assert(response.ok);const accessor=structuredClone(response.result);Object.defineProperty(accessor.canonical,'electricMetric',{get(){calls++;throw Error('Getter');}});assert.equal(validateGraphGaussResult(input,accessor),null);assert.equal(calls,0);
 const altered:GraphGaussSuccess=structuredClone(response);altered.result.energy.total='1'.repeat(65537);const receipt=validate({version:1,requestId:7,input,response:altered});assert.equal(receipt.phase,'rejected');if(receipt.phase==='rejected')assert.equal(receipt.error,'resource-limit');
 const extra=structuredClone(response.result)as unknown as Record<string,unknown>;extra.unclaimed=true;assert.equal(validateGraphGaussResult(input,extra),null);
});
test('untrusted output fractions are never parsed as BigInt operands',()=>{
 const input=preparation('physical-triangle'),response=calculate({version:1,requestId:8,input});assert(response.ok);const altered:GraphGaussSuccess=structuredClone(response);altered.result.energy.total='9'.repeat(64000);let untrustedParsed=false;const original=globalThis.BigInt;globalThis.BigInt=((v:Parameters<typeof BigInt>[0])=>{if(typeof v==='string'&&v.length>40){untrustedParsed=true;throw Error('Untrusted arithmetic');}return original(v);})as typeof BigInt;
 try{assert.equal(validateGraphGaussResult(input,altered.result),null);assert.equal(untrustedParsed,false);}finally{globalThis.BigInt=original;}
});
const budget=JSON.parse(readFileSync('scripts/fixtures/graph-gauss-budget-preparations.json','utf8'))as{label:string;input:GraphGaussInput}[];
for(const x of budget)test(`maximum independent response and wire budget: ${x.label}`,()=>{const response=calculate({version:1,requestId:9,input:x.input});assert(response.ok);assert.equal(validate({version:1,requestId:9,input:x.input,response}).phase,'validated');});
test('reader hook excludes exact arithmetic; validator excludes all producer engines',async()=>{
 for(const entry of ['src/components/learning/use-graph-gauss-calculation.ts','src/lib/learning-graph-gauss-validation.worker.ts']){const bundle=await build({entryPoints:[entry],bundle:true,write:false,platform:'browser',format:'esm',metafile:true,logLevel:'silent'}),inputs=Object.keys(bundle.metafile!.inputs);for(const file of ['learning-graph-gauss.ts','learning-gauge-orbits.ts','learning-quantum-hilbert-core.ts','learning-graph-gauss-evaluator.ts'])assert(!inputs.includes('src/lib/'+file));if(!entry.includes('validation.worker')){assert(!inputs.includes('src/lib/learning-graph-gauss-validation.ts'));assert(!bundle.outputFiles[0].text.includes('BigInt('));}}
});
test('the React hook server-renders idle without constructing a Worker',async()=>{
 const{createElement}=await import('react'),{renderToString}=await import('react-dom/server'),{useGraphGaussCalculation}=await import('../src/components/learning/use-graph-gauss-calculation');const before=Object.getOwnPropertyDescriptor(globalThis,'Worker');let calls=0;Object.defineProperty(globalThis,'Worker',{configurable:true,get(){calls++;throw Error('SSR Worker');}});
 try{function Probe(){return createElement('span',null,useGraphGaussCalculation(preparation('physical-triangle'),true).status);}assert.equal(renderToString(createElement(Probe)),'<span>idle</span>');assert.equal(calls,0);}finally{if(before)Object.defineProperty(globalThis,'Worker',before);else Reflect.deleteProperty(globalThis,'Worker');}
});
