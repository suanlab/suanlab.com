import assert from 'node:assert/strict';
import test from 'node:test';
import {readFileSync} from 'node:fs';
import {build} from 'esbuild';
import {validateFieldBalanceResult} from '../src/lib/learning-field-balance-validation';
import {evaluateFieldBalanceRequest as calculate} from '../src/lib/learning-field-balance-evaluator';
import {evaluateFieldBalanceValidationRequest as validate} from '../src/lib/learning-field-balance-validation-evaluator';
import {fieldBalanceExamples} from '../src/lib/learning-field-balance-examples';
import type {FieldBalanceSuccess} from '../src/lib/learning-field-balance-protocol';
import type {FieldBalanceInput} from '../src/lib/learning-field-balance-input';
const refs=JSON.parse(readFileSync('scripts/fixtures/field-balance-independent.json','utf8'))as{cases:{label:string;input:FieldBalanceInput;expected:unknown}[]};
for(const c of refs.cases)test(`independent forward differentiation accepts chain-rule reference: ${c.label}`,()=>{assert(validateFieldBalanceResult(c.input,c.expected));});
for(const c of fieldBalanceExamples)test(`owned producer/validation protocol and explained preparation: ${c.id}`,()=>{
 const started:unknown[]=[],response=calculate({version:1,requestId:1,input:c.input},message=>started.push(message));assert(response.ok);assert.equal(started.length,1);const receipt=validate({version:1,requestId:1,input:c.input,response});assert.equal(receipt.phase,'validated');if(receipt.phase==='validated'){assert.deepEqual(receipt.response,response);assert(Object.isFrozen(receipt.response.result));}
});
test('every exact numerical leaf,flag,index and convention rejects a single changed claim',()=>{
 const input=fieldBalanceExamples[0].input,response=calculate({version:1,requestId:2,input});assert(response.ok);
 const paths:(string|number)[][]=[];const visit=(value:unknown,path:(string|number)[])=>{if(value&&typeof value==='object'){for(const[key,child]of Object.entries(value))if(key!=='input')visit(child,[...path,Array.isArray(value)?Number(key):key]);}else paths.push(path);};visit(response.result,[]);
 for(const path of paths){const changed:FieldBalanceSuccess=structuredClone(response);const root:Record<string|number,unknown>=changed.result as unknown as Record<string|number,unknown>;let parent:Record<string|number,unknown>=root;for(const key of path.slice(0,-1))parent=parent[key]as Record<string|number,unknown>;const key=path.at(-1)!,old=parent[key];parent[key]=typeof old==='boolean'?!old:typeof old==='number'?old+1:old===null?0:old==='0'?'1':'0';assert.equal(validate({version:1,requestId:2,input,response:changed}).phase,'rejected',path.join('.'));}
 assert.equal(paths.length,100);
});
test('opposite local errors cannot hide behind a correct global sum or zero residuals',()=>{
 const input=fieldBalanceExamples[0].input,response=calculate({version:1,requestId:3,input});assert(response.ok);
 const bad=structuredClone(response);bad.result.cells[0].energy='9/4';bad.result.cells[2].energy='3/4';assert.equal(bad.result.totalEnergy,'7/2');assert.equal(validate({version:1,requestId:3,input,response:bad}).phase,'rejected');
 const periodic=structuredClone(response);(periodic.result.spatialGenerator as unknown as Record<string,unknown>).literalContinuousSpatialTranslationClaimed=true;assert.equal(validate({version:1,requestId:3,input,response:periodic}).phase,'rejected');
 const boundary=fieldBalanceExamples.find(p=>p.id==='fixed-instant-zero')!.input,r=calculate({version:1,requestId:3,input:boundary});assert(r.ok);const forged=structuredClone(r);forged.result.spatialGenerator=response.result.spatialGenerator;assert.equal(validate({version:1,requestId:3,input:boundary,response:forged}).phase,'rejected');
});
test('share spelling,field state,boundary and request IDs belong to the original calculation',()=>{
 const input=fieldBalanceExamples[0].input,response=calculate({version:1,requestId:4,input});assert(response.ok);
 for(const other of [{...input,share:'2/4'},{...input,share:'1'},{...input,field:{...input.field,boundary:'neumann-zero'}}])assert.equal(validate({version:1,requestId:4,input:other,response}).phase,'rejected');
 assert.equal(validate({version:1,requestId:5,input,response}).phase,'rejected');const bad=structuredClone(response);bad.result.input={...input,share:'1'};assert.equal(validate({version:1,requestId:4,input,response:bad}).phase,'rejected');
});
test('protocol owns data without invoking accessors and applies explicit wire budgets',()=>{
 const input=fieldBalanceExamples[0].input;for(const frame of [{version:2,requestId:1,input},{version:1,requestId:0,input},{version:1,requestId:1.5,input},{version:1,requestId:1,input,extra:true},{version:1,requestId:1,input:{...input,share:'2'}}])assert.equal(calculate(frame).ok,false);
 let reads=0;const frame={get version(){reads++;return 1;},requestId:1,input};assert.equal(calculate(frame).ok,false);const response=calculate({version:1,requestId:1,input});assert(response.ok);const result=structuredClone(response.result);Object.defineProperty(result,'totalEnergy',{get(){reads++;throw Error('Accessor invoked');}});assert.equal(validateFieldBalanceResult(input,result),null);assert.equal(reads,0);
 const huge=structuredClone(response);huge.result.totalEnergy='1'.repeat(65537);const receipt=validate({version:1,requestId:1,input,response:huge});assert.equal(receipt.phase,'rejected');if(receipt.phase==='rejected')assert.equal(receipt.error,'resource-limit');
});
const budget=JSON.parse(readFileSync('scripts/fixtures/scalar-field-budget-reference.json','utf8'))as{cases:{label:string;input:FieldBalanceInput['field'];expectedEnergy:string}[]};
for(const c of budget.cases)test(`32-site,18-digit field and fractional allocation fit the transport budget: ${c.label}`,()=>{
 const input={field:c.input,share:'333333333333333333/999999999999999999'},response=calculate({version:1,requestId:5,input});assert(response.ok);assert.equal(response.result.totalEnergy,c.expectedEnergy);assert.equal(validate({version:1,requestId:5,input,response}).phase,'validated');
});
test('reader hook/controller excludes mathematics and validation excludes both producer engines',async()=>{
 for(const entry of ['src/components/learning/use-field-balance-calculation.ts','src/lib/learning-field-balance-validation.worker.ts']){const r=await build({entryPoints:[entry],bundle:true,write:false,platform:'browser',format:'esm',metafile:true,logLevel:'silent'}),inputs=Object.keys(r.metafile!.inputs);assert(!inputs.includes('src/lib/learning-field-balance.ts'));assert(!inputs.includes('src/lib/learning-scalar-field.ts'));if(!entry.includes('validation.worker')){assert(!inputs.includes('src/lib/learning-field-balance-validation.ts'));assert(!r.outputFiles[0].text.includes('BigInt('));}}
});
test('hook renders idle on the server without looking up Worker',async()=>{
 const{createElement}=await import('react'),{renderToString}=await import('react-dom/server'),{useFieldBalanceCalculation}=await import('../src/components/learning/use-field-balance-calculation');const before=Object.getOwnPropertyDescriptor(globalThis,'Worker');let calls=0;Object.defineProperty(globalThis,'Worker',{configurable:true,get(){calls++;throw Error('SSR Worker');}});
 try{function Probe(){return createElement('span',null,useFieldBalanceCalculation(fieldBalanceExamples[0].input,true).status);}assert.equal(renderToString(createElement(Probe)),'<span>idle</span>');assert.equal(calls,0);}finally{if(before)Object.defineProperty(globalThis,'Worker',before);else Reflect.deleteProperty(globalThis,'Worker');}
});
