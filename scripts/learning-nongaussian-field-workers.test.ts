import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';import {build} from 'esbuild';
import {evaluateNongaussianFieldRequest as calculate} from '../src/lib/learning-nongaussian-field-evaluator';
import {evaluateNongaussianFieldValidationRequest as validate} from '../src/lib/learning-nongaussian-field-validation-evaluator';
import {validateNongaussianFieldResult} from '../src/lib/learning-nongaussian-field-validation';
import {nongaussianFieldDefault,nongaussianFieldExamples,type NongaussianFieldInput} from '../src/lib/learning-nongaussian-field-input';
import type {NongaussianFieldSuccess} from '../src/lib/learning-nongaussian-field-protocol';
const root='.runtime/learning/nongaussian-field-worker';mkdirSync(root,{recursive:true});
const fixture=JSON.parse(readFileSync('scripts/fixtures/nongaussian-field-reference.json','utf8'))as{cases:{id:string;input:NongaussianFieldInput;expected:Record<string,unknown>}[]};
for(const c of fixture.cases)test('independent full non-Gaussian response: '+c.id,()=>{
 const events:unknown[]=[],response=calculate({version:1,requestId:1,input:c.input},m=>events.push(m));assert(response.ok);assert.deepEqual(events,[{version:1,requestId:1,phase:'started',kind:'nongaussian-field'}]);const{input,contract,...math}=response.result;assert.deepEqual(math,c.expected);assert.deepEqual(input,c.input);assert(contract.length>100);
 const accepted=validate({version:1,requestId:1,input:c.input,response});assert.equal(accepted.phase,'validated',c.id);if(accepted.phase==='validated'){assert.deepEqual(accepted.response,response);assert.notEqual(accepted.response,response);assert(Object.isFrozen(accepted.response.result)&&Object.isFrozen(accepted.response.input.sourceFactors));}
});
test('every actual leaf of default,four-body and maximum nested responses rejects individual alteration',()=>{
 const preparations=[{id:'default',input:nongaussianFieldDefault},{id:'four-body',input:nongaussianFieldExamples.find(c=>c.id==='pure-four-body')!.input},{id:'maximum',input:fixture.cases.find(c=>c.id==='maximum-coprime')!.input}],rows=[];
 for(const preparation of preparations){const response=calculate({version:1,requestId:2,input:preparation.input});assert(response.ok);const leaves:{path:(string|number)[];value:unknown}[]=[];
 const walk=(v:unknown,path:(string|number)[])=>{if(v===null||typeof v!=='object'){leaves.push({path,value:v});return;}for(const[k,x]of Object.entries(v))walk(x,[...path,Array.isArray(v)?Number(k):k]);};walk(response.result,[]);
 for(const leaf of leaves){const changed:NongaussianFieldSuccess['result']=structuredClone(response.result);let node:unknown=changed;for(const key of leaf.path.slice(0,-1))node=(node as Record<string|number,unknown>)[key];(node as Record<string|number,unknown>)[leaf.path.at(-1)!]=typeof leaf.value==='string'?leaf.value+'!':typeof leaf.value==='number'?leaf.value+1:typeof leaf.value==='boolean'?!leaf.value:'forged';assert.equal(validateNongaussianFieldResult(preparation.input,changed),null,preparation.id+': '+leaf.path.join('.'));}
 assert.deepEqual(new Set(leaves.map(l=>l.path[0])),new Set(Object.keys(response.result)));rows.push({preparation:preparation.id,leaves:leaves.length,rejected:leaves.length,roots:[...new Set(leaves.map(l=>l.path[0]))]});}
 writeFileSync(root+'/leaf-mutations.json',JSON.stringify({preparations:rows,leaves:rows.reduce((n,r)=>n+r.leaves,0),rejected:rows.reduce((n,r)=>n+r.rejected,0)},null,2)+'\n');
});
test('request IDs,nested input attribution and raw spelling must match exactly',()=>{
 const input={...nongaussianFieldDefault,pairFactor:'4/2'},response=calculate({version:1,requestId:3,input});assert(response.ok);for(const data of [{version:1,requestId:4,input,response},{version:1,requestId:3,input:nongaussianFieldDefault,response},{version:1,requestId:3,input:{...input,sourceFactors:['2','1','1','1']},response}])assert.equal(validate(data).phase,'rejected');
 const changed=structuredClone(response);changed.result.input.probes[0]=1;assert.equal(validate({version:1,requestId:3,input,response:changed}).phase,'rejected');
 for(const requestId of [0,-1,1.5,Infinity,Number.MAX_SAFE_INTEGER+1])assert.equal(calculate({version:1,requestId,input}).ok,false);
 assert.equal(calculate({version:1,requestId:3,input:{...input,pairFactor:'0'}}).ok,false);
});
test('accessors,extra keys,cycles,symbols,prototype tricks and oversized payloads fail without getters',()=>{
 const response=calculate({version:1,requestId:4,input:nongaussianFieldDefault});assert(response.ok);let reads=0;const getter={...response.result};Object.defineProperty(getter,'stateCount',{get(){reads++;return 4;}});assert.equal(validateNongaussianFieldResult(nongaussianFieldDefault,getter),null);assert.equal(reads,0);
 const cycle:Record<string,unknown>={...response.result};cycle.self=cycle;const array=structuredClone(response.result);Object.defineProperty(array.mean,'extra',{value:'0'});
 for(const value of [cycle,array,{...response.result,extra:true},{...response.result,[Symbol('extra')]:1},{...response.result,contract:'x'.repeat(65537)},Object.create(response.result)])assert.equal(validateNongaussianFieldResult(nongaussianFieldDefault,value),null);
 const hostile=new Proxy({},{getPrototypeOf(){throw Error('no descriptors');}});assert.equal(calculate(hostile).ok,false);assert.equal(validate(hostile).phase,'rejected');
 const polluted=JSON.parse('{"__proto__":{"version":1,"requestId":4,"ok":true}}');assert.equal(calculate(polluted).ok,false);assert.equal(validate(polluted).phase,'rejected');
});
test('state counts,boundaries,source factors and cumulants cannot be relabelled',()=>{
 for(const id of ['pure-four-body','biased-repeated','odd-frustration','localized-source']){const input=nongaussianFieldExamples.find(c=>c.id===id)!.input,response=calculate({version:1,requestId:5,input});assert(response.ok);const changed=structuredClone(response);changed.result.stateCount+=1;assert.equal(validate({version:1,requestId:5,input,response:changed}).phase,'rejected');}
});
test('untrusted output fraction strings never enter independent arithmetic,even on a fresh input',async()=>{
 const{Rational}=await import('../src/lib/learning-graph-gauge-validation-math');const original=Rational.read,split=String.prototype.split,input={...nongaussianFieldDefault,temperature:'13/7'},response=calculate({version:1,requestId:6,input});assert(response.ok);const forged=structuredClone(response.result);forged.connectedTwo[0][0]='untrusted-fraction';let parsed=false;
 Rational.read=function(s:string){if(s==='untrusted-fraction')parsed=true;return original.call(this,s);};String.prototype.split=function(this:string,separator:unknown,limit?:number){if(String(this)==='untrusted-fraction')parsed=true;return Reflect.apply(split,this,[separator,limit]);}as typeof String.prototype.split;
 try{assert.equal(validateNongaussianFieldResult(input,forged),null);assert.equal(parsed,false);}finally{Rational.read=original;String.prototype.split=split;}
});
test('owned validator results cannot be poisoned to affect the bounded trusted-input cache',()=>{
 const input=nongaussianFieldDefault,response=calculate({version:1,requestId:7,input});assert(response.ok);const accepted=validateNongaussianFieldResult(input,response.result)!;assert(Object.isFrozen(accepted.connectedTwo[0]));assert.throws(()=>{accepted.connectedTwo[0][0]='bad';});assert(validateNongaussianFieldResult(input,response.result));
 for(let i=1;i<20;i++){const x={...input,temperature:`1/${i}`},r=calculate({version:1,requestId:7,input:x});assert(r.ok);assert(validateNongaussianFieldResult(x,r.result));}assert(validateNongaussianFieldResult(input,response.result));
});
test('reader hook excludes arithmetic and independent verification never imports the producer',async()=>{
 for(const entry of ['src/components/learning/use-nongaussian-field-calculation.ts','src/lib/learning-nongaussian-field-validation.worker.ts']){const bundle=await build({entryPoints:[entry],bundle:true,write:false,platform:'browser',format:'esm',metafile:true}),inputs=Object.keys(bundle.metafile!.inputs);for(const name of ['learning-nongaussian-field.ts','learning-nongaussian-field-evaluator.ts','learning-quantum-hilbert-core.ts'])assert(!inputs.includes('src/lib/'+name));if(entry.includes('use-gaussian')){assert(!inputs.includes('src/lib/learning-nongaussian-field-validation.ts'));assert(!bundle.outputFiles[0].text.includes('BigInt('));}writeFileSync(root+'/'+(entry.includes('use-gaussian')?'hook':'independent')+'-dependency-graph.json',JSON.stringify(bundle.metafile,null,2)+'\n');}
});
test('hook server-renders idle without touching Worker',async()=>{
 const React=await import('react'),{renderToStaticMarkup}=await import('react-dom/server'),{useNongaussianFieldCalculation}=await import('../src/components/learning/use-nongaussian-field-calculation');const saved=Object.getOwnPropertyDescriptor(globalThis,'Worker');let calls=0;Object.defineProperty(globalThis,'Worker',{configurable:true,get(){calls++;throw Error('SSR Worker');}});try{function Probe(){return React.createElement('span',null,useNongaussianFieldCalculation(nongaussianFieldDefault,true).status);}assert.equal(renderToStaticMarkup(React.createElement(Probe)),'<span>idle</span>');assert.equal(calls,0);}finally{if(saved)Object.defineProperty(globalThis,'Worker',saved);else Reflect.deleteProperty(globalThis,'Worker');}
});
