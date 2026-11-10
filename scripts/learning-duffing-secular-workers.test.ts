import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {build} from 'esbuild';
import {validateDuffingSecularResult} from '../src/lib/learning-duffing-secular-validation';
import {evaluateDuffingSecularRequest as calculate} from '../src/lib/learning-duffing-secular-evaluator';
import {evaluateDuffingSecularValidationRequest as validate} from '../src/lib/learning-duffing-secular-validation-evaluator';
import {duffingSecularDefault,duffingSecularExamples,type DuffingSecularInput} from '../src/lib/learning-duffing-secular-input';
import type {DuffingSecularSuccess} from '../src/lib/learning-duffing-secular-protocol';
const root='.runtime/learning/duffing-secular-worker';mkdirSync(root,{recursive:true});
const fixture=JSON.parse(readFileSync('scripts/fixtures/duffing-secular-reference.json','utf8'))as{cases:{id:string;input:DuffingSecularInput}[]};
for(const c of fixture.cases)test(`independent full duffing-secular response: ${c.id}`,()=>{
 const events:unknown[]=[],response=calculate({version:1,requestId:1,input:c.input},m=>events.push(m));assert(response.ok);assert.deepEqual(events,[{version:1,requestId:1,phase:'started',kind:'duffing-secular'}]);
 const accepted=validate({version:1,requestId:1,input:c.input,response});assert.equal(accepted.phase,'validated');if(accepted.phase==='validated'){assert.deepEqual(accepted.response,response);assert.notEqual(accepted.response,response);assert(Object.isFrozen(accepted.response.result));}
});
test('every leaf of sampled profiles,derivatives,intervals and metadata rejects individual alteration',()=>{
 const response=calculate({version:1,requestId:2,input:duffingSecularDefault});assert(response.ok);
 const leaves:{path:(string|number)[];value:unknown}[]=[];
 const walk=(v:unknown,path:(string|number)[])=>{if(v===null||typeof v!=='object'){leaves.push({path,value:v});return;}for(const[k,x]of Object.entries(v))walk(x,[...path,Array.isArray(v)?Number(k):k]);};walk(response.result,[]);
 for(const leaf of leaves){const changed:DuffingSecularSuccess['result']=structuredClone(response.result);let node:unknown=changed;for(const key of leaf.path.slice(0,-1))node=(node as Record<string|number,unknown>)[key];(node as Record<string|number,unknown>)[leaf.path.at(-1)!]=typeof leaf.value==='string'?leaf.value+'!':typeof leaf.value==='number'?leaf.value+1:typeof leaf.value==='boolean'?!leaf.value:'forged';assert.equal(validateDuffingSecularResult(duffingSecularDefault,changed),null,leaf.path.join('.'));}
 writeFileSync(root+'/leaf-mutations.json',JSON.stringify({rejected:leaves.length,paths:leaves.map(x=>x.path.join('.'))},null,2)+'\n');for(const root of ['input','elliptic','probe','fullSamples','localSamples','horizonDiagnostics','contract'])assert(leaves.some(leaf=>leaf.path[0]===root),root+' mutation coverage');
});
test('raw spellings,request identity and invalid-input boundaries retain attribution',()=>{
 const input=duffingSecularDefault,response=calculate({version:1,requestId:3,input});assert(response.ok);
 for(const changed of [{...response,requestId:99},{...response,inputKey:'wrong'},{...response,kind:'wrong'},{...response,input:{...input,lambda:'2/200'}}])assert.equal(validate({version:1,requestId:3,input,response:changed}).phase,'rejected');
 assert.equal(calculate({version:1,requestId:0,input}).ok,false);assert.equal(calculate({version:1,requestId:1,input:{...input,lambda:'17'}}).ok,false);assert.equal(validate({version:1,requestId:3,input:{...input,horizon:Infinity},response}).phase,'rejected');
});
test('accessors,extra fields,cycles,symbols and oversized strings are rejected without getters',()=>{
 const response=calculate({version:1,requestId:4,input:duffingSecularDefault});assert(response.ok);let reads=0;
 const getter={...response.result};Object.defineProperty(getter,'initialEnergy',{get(){reads++;return 192;}});assert.equal(validateDuffingSecularResult(duffingSecularDefault,getter),null);assert.equal(reads,0);
 const cycle:Record<string,unknown>={...response.result};cycle.self=cycle;for(const v of [cycle,{...response.result,extra:1},{...response.result,[Symbol('bad')]:1},{...response.result,contract:'x'.repeat(65537)}])assert.equal(validateDuffingSecularResult(duffingSecularDefault,v),null);
});
test('equilibrium,harmonic,tiny coupling and invalid truncated frequencies cannot be relabelled',()=>{
 for(const id of ['equilibrium','harmonic','tiny-coupling-long-time','negative-truncated-frequency']){const input=duffingSecularExamples.find(c=>c.id===id)!.input,response=calculate({version:1,requestId:5,input});assert(response.ok);const changed=structuredClone(response) as DuffingSecularSuccess;changed.result.mode=changed.result.mode==='equilibrium'?'harmonic':'equilibrium';assert.equal(validate({version:1,requestId:5,input,response:changed}).phase,'rejected');}
});
test('untrusted output fractions never become independent arithmetic operands',async()=>{
 const{Rational}=await import('../src/lib/learning-graph-gauge-validation-math');const original=Rational.read,originalSplit=String.prototype.split,input={...duffingSecularDefault,probeFraction:'3/17'},response=calculate({version:1,requestId:6,input});assert(response.ok);const forged=structuredClone(response.result);forged.probe.exactPosition.lower='untrusted-fraction';let parsed=false;
 Rational.read=function(value:string){if(value==='untrusted-fraction'){parsed=true;throw Error('untrusted arithmetic');}return original.call(this,value);};String.prototype.split=function(this:string,separator:unknown,limit?:number){if(String(this)==='untrusted-fraction')parsed=true;return Reflect.apply(originalSplit,this,[separator,limit]);} as typeof String.prototype.split;try{assert.equal(validateDuffingSecularResult(input,forged),null);assert.equal(parsed,false);}finally{Rational.read=original;String.prototype.split=originalSplit;}
});
test('reader hook excludes arithmetic and independent verifier excludes all producer engines',async()=>{
 for(const entry of ['src/components/learning/use-duffing-secular-calculation.ts','src/lib/learning-duffing-secular-validation.worker.ts']){const bundle=await build({entryPoints:[entry],bundle:true,write:false,platform:'browser',format:'esm',metafile:true});const inputs=Object.keys(bundle.metafile!.inputs);
  for(const forbidden of ['learning-duffing-secular.ts','learning-duffing-secular-evaluator.ts','learning-quantum-hilbert-core.ts','learning-interval-transcendentals.ts','learning-quantum-evolution-interval.ts','learning-directed-pi.ts'])assert(!inputs.includes('src/lib/'+forbidden));
  if(entry.includes('use-duffing')){assert(!inputs.includes('src/lib/learning-duffing-secular-validation.ts'));assert(!bundle.outputFiles[0].text.includes('BigInt('));}
  writeFileSync(root+'/'+(entry.includes('use-duffing')?'hook':'independent')+'-dependency-graph.json',JSON.stringify(bundle.metafile,null,2)+'\n');
 }
});
test('hook server-renders idle without creating a Worker',async()=>{
 const React=await import('react'),{renderToStaticMarkup}=await import('react-dom/server'),{useDuffingSecularCalculation}=await import('../src/components/learning/use-duffing-secular-calculation');const saved=Object.getOwnPropertyDescriptor(globalThis,'Worker');let calls=0;Object.defineProperty(globalThis,'Worker',{configurable:true,get(){calls++;throw Error('SSR Worker');}});
 try{function Probe(){return React.createElement('span',null,useDuffingSecularCalculation(duffingSecularDefault,true).status);}assert.equal(renderToStaticMarkup(React.createElement(Probe)),'<span>idle</span>');assert.equal(calls,0);}finally{if(saved)Object.defineProperty(globalThis,'Worker',saved);else Reflect.deleteProperty(globalThis,'Worker');}
});
