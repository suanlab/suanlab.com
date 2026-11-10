import assert from 'node:assert/strict';
import test from 'node:test';
import {readFileSync} from 'node:fs';
import {build} from 'esbuild';
import {validateScalarFieldResult} from '../src/lib/learning-scalar-field-validation';
import {evaluateScalarFieldRequest as calculate} from '../src/lib/learning-scalar-field-evaluator';
import {evaluateScalarFieldValidationRequest as validate} from '../src/lib/learning-scalar-field-validation-evaluator';
import {scalarFieldExamples} from '../src/lib/learning-scalar-field-examples';
import {scalarFieldInputKey,type ScalarFieldInput} from '../src/lib/learning-scalar-field-input';
const fixture=JSON.parse(readFileSync('scripts/fixtures/scalar-field-lattice-reference.json','utf8'))as{cases:{label:string;input:ScalarFieldInput;expected:Record<string,unknown>}[]};
for(const row of fixture.cases)test(`energy-difference validator independently accepts symbolic reference: ${row.label}`,()=>{
 const result={input:row.input,sites:row.input.position.length,timeDiscretized:false,...row.expected};assert(validateScalarFieldResult(row.input,result));
});
for(const item of scalarFieldExamples)test(`owned calculation and validation protocol: ${item.id}`,()=>{
 const input=item.input,started:unknown[]=[],response=calculate({version:1,requestId:1,input},x=>started.push(x));assert(response.ok);assert.equal(started.length,1);assert.equal(response.inputKey,scalarFieldInputKey(input));
 const receipt=validate({version:1,requestId:1,input,response});assert.equal(receipt.phase,'validated');if(receipt.phase==='validated'){assert.deepEqual(receipt.response,response);assert(Object.isFrozen(receipt.response.result));}
});
test('all scalar,vector,matrix,edge,geometry and attribution claims are checked',()=>{
 const input=fixture.cases[0].input,response=calculate({version:1,requestId:5,input});assert(response.ok);
 const mutate=(alter:(r:Record<string,unknown>)=>void)=>{const changed=structuredClone(response);alter(changed.result as unknown as Record<string,unknown>);assert.equal(validate({version:1,requestId:5,input,response:changed}).phase,'rejected');};
 for(const key of ['length','massPerSite','kineticEnergy','gradientEnergy','localPotentialEnergy','energy','potentialPower','kineticPower','energyDerivative','totalMomentum','momentumDerivative','localPotentialForce','boundaryForce','localPotentialLowerBound'])mutate(r=>{r[key]=r[key]==='0'?'1':'0';});
 for(const key of ['positions','kineticBySite','localPotentialBySite','potentialGradient','acceleration'])mutate(r=>{const a=r[key]as string[];a[0]=a[0]==='0'?'1':'0';});
 mutate(r=>{(r.potentialHessian as string[][])[0][0]='0';});mutate(r=>{(r.potentialHessian as string[][])[0][1]='0';});
 for(const key of ['left','right','difference','energy'])mutate(r=>{const e=(r.edges as Record<string,unknown>[])[0];e[key]=key==='left'||key==='right'?null:e[key]==='0'?'1':'0';});
 mutate(r=>{r.sites=99;});mutate(r=>{r.timeDiscretized=true;});mutate(r=>{r.extra=true;});mutate(r=>{(r.potentialGradient as string[]).pop();});
});
test('input spelling,boundary,request and phase ownership reject another valid response',()=>{
 const input=scalarFieldExamples[0].input,response=calculate({version:1,requestId:4,input});assert(response.ok);
 for(const next of [{...input,boundary:'dirichlet-zero'},{...input,spacing:'2/2'},{...input,position:['0','0','0']}])assert.equal(validate({version:1,requestId:4,input:next,response}).phase,'rejected');
 assert.equal(validate({version:1,requestId:3,input,response}).phase,'rejected');
 const changed=structuredClone(response);changed.result.input={...input,boundary:'neumann-zero'};assert.equal(validate({version:1,requestId:4,input,response:changed}).phase,'rejected');
});
test('protocols reject malformed versions,IDs,extra fields and accessor payloads',()=>{
 const input=scalarFieldExamples[0].input;for(const raw of [{version:2,requestId:1,input},{version:1,requestId:0,input},{version:1,requestId:1.5,input},{version:1,requestId:1,input,extra:true},{version:1,requestId:1,input:{...input,quartic:'-1'}}])assert.equal(calculate(raw).ok,false);
 let reads=0;const bad={version:1,requestId:1,input};Object.defineProperty(bad,'input',{get(){reads++;throw Error('Accessor invoked');}});assert.equal(calculate(bad).ok,false);
 const response=calculate({version:1,requestId:1,input});assert(response.ok);const result=structuredClone(response.result);Object.defineProperty(result,'energy',{get(){reads++;throw Error('Accessor invoked');}});assert.equal(validateScalarFieldResult(input,result),null);assert.equal(reads,0);
 const huge=structuredClone(response);huge.result.energy='1'.repeat(65537);const rejected=validate({version:1,requestId:1,input,response:huge});assert.equal(rejected.phase,'rejected');if(rejected.phase==='rejected')assert.equal(rejected.error,'resource-limit');
});
test('reader hook and controller exclude all scalar mathematical engines; validator excludes producer',async()=>{
 for(const[entry,forbidden]of [['src/components/learning/use-scalar-field-calculation.ts',/learning-scalar-field(?:\.ts|-validation\.ts|-evaluator\.ts|-validation-evaluator\.ts)/],['src/lib/learning-scalar-field-validation.worker.ts',/learning-scalar-field(?:\.ts|-evaluator\.ts)/]]as const){
  const r=await build({entryPoints:[entry],bundle:true,write:false,platform:'browser',format:'esm',metafile:true,logLevel:'silent'});assert(!Object.keys(r.metafile!.inputs).some(p=>forbidden.test(p)));
 }
});

test('server rendering the hook reads no Worker and starts no mathematical work',async()=>{
 const{createElement}=await import('react'),{renderToString}=await import('react-dom/server'),{useScalarFieldCalculation}=await import('../src/components/learning/use-scalar-field-calculation');
 const previous=Object.getOwnPropertyDescriptor(globalThis,'Worker');let reads=0;Object.defineProperty(globalThis,'Worker',{configurable:true,get(){reads++;throw Error('SSR Worker access');}});
 try{function Probe(){const state=useScalarFieldCalculation(scalarFieldExamples[0].input,true);return createElement('span',null,state.status);}assert.equal(renderToString(createElement(Probe)),'<span>idle</span>');assert.equal(reads,0);}finally{if(previous)Object.defineProperty(globalThis,'Worker',previous);else Reflect.deleteProperty(globalThis,'Worker');}
});

const budget=JSON.parse(readFileSync('scripts/fixtures/scalar-field-budget-reference.json','utf8'))as{cases:{label:string;input:ScalarFieldInput;expectedEnergy:string}[]};
for(const c of budget.cases)test(`18-digit,32-site transport budget and independent exact energy: ${c.label}`,()=>{
 const response=calculate({version:1,requestId:19,input:c.input});assert(response.ok);assert.equal(response.result.energy,c.expectedEnergy);assert.equal(validate({version:1,requestId:19,input:c.input,response}).phase,'validated');
});
