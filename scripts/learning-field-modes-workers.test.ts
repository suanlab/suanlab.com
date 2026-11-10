import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {build} from 'esbuild';
import {calculateFieldModes} from '../src/lib/learning-field-modes';
import {validateFieldModesResult} from '../src/lib/learning-field-modes-validation';
import {fieldModesExamples} from '../src/lib/learning-field-modes-examples';
import {evaluateFieldModesRequest as calculate} from '../src/lib/learning-field-modes-evaluator';
import {evaluateFieldModesValidationRequest as validate} from '../src/lib/learning-field-modes-validation-evaluator';
import type {FieldModesInput} from '../src/lib/learning-field-modes-input';
const refs=JSON.parse(readFileSync('scripts/fixtures/field-modes-independent.json','utf8'))as{cases:{label:string;input:FieldModesInput}[]};
for(const c of refs.cases)test(`independent recurrence,Rayleigh and projection validation: ${c.label}`,()=>{assert(validateFieldModesResult(c.input,calculateFieldModes(c.input)));});
test('receipts own the result and reject request/input attribution changes',()=>{
 const input=fieldModesExamples[0].input,response=calculate({version:1,requestId:4,input});assert(response.ok);const checked=validate({version:1,requestId:4,input,response});assert.equal(checked.phase,'validated');if(checked.phase==='validated')assert(Object.isFrozen(checked.response.result));
 for(const message of [{version:1,requestId:5,input,response},{version:1,requestId:4,input:{...input,length:'4/1'},response},{version:1,requestId:4,input:{...input,selectedBasis:2},response}])assert.equal(validate(message).phase,'rejected');
});
test('wrong point values,loose intervals,wrong basis signs and refinement claims are rejected',()=>{
 const input=fieldModesExamples[0].input,r=calculateFieldModes(input);
 const mutations:((x:typeof r)=>void)[]=[];
 for(const key of Object.keys(r.exact)as(keyof typeof r.exact)[])mutations.push(x=>{x.exact[key]='999';});
 for(const key of ['eigenvalue','coordinate','velocity','quadraticEnergy','massNorm','wavenumber']as const)mutations.push(x=>{x.modes[1][key]={lower:'999',upper:'999'};});
 mutations.push(x=>{x.modes[1].eigenvalue={lower:'-999',upper:'999'};},x=>{x.modes[1].classification='unstable';},x=>{x.modes[1].exactEigenvalue='3';},x=>{x.modes[1].family='sine';},x=>{x.selected.shape[0]={lower:'-1',upper:'-1'};},x=>{x.selected.operatorResidual[0]={lower:'1',upper:'1'};},x=>{x.referenceIsEquilibrium=false;},x=>{x.referenceHasBoundaryJump=true;},x=>{(x as unknown as Record<string,unknown>).timeDiscretized=true;},x=>{x.refinements[1].sites=4;},x=>{x.refinements[1].physicalWave=2;},x=>{x.refinements[1].waveOperatorGap={lower:'1',upper:'1'};},x=>{x.refinements[1].masslessPhaseSpeedRatio={lower:'1',upper:'1'};},x=>{x.checks.positionParsevalResidual={lower:'1',upper:'1'};});
 for(const change of mutations){const bad=structuredClone(r);change(bad);assert.equal(validateFieldModesResult(input,bad),null);}
 assert.equal(mutations.length,30);
});
test('wire getters,malformed headers and oversized claims cannot execute caller code',()=>{
 const input=fieldModesExamples[0].input;let reads=0;const frame={get version(){reads++;throw Error('getter');},requestId:1,input};assert.equal(calculate(frame).ok,false);assert.equal(reads,0);
 const r=calculateFieldModes(input);Object.defineProperty(r,'contract',{get(){reads++;throw Error('getter');}});assert.equal(validateFieldModesResult(input,r),null);assert.equal(reads,0);
 const response=structuredClone(calculate({version:1,requestId:1,input}));assert(response.ok);response.result.contract='x'.repeat(65537);const receipt=validate({version:1,requestId:1,input,response});assert.equal(receipt.phase,'rejected');if(receipt.phase==='rejected')assert.equal(receipt.error,'resource-limit');
});
test('reader bundle has no mathematical engines; independent validation excludes production trig,roots and modes',async()=>{
 for(const entry of ['src/components/learning/use-field-modes-calculation.ts','src/lib/learning-field-modes-validation.worker.ts']){const r=await build({entryPoints:[entry],bundle:true,write:false,platform:'browser',format:'esm',metafile:true,logLevel:'silent'}),inputs=Object.keys(r.metafile!.inputs);assert(!inputs.includes('src/lib/learning-field-modes.ts'));assert(!inputs.includes('src/lib/learning-spin-angle.ts'));assert(!inputs.includes('src/lib/learning-quantum-evolution-interval.ts'));assert(!inputs.includes('src/lib/learning-quantum-hilbert-core.ts'));if(!entry.includes('validation.worker'))assert(!r.outputFiles[0].text.includes('BigInt('));}
});
for(const boundary of ['periodic','dirichlet-zero','neumann-zero']as const)test(`independent validation of extreme32-site rational result: ${boundary}`,()=>{
 const input={...fieldModesExamples.at(-2)!.input,boundary,length:'1/999999999999999999',speed:'999999999999999999',quadratic:'-999999999999999999',quartic:'999999999999999999',background:'999999999999999999',position:Array.from({length:32},(_,i)=>i%2?'999999999999999999':'-999999999999999999'),velocity:Array.from({length:32},()=> '1/999999999999999999')};assert(validateFieldModesResult(input,calculateFieldModes(input)));
});
test('server rendering stays idle and does not access the Worker global',async()=>{
 const{createElement}=await import('react'),{renderToString}=await import('react-dom/server'),{useFieldModesCalculation}=await import('../src/components/learning/use-field-modes-calculation');const before=Object.getOwnPropertyDescriptor(globalThis,'Worker');let reads=0;Object.defineProperty(globalThis,'Worker',{configurable:true,get(){reads++;throw Error('SSR Worker');}});
 try{function Probe(){return createElement('span',null,useFieldModesCalculation(fieldModesExamples[0].input,true).status);}assert.equal(renderToString(createElement(Probe)),'<span>idle</span>');assert.equal(reads,0);}finally{if(before)Object.defineProperty(globalThis,'Worker',before);else Reflect.deleteProperty(globalThis,'Worker');}
});
