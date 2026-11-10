import test from 'node:test';
import assert from 'node:assert/strict';
import {build} from 'esbuild';
import {renderToString} from 'react-dom/server';
import {createElement} from 'react';
import {quantumInformationPresets} from '../src/lib/learning-quantum-information-presets';
import {evaluateInformationRequest as calculate} from '../src/lib/learning-quantum-information-evaluator';
import {evaluateInformationValidationRequest as validate} from '../src/lib/learning-quantum-information-validation-evaluator';
import {useInformationCalculation} from '../src/components/learning/use-information-calculation';
import type {InformationStarted} from '../src/lib/learning-quantum-information-protocol';
const presets=Object.values(quantumInformationPresets).flat();
for(const p of presets)test(`worker envelopes/independent receipt ${p.input.kind}/${p.id}`,()=>{
 const started:InformationStarted[]=[],result=calculate({version:1,requestId:7,input:p.input},m=>started.push(m));assert(result.ok);
 assert.deepEqual(started,[{version:1,requestId:7,phase:'started',kind:p.input.kind}]);
 const receipt=validate({version:1,requestId:7,input:p.input,response:result});assert.equal(receipt.phase,'validated');
 if(receipt.phase==='validated'){assert.deepEqual(receipt.response.result,result.result);assert(Object.isFrozen(receipt.response.result));}
 const changed=structuredClone(result);(changed.result as unknown as Record<string,unknown>).extra=true;
 assert.equal(validate({version:1,requestId:7,input:p.input,response:changed}).phase,'rejected');
 assert.equal(validate({version:1,requestId:8,input:p.input,response:result}).phase,'rejected');
});
test('request ownership rejects getters, extra fields, unsafe IDs and invalid inputs',()=>{
 let calls=0;const getter={get version(){calls++;return 1;},requestId:1,input:presets[0].input};assert.equal(calculate(getter).ok,false);assert.equal(calls,0);
 for(const id of [0,-1,1.5,NaN,Infinity,Number.MAX_SAFE_INTEGER+1])assert.equal(calculate({version:1,requestId:id,input:presets[0].input}).ok,false);
 for(const x of [{version:2,requestId:1,input:presets[0].input},{version:1,requestId:1,input:{kind:'unknown'}},{version:1,requestId:1,input:presets[0].input,extra:0}])assert.equal(calculate(x).ok,false);
 const result=calculate({version:1,requestId:1,input:{kind:'capacity',probability:'1/10',input:['1','1','0']}});assert(!result.ok&&result.error==='invalid-input');
});
test('maximum sample/rank/operator boundaries fit the transport budgets',()=>{
 const scalar=(x:string)=>({real:[x,'0'],imaginary:['0','0']});
 const inputs=[{kind:'bb84',input:[scalar('1'),scalar('1')],comparison:[scalar('1'),scalar('0')],attackFraction:'1/99999999999999999999999999999999',eveZProbability:'1/2',sampleSize:256},{kind:'entropy',correlationWeight:'1',coherence:'1/3',probabilityOne:'1/3',copies:64,retainedRank:String(BigInt(1)<<BigInt(64))},{kind:'error-correction',input:[scalar('1'),scalar('1')],selectedError:'YYY',physicalXProbability:'1/3',errors:Array(16).fill('III')}];
 for(const input of inputs){const response=calculate({version:1,requestId:1,input});assert(response.ok);assert.equal(validate({version:1,requestId:1,input,response}).phase,'validated');}
});
test('server rendering starts no Workers or calculations',()=>{
 function Probe(){const state=useInformationCalculation(presets[0].input,true);return createElement('span',null,state.status);}
 assert.equal(renderToString(createElement(Probe)),'<span>idle</span>');
});
test('controller and hook bundles exclude mathematics; validator excludes producers',async()=>{
 for(const entry of ['src/lib/learning-quantum-information-controller.ts','src/components/learning/use-information-calculation.ts','src/lib/learning-quantum-information-validation.worker.ts']){
  const r=await build({entryPoints:[entry],bundle:true,write:false,platform:'browser',format:'esm',metafile:true});
  const inputs=Object.keys(r.metafile!.inputs);
  if(entry.includes('validation.worker'))assert(!inputs.some(p=>/learning-quantum-information-(?:calculation|capacity|discrimination|ensemble|entropies|joint-entropy|relative-entropy|bb84|protocols|error-correction|majorization|bell-dilution)\.ts$/.test(p)));
  else{assert(!inputs.some(p=>/learning-quantum-hilbert-core|learning-quantum-information-(?:calculation|response|.*validator|evaluator)/.test(p)));assert(!r.outputFiles[0].text.includes('BigInt('));}
 }
});
