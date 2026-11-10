import test from 'node:test';
import assert from 'node:assert/strict';
import {createElement} from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {build} from 'esbuild';
import {realTimePhaseExamples} from '../src/lib/learning-real-time-phase-input';
import {realTimePhaseExampleLabels} from '../src/data/real-time-phase-teaching';
import {realTimeLearningLimits as limits} from '../src/components/learning/use-saved-real-time-phase';
import {emptyRealTimeState,readRealTimeState,parseStoredRealTimeState,realTimeDraft,readRealTimeDraft,parseRealTimeDraft,readRealTimeWave,readRealTimeMatrix} from '../src/lib/learning-real-time-phase-state';
import Lab from '../src/components/learning/real-time-phase-lab';
import Wave from '../src/components/learning/real-time-phase-wave';
import Matrix from '../src/components/learning/real-time-phase-matrix';
import {calculateRealTimePhase} from '../src/lib/learning-real-time-phase';
import {evaluateRealTimePhaseValidationRequest} from '../src/lib/learning-real-time-phase-validation-evaluator';
import {realTimePhaseInputKey,type RealTimePhaseValidatedSuccess} from '../src/lib/learning-real-time-phase-protocol';

test('all original inputs round trip through editable drafts and saved state without reducing bounds',()=>{
  assert.equal(realTimePhaseExamples.length,32);assert.deepEqual(Object.keys(realTimePhaseExampleLabels).sort(),realTimePhaseExamples.map(e=>e.id).sort());
  for(const {id,input} of realTimePhaseExamples){const draft=realTimeDraft(input);assert.deepEqual(parseRealTimeDraft(draft),input,id);const original=emptyRealTimeState(limits,input),stored=JSON.stringify({version:1,model:original}),copy=parseStoredRealTimeState(stored,limits)!;assert.deepEqual(copy,original);assert.notEqual(copy,original);assert(Object.isFrozen(copy.input)&&Object.isFrozen(copy.draft.fields)&&Object.isFrozen(copy.wave)&&Object.isFrozen(copy.matrix));assert(!/[가-힣]/.test(realTimePhaseExampleLabels[id].en));}
});
test('invalid unfinished drafts persist separately while applied model remains valid',()=>{
  const state=emptyRealTimeState(limits),draft={...state.draft,fields:{...state.draft.fields,time:'',intervals:'3.0',probe:'99'}};
  assert(readRealTimeDraft(draft));assert.equal(parseRealTimeDraft(draft),null);const restored=parseStoredRealTimeState(JSON.stringify({version:1,model:{...state,draft}}),limits)!;assert.deepEqual(restored.input,state.input);assert.deepEqual(restored.draft,draft);
  for(const value of ['3.0','3e0',' 3','+3','03','', '32.1'])assert.equal(parseRealTimeDraft({...state.draft,fields:{...state.draft.fields,intervals:value}}),null,value);
  assert.equal(readRealTimeDraft({...draft,extra:1}),null);assert.equal(readRealTimeDraft({...draft,fields:{...draft.fields,time:'1'.repeat(33)}}),null);
});
test('saved preferences own learning text and reject corrupt, injected, oversized or accessor state',()=>{
  const state=emptyRealTimeState(limits),valid={...state,step:12,lesson:{...state.lesson,guide:5,explanationOpen:true,predictions:Array(6).fill('\u0000'.repeat(2000)),answers:Array(8).fill(2)},matrix:{...state.matrix,kind:'hessian',selectedCell:960,tableOpen:true,flat:false},wave:{mode:31,shift:-2,sample:64,comparisonHbar:'1/32',comparisonLog:true,foldQuantity:'log'}};
  assert(parseStoredRealTimeState(JSON.stringify({version:1,model:valid}),limits));
  for(const patch of [{step:13},{step:-1},{input:{...state.input,hbar:'0'}},{extra:1},{lesson:{...state.lesson,answers:Array(8).fill(3)}},{matrix:{...state.matrix,selectedCell:961}},{wave:{...state.wave,sample:65}}])assert.equal(readRealTimeState({...state,...patch},limits),null);
  for(const raw of ['null','{','[]',JSON.stringify({version:2,model:state}),JSON.stringify({version:1,model:state,response:{ok:true}}),' '.repeat(100001)])assert.equal(parseStoredRealTimeState(raw,limits),null);
  let reads=0;const fields={...state.draft.fields};Object.defineProperty(fields,'time',{get(){reads++;return'1';}});assert.equal(readRealTimeDraft({...state.draft,fields}),null);
  const root={...state};Object.defineProperty(root,'input',{get(){reads++;return state.input;}});assert.equal(readRealTimeState(root,limits),null);assert.equal(reads,0);
});
test('display bounds preserve N32 cells and all sample indices, without run flags or results',()=>{
  const s=emptyRealTimeState(limits);
  for(const patch of [{mode:0},{mode:32},{shift:0.1},{shift:Infinity},{sample:64.5},{comparisonHbar:'1/7'},{comparisonLog:0},{foldQuantity:'phase'},{enabled:true}])assert.equal(readRealTimeWave({...s.wave,...patch}),null);
  for(const patch of [{kind:'posterior'},{component:'amplitude'},{scale:'log'},{selectedCell:-1},{tableOpen:'yes'},{flat:0},{receipt:{}}])assert.equal(readRealTimeMatrix({...s.matrix,...patch}),null);
  assert(readRealTimeWave({...s.wave,mode:31,sample:64,shift:2}));assert(readRealTimeMatrix({...s.matrix,selectedCell:960}));
});
test('lab SSR is bilingual and exposes no unverified result before restore',()=>{
  for(const language of ['ko','en'] as const)for(const embedded of [false,true]){const html=renderToStaticMarkup(createElement(Lab,{language,embedded}));assert(html.includes('data-real-time-storage-ready="false"'));assert(!html.includes('data-real-time-wave='));assert(!html.includes('data-real-time-selected'));if(language==='en')assert(!/[가-힣]/.test(html));}
});
for(const name of ['caustic-compatible','fixed-width-free','no-real-stationary-points'])test('controlled display renders verified response: '+name,()=>{
  const input=realTimePhaseExamples.find(e=>e.id===name)!.input,result=calculateRealTimePhase(input),response={version:1 as const,requestId:1,ok:true as const,kind:'real-time-phase' as const,input,inputKey:realTimePhaseInputKey(input),result};
  const receipt=evaluateRealTimePhaseValidationRequest({version:1,requestId:1,input,response});assert.equal(receipt.phase,'validated');const owned=(receipt as {response:RealTimePhaseValidatedSuccess}).response,state=emptyRealTimeState(limits,input);let writes=0;
  for(const language of ['ko','en'] as const){const html=renderToStaticMarkup(createElement(Wave,{response:owned,language,display:{...state.wave,mode:31,sample:64,shift:2,comparisonHbar:'1/8',comparisonLog:true,foldQuantity:'log'},onDisplay:()=>writes++}));assert(html.includes('data-real-time-wave='));if(language==='en')assert(!/[가-힣]/.test(html));
    const matrix=renderToStaticMarkup(createElement(Matrix,{response:owned,language,preferences:{...state.matrix,kind:'final-covariance',tableOpen:true,selectedCell:960},onPreferences:()=>writes++}));if(input.mode==='packet'){assert(matrix.includes('data-real-time-matrix="final-covariance"'));assert(matrix.includes('Full matrix interval table')||language==='ko');}if(language==='en')assert(!/[가-힣]/.test(matrix));
  }assert.equal(writes,0);
});
test('whole lab browser graph keeps producer and independent mathematical engines out of the reader',async()=>{
  const output=await build({entryPoints:['src/components/learning/real-time-phase-lab.tsx'],bundle:true,write:false,splitting:true,outdir:'unused',platform:'browser',format:'esm',jsx:'automatic',metafile:true});const inputs=Object.keys(output.metafile!.inputs);
  assert(!inputs.some(p=>/learning-real-time-phase(?:\.ts|-exact|-validation\.ts|-evaluator)|learning-quantum-hilbert-core/.test(p)),inputs.join('\n'));
});
