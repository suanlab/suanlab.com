import test from 'node:test';
import assert from 'node:assert/strict';
import {createElement} from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {build} from 'esbuild';
import {mkdirSync,writeFileSync} from 'node:fs';
import {calculateRealTimePhase} from '../src/lib/learning-real-time-phase';
import {evaluateRealTimePhaseValidationRequest} from '../src/lib/learning-real-time-phase-validation-evaluator';
import {realTimePhaseExamples,readRealTimePhaseInput} from '../src/lib/learning-real-time-phase-input';
import {realTimePhaseInputKey,type RealTimePhaseValidatedSuccess} from '../src/lib/learning-real-time-phase-protocol';
import {realTimeChainDrawing,realTimePacketDrawing,realTimeFoldControls,realTimeContinuumDiagnostic} from '../src/lib/learning-real-time-phase-wave-view';
import Wave from '../src/components/learning/real-time-phase-wave';
const root='.runtime/learning/real-time-phase-waves';mkdirSync(root,{recursive:true});const cases:unknown[]=[];
for(const example of realTimePhaseExamples)test(`owned wave view: ${example.id}`,()=>{
  const input=example.input,result=calculateRealTimePhase(input),response={version:1 as const,requestId:1,ok:true as const,kind:'real-time-phase' as const,input,inputKey:realTimePhaseInputKey(input),result};
  const receipt=evaluateRealTimePhaseValidationRequest({version:1,requestId:1,input,response});assert.equal(receipt.phase,'validated');const owned=(receipt as {phase:'validated';response:RealTimePhaseValidatedSuccess}).response;
  if(result.kind!=='canonical-fold')assert(realTimeContinuumDiagnostic(owned)!.flat().every(Number.isFinite));
  if(result.kind==='chain'){
    let negative=0,positive=0,zero=0;
    for(let k=1;k<result.input.intervals;k++){
      const v=realTimeChainDrawing(owned,k)!;assert.equal(v.mode.length,result.input.intervals+1);assert.equal(v.mode[0].y,0);assert.equal(v.mode.at(-1)!.y,0);
      if(v.status==='negative')negative++;else if(v.status==='positive')positive++;else zero++;
      if(v.status==='zero')assert.equal(v.diagnosticEigenvalue,null);
      assert(v.mode.every(p=>Math.abs(p.y)<=1));
    }
    assert.deepEqual({negative,positive,zero},result.exact.inertia);
    const v=realTimeChainDrawing(owned,1)!;assert(v.unregulatedPhase.real!==null&&v.unregulatedPhase.imaginary!==null);assert(Math.abs(Math.hypot(v.unregulatedPhase.real,v.unregulatedPhase.imaginary)-1)<1e-12);
    if(example.id==='negative-mode-hand'){assert.equal(v.status,'negative');assert(Math.abs(v.unregulatedPhase.real-Math.cos(.5-3*Math.PI/4))<1e-12);}
    if(result.exact.stationaryStatus==='no-stationary-path')assert.equal(v.path,null);
  }else if(result.kind==='packet'){
    const v=realTimePacketDrawing(owned)!;assert.equal(v.points.length,65);
    for(const p of v.points){assert(p.density!==null&&p.real!==null&&p.imaginary!==null&&p.phase.real!==null&&p.phase.imaginary!==null);
      assert(Math.abs(p.density-Math.exp(-p.x*p.x/2)/Math.sqrt(2*Math.PI))<1e-12);
      assert(Math.abs(Math.hypot(p.real,p.imaginary)-Math.exp(-p.x*p.x/4))<1e-12);
      assert(Math.abs(Math.hypot(p.phase.real,p.phase.imaginary)-1)<1e-12);
    }
  }else{
    assert.equal(realTimeFoldControls.length,33);for(const control of realTimeFoldControls)assert(readRealTimePhaseInput({mode:'fold',control,hbar:input.hbar}));
  }
  for(const language of ['ko','en'] as const){const html=renderToStaticMarkup(createElement(Wave,{response:owned,language}));assert(html.includes('data-real-time-wave'));assert(!/NaN|Infinity|undefined/.test(html));if(language==='en')assert(!/[가-힣]/u.test(html));if(result.kind==='chain'&&result.wave.unregulated.kind==='parity-delta')assert(html.includes('data-real-time-delta'));}
  cases.push({id:example.id,kind:result.kind});writeFileSync(root+'/view-progress.json',JSON.stringify({cases},null,2)+'\n');
});
test('wave/series reader excludes numerical engines and keeps native Worker factories',async()=>{
  const output=await build({entryPoints:['src/components/learning/real-time-phase-wave.tsx'],bundle:true,write:false,platform:'browser',format:'esm',jsx:'automatic',metafile:true});
  const inputs=Object.keys(output.metafile!.inputs);assert(inputs.includes('src/lib/learning-real-time-phase-worker-factory.ts'));
  assert(!inputs.some(p=>/learning-real-time-phase(?:\.ts|-exact|-validation\.ts|-evaluator)|learning-quantum-hilbert-core|three\//.test(p)),inputs.join('\n'));
});
