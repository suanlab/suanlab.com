import {observe,f,q as exactObservation} from '../src/lib/learning-quantum-hilbert-core';
import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';import {build} from 'esbuild';import {Worker} from 'node:worker_threads';
import {evaluateUncertaintyRequest as evaluate} from '../src/lib/learning-uncertainty-evaluator';
import {UncertaintyCalculationController as Controller,type UncertaintyWorkerPort} from '../src/lib/learning-uncertainty-controller';
import {uncertaintyKinds as kinds,snapshotUncertaintyInput as snapshot,uncertaintyInputKey as key,isUncertaintyInput as valid,isUncertaintySuccessForInput as success,parseUncertaintyInputDraft as draft,parseUncertaintyStoredInput as stored,type UncertaintyInput} from '../src/lib/learning-uncertainty-protocol';
const families=kinds.map((kind,i)=>({kind,cases:JSON.parse(readFileSync(`scripts/fixtures/${i===0?'uncertainty-covariance':'povm-instrument'}-reference.json`,'utf8')).cases})) as {kind:typeof kinds[number];cases:{label:string;input:UncertaintyInput['parameters']}[]}[];
const inputs=families.map(f=>snapshot({kind:f.kind,parameters:f.cases[0].input} as UncertaintyInput));
for(const family of families)for(const c of family.cases)test(`Uncertainty transport: ${family.kind}: ${c.label}`,()=>{const input=snapshot({kind:family.kind,parameters:c.input} as UncertaintyInput),r=evaluate({version:1,requestId:7,input});assert(r.ok);assert(success(structuredClone(r),input,7),c.label);assert(!success(r,input,8));});
class Port implements UncertaintyWorkerPort{
 onmessage:((e:MessageEvent)=>void)|null=null;onerror:((e:ErrorEvent)=>void)|null=null;onmessageerror:((e:MessageEvent)=>void)|null=null;sent:unknown[]=[];terminated=0;
 postMessage(x:unknown){this.sent.push(x);}terminate(){this.terminated++;}emit(data:unknown){this.onmessage?.({data} as MessageEvent);}
}
test('caller mutations cannot alter the active immutable preparation or its request key',()=>{
 const value=structuredClone(inputs[0]),before=key(value),port=new Port(),c=new Controller(()=>port,()=>{});c.run(value);if(value.kind===kinds[0])value.parameters.component0Real0=17;assert.equal(c.getSnapshot().inputKey,before);assert(Object.isFrozen(c.getSnapshot().input!.parameters));port.emit(evaluate(port.sent[0]));assert.equal(c.getSnapshot().status,'ready');assert.equal(c.getSnapshot().inputKey,before);assert.equal(port.terminated,1);c.dispose();
});
test('owned failure frames map to explicit errors and malformed current frames terminate the port',()=>{
 for(const[frame,error]of [[{version:1,requestId:1,ok:false,error:'invalid-input'},'invalid-input'],[{version:1,requestId:1,ok:false,error:'calculation-failed'},'worker-failed'],[{version:1,requestId:1,ok:false,error:'unknown'},'invalid-response'],[{version:1,requestId:1,ok:false,error:'invalid-input',extra:true},'invalid-response'],[{version:2,requestId:1,ok:true},'invalid-response'],[null,'invalid-response']]as const){const p=new Port(),c=new Controller(()=>p,()=>{});c.run(inputs[0]);p.emit(frame);const state=c.getSnapshot();assert.equal(state.status,'error');assert(state.status==='error');assert.equal(state.error,error);assert.equal(p.terminated,1);assert.equal(p.onmessage,null);assert.equal(p.onerror,null);assert.equal(p.onmessageerror,null);c.dispose();}
});
test('replacement, stale callbacks, cancellation and final completion release exactly one worker',()=>{
 const ports:Port[]=[],states:string[]=[];const c=new Controller(()=>{const p=new Port();ports.push(p);return p;},s=>states.push(s.status));const id=c.run(inputs[0])!,stale=ports[0].onmessage!;c.run(inputs[1]);assert.equal(ports[0].terminated,1);assert.equal(ports[0].onmessage,null);stale({data:evaluate({version:1,requestId:id,input:inputs[0]})}as MessageEvent);assert.equal(c.getSnapshot().status,'pending');ports[1].emit({version:1,requestId:id,phase:'started',kind:inputs[0].kind});assert.equal(c.getSnapshot().status,'pending');ports[1].emit(evaluate(ports[1].sent[0]));assert.equal(c.getSnapshot().status,'ready');assert.equal(ports[1].terminated,1);c.cancel();assert.equal(c.getSnapshot().status,'idle');c.dispose();assert.equal(c.run(inputs[0]),null);assert.deepEqual(states,['pending','pending','ready','idle']);
});
test('unavailable, clone failure, malformed progress, errors and retry never calculate synchronously',()=>{
 const unavailable=new Controller(()=>{throw Error('unavailable');},()=>{});unavailable.run(inputs[0]);assert.equal(unavailable.getSnapshot().status,'error');unavailable.dispose();
 for(const action of ['clone','malformed','progress','error','messageerror','cancel','dispose','invalid']){
  const port=new Port();if(action==='clone')port.postMessage=()=>{throw Error('clone');};const c=new Controller(()=>port,()=>{});c.run(inputs[0]);
  if(action==='malformed')port.emit({version:1,requestId:1,ok:true});if(action==='progress')port.emit({version:1,requestId:1,phase:'started',kind:kinds[1]});if(action==='error')port.onerror?.({preventDefault(){}}as ErrorEvent);if(action==='messageerror')port.onmessageerror?.({}as MessageEvent);if(action==='cancel')c.cancel();if(action==='dispose')c.dispose();if(action==='invalid')c.run({kind:'invalid'});assert.equal(port.terminated,1);assert.equal(port.onmessage,null);if(!['cancel','dispose'].includes(action))assert.equal(c.getSnapshot().status,'error');c.dispose();assert.equal(port.terminated,1);
 }
 const ports:Port[]=[],c=new Controller(()=>{const p=new Port();ports.push(p);return p;},()=>{});c.run(inputs[0]);ports[0].onerror?.({preventDefault(){}}as ErrorEvent);c.run(inputs[1]);ports[1].emit(evaluate(ports[1].sent[0]));assert.equal(c.getSnapshot().status,'ready');assert(ports.every(p=>p.terminated===1));c.dispose();
});
test('reentrant pending subscribers and worker factories cannot strand a worker',()=>{
 let c:Controller,created=0;const port=new Port();c=new Controller(()=>{created++;return port;},s=>{if(s.status==='pending')c.cancel();});c.run(inputs[0]);assert.equal(created,0);c.dispose();
 c=new Controller(()=>{c.cancel();return port;},()=>{});c.run(inputs[0]);assert.equal(port.terminated,1);assert.equal(c.getSnapshot().status,'idle');c.dispose();
});
test('reentrant ready notification can launch the next request without reviving a completed port',()=>{
 const ports:Port[]=[];let c:Controller,replaced=false;c=new Controller(()=>{const p=new Port();ports.push(p);return p;},s=>{if(s.status==='ready'&&!replaced){replaced=true;c.run(inputs[1]);}});c.run(inputs[0]);const late=ports[0].onmessage!;ports[0].emit(evaluate(ports[0].sent[0]));assert.equal(ports[0].terminated,1);assert.equal(c.getSnapshot().status,'pending');late({data:evaluate(ports[0].sent[0])}as MessageEvent);assert.equal(c.getSnapshot().status,'pending');ports[1].emit(evaluate(ports[1].sent[0]));assert.equal(c.getSnapshot().status,'ready');assert(ports.every(p=>p.terminated===1));c.dispose();
});
test('envelopes and mixed families fail before calculation',()=>{
 for(const value of [null,{}, {version:2,requestId:1,input:inputs[0]}, {version:1,requestId:0,input:inputs[0]}, {version:1,requestId:1,input:inputs[0],extra:1}])assert.equal(evaluate(value).ok,false);
 for(const input of [{kind:kinds[0],parameters:inputs[1].parameters},{kind:'unknown',parameters:inputs[0].parameters}])assert.deepEqual(evaluate({version:1,requestId:1,input}),{version:1,requestId:1,ok:false,error:'invalid-input'});
});

test('reader dependency graph omits calculation engines; actual worker entry emits started and validated results',async()=>{
 const reader=await build({entryPoints:['src/components/learning/use-uncertainty-calculation.ts'],bundle:true,write:false,metafile:true,platform:'browser',format:'esm',target:'es2020'}),bundled=Object.keys(reader.metafile!.inputs);
 assert(!bundled.some(p=>/learning-(uncertainty-covariance|uncertainty-covariance-math|povm-instrument|povm-algebraic|quantum-evolution-interval|observable-matrix|quantum-hilbert-core)\.ts|entropy-production-math|interval-transcendentals/.test(p)),bundled.join('\n'));
 const code=await build({entryPoints:['src/lib/learning-uncertainty.worker.ts'],bundle:true,write:false,platform:'browser',format:'iife',target:'es2020'});
 const worker=new Worker(`const{parentPort}=require('node:worker_threads');globalThis.self={postMessage:x=>parentPort.postMessage(x)};${code.outputFiles[0].text}\nparentPort.on('message',data=>self.onmessage({data}));`,{eval:true});
 try{for(let i=0;i<inputs.length;i++){
  const messages:unknown[]=[];const done=new Promise<void>((resolve,reject)=>{const timer=setTimeout(()=>reject(Error('worker timeout')),15000);const error=(e:Error)=>{clearTimeout(timer);reject(e);};const receive=(message:unknown)=>{messages.push(message);if(messages.length===2){clearTimeout(timer);worker.off('message',receive);worker.off('error',error);resolve();}};worker.on('message',receive);worker.once('error',error);});worker.postMessage({version:1,requestId:i+1,input:inputs[i]});await done;assert.deepEqual(messages[0],{version:1,requestId:i+1,phase:'started',kind:inputs[i].kind});assert(success(messages[1],inputs[i],i+1));
 }}finally{await worker.terminate();}
});
test('React server rendering starts no Worker and exposes no response before mounting',async()=>{
 const React=await import('react'),{renderToStaticMarkup}=await import('react-dom/server'),{useUncertaintyCalculation}=await import('../src/components/learning/use-uncertainty-calculation');
 const previous=Object.getOwnPropertyDescriptor(globalThis,'Worker');let reads=0;Object.defineProperty(globalThis,'Worker',{configurable:true,get(){reads++;throw Error('Worker unavailable during SSR');}});
 try{for(const input of [...inputs,null]){const Probe=()=>{const state=useUncertaintyCalculation(input);assert.equal(state.response,null);assert.equal(state.requestId,null);return React.createElement('output',null,state.status);};assert.equal(renderToStaticMarkup(React.createElement(Probe)),'<output>idle</output>');}assert.equal(reads,0);}finally{if(previous)Object.defineProperty(globalThis,'Worker',previous);else Reflect.deleteProperty(globalThis,'Worker');}
});


test('uncertainty input identity, lexical integers and storage reject rounded or foreign preparations',()=>{
 for(const input of inputs){
  assert.equal(key(input),key({...input,parameters:Object.fromEntries(Object.entries(input.parameters).reverse())}as UncertaintyInput));assert.deepEqual(draft(JSON.stringify(input)),input);assert.deepEqual(stored(JSON.stringify({version:1,value:input})),input);
  assert.equal(stored(JSON.stringify({version:2,value:input})),null);assert.equal(stored(JSON.stringify({version:1,value:input,extra:1})),null);assert(Object.isFrozen(snapshot(input)));assert(Object.isFrozen(snapshot(input).parameters));
  const field=input.kind===kinds[0]?'weight0':'blochDenominator';for(const token of ['1e-9999','1.00000000000000001','9007199254740993','1e9999']){const raw=JSON.stringify(input).replace(new RegExp('"'+field+'":-?\\d+'),'"'+field+'":'+token);assert.equal(draft(raw),null);assert.equal(stored('{"version":1,"value":'+raw+'}'),null);}
  assert(!valid({...input,extra:1}));assert(!valid({...input,[Symbol('foreign')]:1}));assert(!valid({...input,parameters:{...input.parameters,[Symbol('foreign')]:1}}));assert(!valid(Object.assign(Object.create(input),{})));
 }
 assert.equal(draft(' '.repeat(32769)),null);assert.equal(draft('not-json'),null);assert.equal(stored('{}'),null);
});
const attributedInputs=[snapshot({kind:kinds[0],parameters:families[0].cases.find(c=>c.label==='pure XY covariance saturation')!.input}as UncertaintyInput),snapshot({kind:kinds[1],parameters:families[1].cases.find(c=>c.label==='irrational retention')!.input}as UncertaintyInput)];
const responses=attributedInputs.map(input=>{const r=evaluate({version:1,requestId:1,input});assert(r.ok);return r;});
function rejectMutation(family:number,change:(r:any)=>void){const x=structuredClone(responses[family]);change(x);assert.notDeepEqual(x,responses[family],'Mutation must change the response');assert(!success(x,attributedInputs[family],1));}
test('every root field and representative nested schema requires exact owned fields',()=>{
 for(const[i,r]of responses.entries()){
  for(const k of Object.keys(r))rejectMutation(i,x=>{delete x[k];});for(const k of Object.keys(r.result))rejectMutation(i,x=>{delete x.result[k];});
  rejectMutation(i,x=>{x.extra=1;});rejectMutation(i,x=>{x.result.extra=1;});rejectMutation(i,x=>{x.result[Symbol('foreign')]=1;});rejectMutation(i,x=>{x.inputKey+=' ';});rejectMutation(i,x=>{x.result.precisionBits=53;});rejectMutation(i,x=>{x.kind=kinds[1-i];});
 }
 const groups=[['moments','bounds','complexProjection','linearCombination','quantumSpectrum','varianceDecomposition','classification'],['preparation','rawMeter','classification']];for(const[i,names]of groups.entries())for(const name of names)for(const field of Object.keys((responses[i].result as any)[name]))rejectMutation(i,x=>{delete x.result[name][field];});
 for(const name of ['channel','disturbance','rawOzawa','classification','exactResiduals','directedResiduals'])for(const field of Object.keys((responses[1].result as any).instruments[0][name]))rejectMutation(1,x=>{delete x.result.instruments[0][name][field];});
 rejectMutation(0,x=>{delete x.result.weightedFluctuationColumns.a[0][0];});rejectMutation(0,x=>{x.result.quantumSpectrum.parts.extra=1;});rejectMutation(1,x=>{delete x.result.instruments[0].outcomes[0];});rejectMutation(1,x=>{x.result.instruments[0].channel.choi[0][Symbol('foreign')]=1;});
});
test('observation certificates reject malformed fractions, forged drawing values and wide bounds',()=>{
 for(let i=0;i<2;i++)for(const change of [(q:any)=>{q.value=123;},(q:any)=>{q.value=NaN;},(q:any)=>{q.bounds.upper.denominator.coefficient='0';},(q:any)=>{q.bounds.lower.numerator.coefficient='1'.repeat(4097);},(q:any)=>{q.bounds.upper.numerator.exponent=1;},(q:any)=>{q.bounds.extra=1;},(q:any)=>{q.bounds.upper.numerator[Symbol('foreign')]=1;},(q:any)=>{q.logMagnitude=9999;},(q:any)=>{q.extra=1;}])rejectMutation(i,x=>change(i===0?x.result.moments.varianceA:x.result.strength));
 const broad=observe({lower:f(BigInt(-1)),upper:f(BigInt(1))});rejectMutation(0,x=>{x.result.directedResiduals.weightedGram[0][0].real=broad;});rejectMutation(1,x=>{x.result.instruments[0].directedResiduals.rawRmsError.real=broad;});
});
test('covariance attribution rejects changed state, means, commutators, slacks and combination accounts',()=>{
 const q=exactObservation(f(BigInt(1),BigInt(3)));
 for(const field of ['varianceA','varianceB','symmetricCovariance','commutatorHalfOverI'])rejectMutation(0,x=>{x.result.moments[field]=q;});
 for(const field of ['varianceProduct','robertsonSquaredBound','schrodingerSquaredBound','robertsonSlack','schrodingerSlack'])rejectMutation(0,x=>{x.result.bounds[field]=q;});
 rejectMutation(0,x=>{x.result.density[0][0].real=q;});rejectMutation(0,x=>{x.result.observables.a[0][0].real=q;});rejectMutation(0,x=>{x.result.observables.commutator[0][0].real=q;});rejectMutation(0,x=>{x.result.moments.centeredJordanMean.real=q;});rejectMutation(0,x=>{x.result.linearCombination.variance=q;});rejectMutation(0,x=>{x.result.complexProjection.minimumResidualVariance=q;});rejectMutation(0,x=>{x.result.quantumSpectrum.determinant=q;});rejectMutation(0,x=>{x.result.classification.operatorsCommute=true;});rejectMutation(0,x=>{x.result.varianceDecomposition.withinA=q;});rejectMutation(0,x=>{x.result.quantumSpectrum.parts[0].zeroEigenvalueExact=false;});
});
test('POVM attribution rejects altered effects, feedback, algebraic signs and channel descriptions',()=>{
 const q=exactObservation(f(BigInt(1),BigInt(3)));
 rejectMutation(1,x=>{x.result.effects[0][0][0].real=q;});rejectMutation(1,x=>{x.result.effectProbabilities[0]=q;});rejectMutation(1,x=>{x.result.measurementAxis[0]=q;});rejectMutation(1,x=>{x.result.feedback[0].rotation[0][0]=q;});rejectMutation(1,x=>{x.result.coherenceRetention.exactSign=-1;});rejectMutation(1,x=>{x.result.coherenceRetention.rootPart=q;});
 rejectMutation(1,x=>{x.result.instruments.reverse();});rejectMutation(1,x=>{x.result.instruments[0].outcomes.reverse();});rejectMutation(1,x=>{x.result.instruments[0].nonselectiveBloch[1]=x.result.instruments[0].nonselectiveBloch[0];});rejectMutation(1,x=>{x.result.instruments[0].channel.choi=x.result.instruments[2].channel.choi;});rejectMutation(1,x=>{x.result.instruments[0].channel.choiRank=1;x.result.instruments[0].channel.minimalKrausCount=1;});rejectMutation(1,x=>{x.result.instruments[0].channel.affineMatrix[0][0]=x.result.instruments[0].channel.affineMatrix[2][2];});rejectMutation(1,x=>{x.result.instruments[0].classification.channelIdentity=true;});
});
test('POVM meter and RMS attribution preserve calibration, corrected relation and ordered laws',()=>{
 rejectMutation(1,x=>{x.result.rawMeter=x.result.calibratedMeter;});rejectMutation(1,x=>{x.result.calibratedMeter=null;});rejectMutation(1,x=>{x.result.instruments[0].calibratedOzawa=null;});rejectMutation(1,x=>{x.result.instruments[0].rawOzawa.naiveProductBoundViolated=false;});rejectMutation(1,x=>{x.result.instruments[0].rawOzawa.correctedLeft=exactObservation(f(BigInt(0)));});rejectMutation(1,x=>{x.result.instruments[0].rawOzawa.correctedSlack=exactObservation(f(BigInt(0)));x.result.instruments[0].rawOzawa.relation='exact-equality';});rejectMutation(1,x=>{x.result.instruments[0].disturbance.rmsSquared=x.result.instruments[2].disturbance.rmsSquared;});rejectMutation(1,x=>{x.result.instruments[0].outcomes[0].followupJointProbabilities[0]=x.result.instruments[0].outcomes[0].conditionalPurity;});
});
test('structural zeros preserve null projections, merged eigenspaces, impossible outcomes and no calibration',()=>{
 for(const label of ['zero variance A nonzero B','dimension 1 pure','mixed Pauli XY']){
  const input=snapshot({kind:kinds[0],parameters:families[0].cases.find(c=>c.label===label)!.input}as UncertaintyInput),response=evaluate({version:1,requestId:1,input});assert(response.ok&&response.kind===kinds[0]);assert(success(response,input,1));
  const x:any=structuredClone(response);if(label==='zero variance A nonzero B')x.result.complexProjection.coefficientBOnA=x.result.moments.meanA;else{x.result.quantumSpectrum.parts.push(structuredClone(x.result.quantumSpectrum.parts[0]));}assert(!success(x,input,1));
 }
 for(const label of ['zero outcome positive Z','positive Y strength 0/1']){
  const input=snapshot({kind:kinds[1],parameters:families[1].cases.find(c=>c.label===label)!.input}as UncertaintyInput),response=evaluate({version:1,requestId:1,input});assert(response.ok&&response.kind===kinds[1]);assert(success(response,input,1));
  const x:any=structuredClone(response);if(label==='zero outcome positive Z'){const o=x.result.instruments[0].outcomes.find((o:any)=>o.probability.sign===0);o.conditionalDensity=x.result.instruments[0].nonselectiveDensity;}else{x.result.calibratedMeter=x.result.rawMeter;}assert(!success(x,input,1));
 }
});
