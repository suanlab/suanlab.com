import test from'node:test';import assert from'node:assert/strict';import{readFileSync}from'node:fs';import{build}from'esbuild';import{Worker}from'node:worker_threads';
import{evaluateObservableRequest as evaluate}from'../src/lib/learning-observable-evaluator';
import{ObservableCalculationController as Controller,type ObservableWorkerPort}from'../src/lib/learning-observable-controller';
import{observableKinds as kinds,snapshotObservableInput as snapshot,observableInputKey as key,isObservableInput as valid,isObservableSuccessForInput as success,parseObservableInputDraft as draft,parseObservableStoredInput as stored,type ObservableInput}from'../src/lib/learning-observable-protocol';
const refs=JSON.parse(readFileSync('scripts/fixtures/quantum-observable-reference.json','utf8'));
const families=[{kind:kinds[0],cases:refs.finiteCases},{kind:kinds[1],cases:refs.sequentialCases}] as {kind:typeof kinds[number];cases:{label:string;input:ObservableInput['parameters']}[]}[];
const inputs=families.map(f=>snapshot({kind:f.kind,parameters:f.cases[0].input}as ObservableInput));
for(const family of families)for(const c of family.cases)test(`Observable transport: ${family.kind}: ${c.label}`,()=>{const input=snapshot({kind:family.kind,parameters:c.input}as ObservableInput),r=evaluate({version:1,requestId:7,input});assert(r.ok);assert(success(structuredClone(r),input,7),c.label);assert(!success(r,input,8));});
test('canonical integer drafts, stored preparations and owned inputs reject rounded fractional evidence',()=>{for(const input of inputs){const name=input.kind===kinds[0]?'gapNumerator':'aPositiveNumerator';assert.equal(key(input),key({...input,parameters:Object.fromEntries(Object.entries(input.parameters).reverse())}as ObservableInput));assert.deepEqual(draft(JSON.stringify(input)),input);assert.deepEqual(stored(JSON.stringify({version:1,value:input})),input);assert(Object.isFrozen(snapshot(input).parameters));for(const token of ['1e-9999','1.00000000000000001','9007199254740993','1e9999']){const raw=JSON.stringify(input).replace(new RegExp('"'+name+'":\\d+'),'"'+name+'":'+token);assert.equal(draft(raw),null);assert.equal(stored('{"version":1,"value":'+raw+'}'),null);}assert(!valid({...input,extra:1}));assert(!valid({...input,parameters:{...input.parameters,[Symbol('foreign')]:1}}));assert.equal(stored(JSON.stringify({version:2,value:input})),null);}});
test('complete schemas reject misattribution, rounded values, malformed branches and false probability accounts',()=>{
 for(const input of inputs){const r=evaluate({version:1,requestId:1,input});assert(r.ok);const change=(fn:(x:any)=>void)=>{const x=structuredClone(r);fn(x);assert.equal(success(x,input,1),false);};
 change(x=>{x.inputKey+=' ';});change(x=>{x.kind='unknown';});change(x=>{x.result.extra=1;});change(x=>{x.result.precisionBits=53;});change(x=>{x.result.stateAmplitudes.pop();});change(x=>{x.result.stateAmplitudes[0].real.value=100;});change(x=>{x.result.parameters.stateReal0++;});
 if(input.kind===kinds[0]){change(x=>{x.result.outcomes.reverse();});change(x=>{x.result.outcomes[0].rank=2;});change(x=>{x.result.outcomes[0].conditionalAmplitudes=null;});change(x=>{x.result.outcomes[0].projectedNumerators=[];});change(x=>{x.result.classification.fullyDegenerate=true;});change(x=>{x.result.classification.varianceIsExactlyZero=true;});change(x=>{x.result.passiveBasis.rawNormSquared=x.result.rawNormSquared;});change(x=>{x.result.mean=x.result.variance;});change(x=>{x.result.nonselectiveDensity[0][0].imaginary=x.result.variance;});}
 else{change(x=>{x.result.aThenB.jointProbabilities.reverse();});change(x=>{x.result.aThenB.firstMarginal.reverse();});change(x=>{x.result.refinedA.labels.reverse();});change(x=>{x.result.refinedA.distinctInstrumentEvenWhenEigenvaluesCoincide=false;});change(x=>{x.result.classification.aFullyDegenerate=true;});change(x=>{x.result.classification.coarseAndRefinedNonselectiveStateEqual=true;});change(x=>{x.result.a.outcomes[0].conditionalAmplitudes=x.result.stateAmplitudes;});change(x=>{x.result.aThenB.conditionalSecondProbabilities[0]=[];});change(x=>{x.result.aThenB.secondMarginal.reverse();});change(x=>{x.result.commutatorSquaredNorm=x.result.coarseRefinementSquaredDistance;});}
 }
});
test('every root and measurement field is required and sparse or decorated arrays are rejected',()=>{
 for(const input of inputs){const r=evaluate({version:1,requestId:3,input});assert(r.ok);const change=(fn:(x:any)=>void)=>{const x=structuredClone(r);fn(x);assert(!success(x,input,3));};
  for(const k of Object.keys(r.result))change(x=>{delete x.result[k];});
  const groups=input.kind===kinds[0]?[null]:['a','b','refinedA'];
  for(const group of groups){const get=(x:any)=>group?x.result[group]:x.result;for(const k of ['matrix','outcomes','mean','secondMoment','variance','nonselectiveDensity','nonselectivePurity','residuals'])change(x=>{delete get(x)[k];});for(const k of Object.keys(get(r).outcomes[0]))change(x=>{delete get(x).outcomes[0][k];});change(x=>{delete get(x).matrix[0][0];});change(x=>{get(x).matrix[0][Symbol('foreign')]=1;});change(x=>{get(x).outcomes.extra=1;});change(x=>{get(x).residuals.projectorOrthogonality[0][0]=get(x).residuals.hermiticity;});}
 }
});
test('observation bounds and rounded display are validated before response acceptance',()=>{
 for(const input of inputs){const r=evaluate({version:1,requestId:2,input});assert(r.ok);const change=(fn:(q:any)=>void)=>{const x=structuredClone(r);fn(x.result.stateAmplitudes[0].real);assert(!success(x,input,2));};
  change(q=>{q.bounds.lower.numerator.coefficient='1'.repeat(4097);});change(q=>{q.bounds.upper.denominator.coefficient='0';});change(q=>{q.bounds.lower.numerator.exponent=1;});change(q=>{q.bounds.extra=1;});change(q=>{q.extra=true;});change(q=>{q.logMagnitude=Infinity;});change(q=>{q.logMagnitude=9999;});change(q=>{q.value=NaN;});
 }
});
test('degenerate branches and refinement metadata cannot be reassigned to another instrument',()=>{
 const input=inputs[1],r=evaluate({version:1,requestId:1,input});assert(r.ok&&r.kind===kinds[1]);
 for(const edit of [(x:any)=>{x.result.refinedA.selectedStandardAxis=0;},(x:any)=>{x.result.classification.coarsePreservesInitialState=false;},(x:any)=>{x.result.coarseMinusRefinedDensity=x.result.originalDensity;},(x:any)=>{x.result.aThenB.conditionalSecondProbabilities[0]=x.result.aThenB.secondMarginal;}]){const x=structuredClone(r);edit(x);assert(!success(x,input,1));}
 const flat=snapshot({kind:kinds[0],parameters:{...inputs[0].parameters,gapNumerator:0}}as ObservableInput),f=evaluate({version:1,requestId:1,input:flat});assert(f.ok&&f.kind===kinds[0]);assert.equal(f.result.outcomes.length,1);assert.equal(f.result.outcomes[0].rank,2);assert(success(f,flat,1));
});

class Port implements ObservableWorkerPort{
 onmessage:((e:MessageEvent)=>void)|null=null;onerror:((e:ErrorEvent)=>void)|null=null;onmessageerror:((e:MessageEvent)=>void)|null=null;sent:unknown[]=[];terminated=0;
 postMessage(x:unknown){this.sent.push(x);}terminate(){this.terminated++;}emit(data:unknown){this.onmessage?.({data} as MessageEvent);}
}
test('caller mutations cannot alter the active immutable preparation or its request key',()=>{
 const value=structuredClone(inputs[0]),before=key(value),port=new Port(),c=new Controller(()=>port,()=>{});c.run(value);value.parameters.stateReal0=17;assert.equal(c.getSnapshot().inputKey,before);assert(Object.isFrozen(c.getSnapshot().input!.parameters));port.emit(evaluate(port.sent[0]));assert.equal(c.getSnapshot().status,'ready');assert.equal(c.getSnapshot().inputKey,before);assert.equal(port.terminated,1);c.dispose();
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
 const reader=await build({entryPoints:['src/components/learning/use-observable-calculation.ts'],bundle:true,write:false,metafile:true,platform:'browser',format:'esm',target:'es2020'}),bundled=Object.keys(reader.metafile!.inputs);
 assert(!bundled.some(p=>/learning-(hermitian-spectrum|sequential-observation|observable-matrix|quantum-hilbert-core)\.ts|entropy-production-math|interval-transcendentals/.test(p)),bundled.join('\n'));
 const code=await build({entryPoints:['src/lib/learning-observable.worker.ts'],bundle:true,write:false,platform:'browser',format:'iife',target:'es2020'});
 const worker=new Worker(`const{parentPort}=require('node:worker_threads');globalThis.self={postMessage:x=>parentPort.postMessage(x)};${code.outputFiles[0].text}\nparentPort.on('message',data=>self.onmessage({data}));`,{eval:true});
 try{for(let i=0;i<inputs.length;i++){
  const messages:unknown[]=[];const done=new Promise<void>((resolve,reject)=>{const timer=setTimeout(()=>reject(Error('worker timeout')),15000);const error=(e:Error)=>{clearTimeout(timer);reject(e);};const receive=(message:unknown)=>{messages.push(message);if(messages.length===2){clearTimeout(timer);worker.off('message',receive);worker.off('error',error);resolve();}};worker.on('message',receive);worker.once('error',error);});worker.postMessage({version:1,requestId:i+1,input:inputs[i]});await done;assert.deepEqual(messages[0],{version:1,requestId:i+1,phase:'started',kind:inputs[i].kind});assert(success(messages[1],inputs[i],i+1));
 }}finally{await worker.terminate();}
});
test('React server rendering starts no Worker and exposes no response before mounting',async()=>{
 const React=await import('react'),{renderToStaticMarkup}=await import('react-dom/server'),{useObservableCalculation}=await import('../src/components/learning/use-observable-calculation');
 const previous=Object.getOwnPropertyDescriptor(globalThis,'Worker');let reads=0;Object.defineProperty(globalThis,'Worker',{configurable:true,get(){reads++;throw Error('Worker unavailable during SSR');}});
 try{for(const input of [...inputs,null]){const Probe=()=>{const state=useObservableCalculation(input);assert.equal(state.response,null);assert.equal(state.requestId,null);return React.createElement('output',null,state.status);};assert.equal(renderToStaticMarkup(React.createElement(Probe)),'<output>idle</output>');}assert.equal(reads,0);}finally{if(previous)Object.defineProperty(globalThis,'Worker',previous);else Reflect.deleteProperty(globalThis,'Worker');}
});

