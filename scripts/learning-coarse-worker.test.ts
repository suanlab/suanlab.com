import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';import {build} from 'esbuild';import {Worker} from 'node:worker_threads';
import {evaluateCoarseRequest as evaluate} from '../src/lib/learning-coarse-evaluator';
import {CoarseCalculationController as Controller,type CoarseWorkerPort} from '../src/lib/learning-coarse-controller';
import {coarseKinds as kinds,snapshotCoarseInput as snapshot,coarseInputKey as key,isCoarseInput as valid,isCoarseSuccessForInput as success,parseCoarseInputDraft as draft,parseCoarseStoredInput as stored,type CoarseInput} from '../src/lib/learning-coarse-protocol';
const families=['markov-coarse','linear-memory'].map((name,i)=>({kind:kinds[i],cases:JSON.parse(readFileSync(`scripts/fixtures/${name}-reference.json`,'utf8')).cases as{label:string;input:CoarseInput['parameters']}[]}));
const inputs=families.map(f=>snapshot({kind:f.kind,parameters:f.cases[0].input} as CoarseInput));
for(const family of families)for(const c of family.cases)test(`coarse transport: ${family.kind}: ${c.label}`,()=>{const input=snapshot({kind:family.kind,parameters:c.input} as CoarseInput),r=evaluate({version:1,requestId:7,input});assert(r.ok);assert(success(structuredClone(r),input,7),`${family.kind} ${c.label}`);assert(!success(r,input,8));});
test('canonical owned snapshots and lexical integer evidence include the discrete initial law',()=>{
 for(const input of inputs){const reversed={kind:input.kind,parameters:Object.fromEntries(Object.entries(input.parameters).reverse())} as CoarseInput;assert.equal(key(input),key(reversed));assert.deepEqual(draft(JSON.stringify(input)),input);assert.deepEqual(stored(JSON.stringify({version:1,value:input})),input);assert(Object.isFrozen(snapshot(input).parameters));
  for(const token of ['1e-9999','1.00000000000000001','9007199254740993','1e9999']){const name=input.kind===kinds[0]?'memoryDepth':'timeNumerator',raw=JSON.stringify(input).replace(new RegExp('"'+name+'":\\d+'),'"'+name+'":'+token);assert.equal(draft(raw),null);assert.equal(stored('{"version":1,"value":'+raw+'}'),null);}
  assert(!valid({...input,extra:1}));assert(!valid({...input,parameters:{...input.parameters,[Symbol('extra')]:1}}));assert.equal(stored(JSON.stringify({version:2,value:input})),null);
 }
 const p=structuredClone(inputs[0]);assert(p.kind===kinds[0]);const own=snapshot(p);assert(own.kind===kinds[0]);p.parameters.initialWeights=[0,0,1];assert.deepEqual(own.parameters.initialWeights,[1,0,0]);assert(Object.isFrozen(own.parameters.initialWeights));
 for(const token of ['1e-9999','1.00000000000000001'])assert.equal(draft(JSON.stringify(inputs[0]).replace('[1,0,0]','['+token+',0,0]')),null);
});
test('complete schemas reject incomplete, nonfinite, misattributed and reordered dynamics',()=>{
 for(const input of inputs){const base=evaluate({version:1,requestId:1,input});assert(base.ok);const mutate=(fn:(x:any)=>void)=>{const x=structuredClone(base);fn(x);assert(!success(x,input,1));};
  mutate(x=>{delete x.result.current;});mutate(x=>{x.result.extra=1;});mutate(x=>{x.inputKey+=' ';});mutate(x=>{x.kind='other';});mutate(x=>{x.result.history.reverse();});mutate(x=>{x.result.current=structuredClone(x.result.history[0]);});mutate(x=>{x.result.history[0].step=1;});mutate(x=>{x.result.history[0].extra=1;});
  if(input.kind===kinds[0]){mutate(x=>{x.result.parameters.memoryDepth++;});mutate(x=>{x.result.current.micro.pop();});mutate(x=>{x.result.history[0].micro[0].value=100;});mutate(x=>{x.result.transition[0][0].value=1;});mutate(x=>{x.result.classification.continuousTimeModel=true;});mutate(x=>{x.result.classification.strongLumpableAllInitialLaws=true;});mutate(x=>{x.result.history[0].truncatedIsProbability=false;});mutate(x=>{x.result.memoryKernels[0].lag=1;});mutate(x=>{x.result.history[0].time=0;});}
  else{mutate(x=>{x.result.parameters.timeNumerator++;});mutate(x=>{x.result.current.x.value=NaN;});mutate(x=>{x.result.history[1].x.value=100;});mutate(x=>{x.result.history[0].time.value=1;});mutate(x=>{x.result.history[1].flowMethod='exact-zero-time';});mutate(x=>{x.result.classification.physicalNoiseSpecified=true;});mutate(x=>{x.result.current.convolutionOfObservedState=null;});mutate(x=>{x.result.classification.asymptoticallyStable=false;});}
 }
});
class Port implements CoarseWorkerPort{
 onmessage:((e:MessageEvent)=>void)|null=null;onerror:((e:ErrorEvent)=>void)|null=null;onmessageerror:((e:MessageEvent)=>void)|null=null;sent:unknown[]=[];terminated=0;
 postMessage(x:unknown){this.sent.push(x);}terminate(){this.terminated++;}emit(data:unknown){this.onmessage?.({data} as MessageEvent);}
}
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
test('envelopes and mixed families fail before calculation',()=>{
 for(const value of [null,{}, {version:2,requestId:1,input:inputs[0]}, {version:1,requestId:0,input:inputs[0]}, {version:1,requestId:1,input:inputs[0],extra:1}])assert.equal(evaluate(value).ok,false);
 for(const input of [{kind:kinds[0],parameters:inputs[1].parameters},{kind:'unknown',parameters:inputs[0].parameters}])assert.deepEqual(evaluate({version:1,requestId:1,input}),{version:1,requestId:1,ok:false,error:'invalid-input'});
});
test('reader dependency graph omits calculation engines; actual worker entry emits started and validated results',async()=>{
 const reader=await build({entryPoints:['src/components/learning/use-coarse-calculation.ts'],bundle:true,write:false,metafile:true,platform:'browser',format:'esm',target:'es2020'}),bundled=Object.keys(reader.metafile!.inputs);
 assert(!bundled.some(p=>/learning-(markov-coarse|linear-memory|linear-memory-math)\.ts|entropy-production-math|interval-transcendentals/.test(p)),bundled.join('\n'));
 const code=await build({entryPoints:['src/lib/learning-coarse.worker.ts'],bundle:true,write:false,platform:'browser',format:'iife',target:'es2020'});
 const worker=new Worker(`const{parentPort}=require('node:worker_threads');globalThis.self={postMessage:x=>parentPort.postMessage(x)};${code.outputFiles[0].text}\nparentPort.on('message',data=>self.onmessage({data}));`,{eval:true});
 try{for(let i=0;i<inputs.length;i++){
  const messages:unknown[]=[];const done=new Promise<void>((resolve,reject)=>{const timer=setTimeout(()=>reject(Error('worker timeout')),15000);const error=(e:Error)=>{clearTimeout(timer);reject(e);};const receive=(message:unknown)=>{messages.push(message);if(messages.length===2){clearTimeout(timer);worker.off('message',receive);worker.off('error',error);resolve();}};worker.on('message',receive);worker.once('error',error);});worker.postMessage({version:1,requestId:i+1,input:inputs[i]});await done;assert.deepEqual(messages[0],{version:1,requestId:i+1,phase:'started',kind:inputs[i].kind});assert(success(messages[1],inputs[i],i+1));
 }}finally{await worker.terminate();}
});

test('negative pseudo-laws and extreme initial layers retain their defining metadata',()=>{
 const scenarios=[[0,'negative truncated pseudo-law'],[1,'stiff hidden layer'],[1,'near marginal rounded stability'],[1,'marginal exact fractions']] as const;
 for(const [i,label]of scenarios){const p=families[i].cases.find(c=>c.label===label)!.input,input={kind:kinds[i],parameters:p}as CoarseInput,r=evaluate({version:1,requestId:1,input});assert(r.ok);assert(success(r,input,1));const x:any=structuredClone(r);
  if(i===0)x.result.history[2].truncatedIsProbability=true;else if(label==='stiff hidden layer')x.result.current.initialHiddenMagnitudeUnresolvedAtWorkingPrecision=false;else x.result.classification.marginalZeroMode=!x.result.classification.marginalZeroMode;
  assert(!success(x,input,1));
 }
});

test('non-dyadic initial data are enclosed, while displaced initial intervals are rejected',()=>{
 for(const label of ['initially quasisteady','rational coefficients','near marginal rounded stability']){
  const input={kind:kinds[1],parameters:families[1].cases.find(c=>c.label===label)!.input} as CoarseInput;
  const response=evaluate({version:1,requestId:1,input});assert(response.ok&&response.kind===kinds[1]);
  assert(success(response,input,1));
  assert.equal(response.result.history[0].y.resolution,'interval-resolved');
  const displaced=structuredClone(response);
  displaced.result.history[0].y=structuredClone(displaced.result.driftMatrix[0][0]);
  assert(!success(displaced,input,1));
 }
});
