import {exact as exactObservation} from '../src/lib/learning-quantum-hilbert-core';
import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';import {build} from 'esbuild';import {Worker} from 'node:worker_threads';
import {evaluateQuantumVariationalRequest as evaluate} from '../src/lib/learning-quantum-variational-evaluator';
import {QuantumVariationalCalculationController as Controller,type QuantumVariationalWorkerPort} from '../src/lib/learning-quantum-variational-controller';
import {quantumVariationalKinds as kinds,snapshotQuantumVariationalInput as snapshot,quantumVariationalInputKey as key,isQuantumVariationalInput as valid,isQuantumVariationalSuccessForInput as success,parseQuantumVariationalInputDraft as draft,parseQuantumVariationalStoredInput as stored,type QuantumVariationalInput} from '../src/lib/learning-quantum-variational-protocol';
const families=kinds.map((kind,i)=>({kind,cases:JSON.parse(readFileSync(`scripts/fixtures/${i===0?'rayleigh':'ritz'}-reference.json`,'utf8')).cases})) as {kind:typeof kinds[number];cases:{label:string;input:QuantumVariationalInput['parameters']}[]}[];
const inputs=families.map(f=>snapshot({kind:f.kind,parameters:f.cases[0].input} as QuantumVariationalInput));
for(const family of families)for(const c of family.cases)test(`QuantumVariational transport: ${family.kind}: ${c.label}`,()=>{const input=snapshot({kind:family.kind,parameters:c.input} as QuantumVariationalInput),r=evaluate({version:1,requestId:7,input});assert(r.ok);assert(success(structuredClone(r),input,7),c.label);assert(!success(r,input,8));});
class Port implements QuantumVariationalWorkerPort{
 onmessage:((e:MessageEvent)=>void)|null=null;onerror:((e:ErrorEvent)=>void)|null=null;onmessageerror:((e:MessageEvent)=>void)|null=null;sent:unknown[]=[];terminated=0;
 postMessage(x:unknown){this.sent.push(x);}terminate(){this.terminated++;}emit(data:unknown){this.onmessage?.({data} as MessageEvent);}
}
test('caller mutations cannot alter the active immutable preparation or its request key',()=>{
 const value=structuredClone(inputs[0]),before=key(value),port=new Port(),c=new Controller(()=>port,()=>{});c.run(value);if(value.kind===kinds[0])value.parameters.stateReal0=17;assert.equal(c.getSnapshot().inputKey,before);assert(Object.isFrozen(c.getSnapshot().input!.parameters));port.emit(evaluate(port.sent[0]));assert.equal(c.getSnapshot().status,'ready');assert.equal(c.getSnapshot().inputKey,before);assert.equal(port.terminated,1);c.dispose();
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
 const reader=await build({entryPoints:['src/components/learning/use-quantum-variational-calculation.ts'],bundle:true,write:false,metafile:true,platform:'browser',format:'esm',target:'es2020'}),bundled=Object.keys(reader.metafile!.inputs);
 assert(!bundled.some(p=>/learning-(rayleigh|ritz(?:-input|-exact|-interval)?|tensor-roots|quantum-evolution-interval|observable-matrix|quantum-hilbert-core|quantum-variational-math)\.ts|entropy-production-math|interval-transcendentals/.test(p)),bundled.join('\n'));
 const code=await build({entryPoints:['src/lib/learning-quantum-variational.worker.ts'],bundle:true,write:false,platform:'browser',format:'iife',target:'es2020'});
 const worker=new Worker(`const{parentPort}=require('node:worker_threads');globalThis.self={postMessage:x=>parentPort.postMessage(x)};${code.outputFiles[0].text}\nparentPort.on('message',data=>self.onmessage({data}));`,{eval:true});
 try{for(let i=0;i<inputs.length;i++){
  const messages:unknown[]=[];const done=new Promise<void>((resolve,reject)=>{const timer=setTimeout(()=>reject(Error('worker timeout')),60000);const error=(e:Error)=>{clearTimeout(timer);reject(e);};const receive=(message:unknown)=>{messages.push(message);if(messages.length===2){clearTimeout(timer);worker.off('message',receive);worker.off('error',error);resolve();}};worker.on('message',receive);worker.once('error',error);});worker.postMessage({version:1,requestId:i+1,input:inputs[i]});await done;assert.deepEqual(messages[0],{version:1,requestId:i+1,phase:'started',kind:inputs[i].kind});assert(success(messages[1],inputs[i],i+1));
 }}finally{await worker.terminate();}
});
test('React server rendering starts no Worker and exposes no response before mounting',async()=>{
 const React=await import('react'),{renderToStaticMarkup}=await import('react-dom/server'),{useQuantumVariationalCalculation}=await import('../src/components/learning/use-quantum-variational-calculation');
 const previous=Object.getOwnPropertyDescriptor(globalThis,'Worker');let reads=0;Object.defineProperty(globalThis,'Worker',{configurable:true,get(){reads++;throw Error('Worker unavailable during SSR');}});
 try{for(const input of [...inputs,null]){const Probe=()=>{const state=useQuantumVariationalCalculation(input);assert.equal(state.response,null);assert.equal(state.requestId,null);return React.createElement('output',null,state.status);};assert.equal(renderToStaticMarkup(React.createElement(Probe)),'<output>idle</output>');}assert.equal(reads,0);}finally{if(previous)Object.defineProperty(globalThis,'Worker',previous);else Reflect.deleteProperty(globalThis,'Worker');}
});


test('quantumVariational input identity, lexical integers and storage reject rounded or foreign preparations',()=>{
 for(const input of inputs){
  assert.equal(key(input),key({...input,parameters:Object.fromEntries(Object.entries(input.parameters).reverse())}as QuantumVariationalInput));assert.deepEqual(draft(JSON.stringify(input)),input);assert.deepEqual(stored(JSON.stringify({version:1,value:input})),input);
  assert.equal(stored(JSON.stringify({version:2,value:input})),null);assert.equal(stored(JSON.stringify({version:1,value:input,extra:1})),null);assert(Object.isFrozen(snapshot(input)));assert(Object.isFrozen(snapshot(input).parameters));
  const field=input.kind===kinds[0]?'stateReal0':'hReal00';for(const token of ['1e-9999','1.00000000000000001','9007199254740993','1e9999']){const raw=JSON.stringify(input).replace(new RegExp('"'+field+'":-?\\d+'),'"'+field+'":'+token);assert.equal(draft(raw),null);assert.equal(stored('{"version":1,"value":'+raw+'}'),null);}
  assert(!valid({...input,extra:1}));assert(!valid({...input,[Symbol('foreign')]:1}));assert(!valid({...input,parameters:{...input.parameters,[Symbol('foreign')]:1}}));assert(!valid(Object.assign(Object.create(input),{})));
 }
 assert.equal(draft(' '.repeat(32769)),null);assert.equal(draft('not-json'),null);assert.equal(stored('{}'),null);
});


import{quantumVariationalRitzKeys,isQuantumVariationalRitzParameters,rayleighKeys}from'../src/lib/learning-quantum-variational-input';
import{ritzKeys,isRitzParameters}from'../src/lib/learning-ritz-input';
const attributedInputs=[snapshot({kind:kinds[0],parameters:families[0].cases.find(c=>c.label==='four distinct signed roots')!.input}as QuantumVariationalInput),snapshot({kind:kinds[1],parameters:families[1].cases.find(c=>c.label==='general complex 3x2 case0')!.input}as QuantumVariationalInput)];
const attributed=attributedInputs.map(input=>{const r=evaluate({version:1,requestId:17,input});assert(r.ok);return r;});
const pathValue=(root:any,path:(string|number)[])=>path.reduce((v,k)=>v[k],root);
function rejectsMutation(family:number,label:string,edit:(packet:any)=>void){const p=structuredClone(attributed[family]);const before=JSON.stringify(p);edit(p);assert.notEqual(JSON.stringify(p),before,label+' changes a value');assert(!success(p,attributedInputs[family],17),label);}

test('every public result root field requires owned data with no missing,extra,symbol or accessor fields',()=>{
 for(let family=0;family<2;family++){
  for(const name of Object.keys(attributed[family].result)){const p=structuredClone(attributed[family]);delete(p.result as any)[name];assert(!success(p,attributedInputs[family],17),name);}
  for(const path of [[],['result'],['result','parameters'],['result','classification'],['result','spectrum',0],['result','energy'],['result','energy','bounds'],['result','energy','bounds','lower'],['result','energy','bounds','lower','numerator'],['input'],['input','parameters']]){
   for(const key of ['foreign',Symbol('foreign')]){const p=structuredClone(attributed[family]);const target=pathValue(p,path);target[key]=0;assert(!success(p,attributedInputs[family],17),String(path));}
  }
  let calls=0;const p=structuredClone(attributed[family]);Object.defineProperty(p.result,'energy',{enumerable:true,get(){calls++;return exactObservation(BigInt(1),BigInt(1));}});assert(!success(p,attributedInputs[family],17));assert.equal(calls,0);
 }
});
test('dense arrays,reduced observation structure and numeric display metadata reject malformed packets',()=>{
 for(let family=0;family<2;family++){
  for(const edit of [(p:any)=>{delete p.result.state[0];},(p:any)=>{p.result.state.extra=1;},(p:any)=>{p.result.state[Symbol('extra')]=1;},(p:any)=>{p.result.energy.value=1234567;},(p:any)=>{p.result.energy.logMagnitude=9999;},(p:any)=>{p.result.energy.bounds.lower.denominator.coefficient='0';},(p:any)=>{p.result.energy.bounds.lower.numerator.exponent=1;},(p:any)=>{p.result.energy.bounds.lower.numerator.coefficient='1'.repeat(8193);},(p:any)=>{p.result.energy=Object.assign(Object.create(p.result.energy),{});},(p:any)=>{p.result.energy.value=Infinity;}]){
   const p=structuredClone(attributed[family]);edit(p);assert(!success(p,attributedInputs[family],17));
  }
 }
});
test('Rayleigh exact/normalized/moment/path/root and ground accounts remain tied to the request',()=>{
 for(const path of [['hamiltonian',0,0,'real'],['state',0,'imaginary'],['normalizedState',0,'real'],['direction',0,'real'],['rawNormSquared'],['energy'],['secondMoment'],['variance'],['residualNumerators',0,'real'],['normalizedResidual',0,'real'],['residualNorm'],['centeredHamiltonian',0,0,'real'],['variation','firstDerivative'],['variation','secondDerivative'],['variation','normFirstDerivative'],['path','rawState',0,'real'],['path','parameter'],['characteristicPolynomial',1],['squareFreeFactors',0,'ascending',0],['spectrum',0,'projector',0,0,'real'],['spectrum',0,'projectedRawState',0,'real'],['spectrum',0,'probability'],['spectrum',0,'lower'],['ground','energy'],['ground','overlap'],['ground','leakage'],['ground','excessEnergy'],['ground','gap'],['ground','energyOverGap'],['ground','certifiedLeakageUpperBound']]as(string|number)[][]){
  rejectsMutation(0,path.join('.'),p=>{const parent=pathValue(p.result,path.slice(0,-1));parent[path.at(-1)!]=exactObservation(BigInt(7),BigInt(1));});
 }
 rejectsMutation(0,'eigenstate flag',p=>p.result.classification.exactEigenstate=!p.result.classification.exactEigenstate);rejectsMutation(0,'root multiplicity',p=>p.result.spectrum[0].multiplicity++);rejectsMutation(0,'precision claim',p=>p.result.precisionBits=256);rejectsMutation(0,'contract interpretation',p=>p.result.approximationContract+=' altered');
});
test('Ritz metric,space,coordinate,spectral,minimum and residual fields reject foreign mathematics',()=>{
 for(const path of [['basis',0,0,'real'],['gram',0,0,'real'],['gramInverse',0,0,'real'],['gramDeterminant','real'],['projectedHamiltonian',0,0,'real'],['coefficientOperator',0,0,'real'],['spaceProjector',0,0,'real'],['coefficients',0,'real'],['coefficientEuclideanNormSquared'],['state',0,'real'],['normalizedState',0,'real'],['rawNormSquared'],['energy'],['compressedSecondMoment'],['projectedVariance'],['outsideVariance'],['coordinateStationarityResidual',0,'real'],['characteristicPolynomial',1],['spectrum',0,'physicalProjector',0,0,'real'],['spectrum',0,'coefficientProjector',0,0,'real'],['spectrum',0,'probability'],['spectrum',0,'fullResidualOperator',0,0,'real'],['spectrum',0,'fullResidualFrobeniusSquared'],['ambientSpectrum',0,'lower'],['gramSpectrum',0,'upper'],['gramConditionNumber'],['nestedSubspaces',0,'lower'],['nestedSubspaces',0,'minimumEnergy'],['basisChange','matrix',0,0,'real'],['basisChange','basis',0,0,'real'],['basisChange','coefficients',0,'real'],['basisChange','gram',0,0,'real'],['basisChange','projectedHamiltonian',0,0,'real'],['basisChange','coefficientEuclideanNormSquared'],['basisChange','state',0,'real'],['basisChange','spaceProjector',0,0,'real']]as(string|number)[][]){
  rejectsMutation(1,path.join('.'),p=>{const parent=pathValue(p.result,path.slice(0,-1));parent[path.at(-1)!]=exactObservation(BigInt(17),BigInt(1));});
 }
 rejectsMutation(1,'nested multiplicity',p=>p.result.nestedSubspaces[0].minimumMultiplicity=2);rejectsMutation(1,'ambient exact flag',p=>p.result.ambientSpectrum[0].exact=!p.result.ambientSpectrum[0].exact);rejectsMutation(1,'Gram exact flag',p=>p.result.gramSpectrum[0].exact=!p.result.gramSpectrum[0].exact);rejectsMutation(1,'interlacing index',p=>p.result.interlacing[0].upperAmbientIndex=0);rejectsMutation(1,'selected Ritz stationarity',p=>p.result.classification.selectedStateRitzStationary=!p.result.classification.selectedStateRitzStationary);
});
test('response attribution binds kind,request,input identity and exact precision contracts',()=>{
 for(let i=0;i<2;i++)for(const edit of [(p:any)=>p.requestId=18,(p:any)=>p.kind=kinds[1-i],(p:any)=>p.input=attributedInputs[1-i],(p:any)=>p.inputKey='foreign',(p:any)=>p.result.parameters.hReal00++,(p:any)=>p.result.arithmeticWorkBits=128,(p:any)=>p.result.rootIsolationBits=512]){const p=structuredClone(attributed[i]);edit(p);assert(!success(p,attributedInputs[i],17));}
});
test('light Ritz input rank and key guards agree with the numerical input contract for all references and perturbations',()=>{
 assert.deepEqual(quantumVariationalRitzKeys,ritzKeys);assert.equal(rayleighKeys.length,36);
 for(const row of families[1].cases){assert(isQuantumVariationalRitzParameters(row.input));assert(isRitzParameters(row.input));for(const changed of [{...row.input,basisReal33:1},{...row.input,scaleNumerator:0},{...row.input,columns:5}])assert.equal(isQuantumVariationalRitzParameters(changed),isRitzParameters(changed));}
 const p={...inputs[1].parameters};for(let i=0;i<4;i++){p[`basisReal${i}1`]=p[`basisReal${i}0`];p[`basisImag${i}1`]=p[`basisImag${i}0`];}assert(!isQuantumVariationalRitzParameters(p));
});
test('failure/progress frames reject accessors without evaluating getters or retaining a worker',()=>{
 const p=new Port(),c=new Controller(()=>p,()=>{});c.run(inputs[0]);let reads=0;const bad={version:1,requestId:1,phase:'started',kind:kinds[0]};Object.defineProperty(bad,'requestId',{get(){reads++;return 1;}});p.emit(bad);assert.equal(c.getSnapshot().status,'error');assert.equal(reads,0);assert.equal(p.terminated,1);c.dispose();
});
test('polynomial/root certificates reject changed auxiliary intervals,wrong minima and lost degeneracies',()=>{
 for(const [family,label]of [[0,'rational non dyadic spectrum'],[1,'Ritz stationary but not ambient eigenstate']]as const){const row=families[family].cases.find(c=>c.label===label)!;const input=snapshot({kind:kinds[family],parameters:row.input}as QuantumVariationalInput),r=evaluate({version:1,requestId:21,input});assert(r.ok);assert(success(r,input,21));const p=structuredClone(r);const index=p.result.spectrum.findIndex(v=>v.multiplicity>1);assert(index>=0);p.result.spectrum[index].multiplicity=1;assert(!success(p,input,21));}
});
