import { mkdirSync as ensureTestArtifactDirectory } from 'node:fs';
import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { build } from 'esbuild';
import { runInNewContext } from 'node:vm';
import { ManyBodyCalculationController as Controller, type ManyBodyCalculationState as State, type ManyBodyWorkerPort } from '../src/lib/learning-many-body-controller';
import { snapshotManyBodyWireInTasks as copy, ManyBodyCopyCancelled } from '../src/lib/learning-many-body-owned-wire-async';
import { snapshotManyBodyWire as syncCopy } from '../src/lib/learning-many-body-owned-wire';
import { evaluateManyBodyValidationRequest as validate } from '../src/lib/learning-many-body-validation-evaluator';
import { evaluateManyBodyRequest as calculate } from '../src/lib/learning-many-body-evaluator';
import { manyBodyKinds, snapshotManyBodyInput, type ManyBodyInput } from '../src/lib/learning-many-body-input';
import { createElement } from 'react';
import { renderToString } from 'react-dom/server';
import { useManyBodyCalculation } from '../src/components/learning/use-many-body-calculation';
ensureTestArtifactDirectory('.runtime/learning/many-body-controller', { recursive: true });

const runtime='.runtime/learning/many-body-controller';
const refs=manyBodyKinds.map(kind=>({kind,...JSON.parse(readFileSync(`scripts/fixtures/${kind}-reference.json`,'utf8')).cases[0]}));
const input=(i=0)=>snapshotManyBodyInput({kind:refs[i].kind,parameters:refs[i].input});
class Port implements ManyBodyWorkerPort {
 onmessage:ManyBodyWorkerPort['onmessage']=null;onerror:ManyBodyWorkerPort['onerror']=null;onmessageerror:ManyBodyWorkerPort['onmessageerror']=null;
 sent:any[]=[];terminated=0;throwPost=false;
 postMessage(value:unknown){if(this.throwPost)throw Error('clone failure');this.sent.push(structuredClone(value));}
 terminate(){this.terminated++;}
 emit(data:unknown){this.onmessage?.call(this as unknown as Worker,{data} as MessageEvent);}
}
function harness(options:{pause?:()=>Promise<void>;nodesPerTask?:number}={}){
 const calculations:Port[]=[],validations:Port[]=[],states:State[]=[];
 const factories={calculate:()=>{const p=new Port();calculations.push(p);return p;},validate:()=>{const p=new Port();validations.push(p);return p;}};
 const controller=new Controller(factories,s=>states.push(s),{pause:async()=>{},...options});
 return{controller,calculations,validations,states,factories};
}
async function until(check:()=>boolean){const start=performance.now();while(!check()){assert(performance.now()-start<10000,'async controller deadline');await new Promise(r=>setTimeout(r,0));}}
function deliver(p:Port){const packet=calculate(p.sent[0],m=>p.emit(m));p.emit(packet);return packet;}
async function finish(h:ReturnType<typeof harness>){await until(()=>h.validations.length>0);const p=h.validations.at(-1)!;p.emit(validate(p.sent[0]));await until(()=>h.controller.getSnapshot().status!=='pending');}
let validFamilies=0,rejectedFamilies=0;
for(const[i,ref]of refs.entries()){
 test(`controller independently validates ${ref.kind} in a separate worker port`,async()=>{
  const h=harness();assert.equal(h.controller.run(input(i)),1);const source=h.calculations[0];deliver(source);assert.equal(source.terminated,1);assert.equal(h.controller.getSnapshot().status,'pending');await finish(h);
  const s=h.controller.getSnapshot();assert(s.status==='ready');assert.equal(s.response.kind,ref.kind);assert(Object.isFrozen(s.response)&&Object.isFrozen(s.response.result)&&Object.isFrozen(s.input.parameters));assert.equal(h.validations[0].terminated,1);assert.deepEqual(h.states.map(s=>s.status==='pending'?s.phase:s.status),['starting','calculating','validating','ready']);validFamilies++;h.controller.dispose();
 });
 test(`controller rejects ${ref.kind} calculation data through the independent mathematical worker`,async()=>{
  const h=harness();h.controller.run(input(i));const p=h.calculations[0],packet:any=calculate(p.sent[0]);assert(packet.ok);packet.result.extra=0;p.emit(packet);await finish(h);const s=h.controller.getSnapshot();assert(s.status==='error'&&s.error==='invalid-response');assert(!h.states.some(s=>s.status==='ready'));assert.equal(h.validations[0].terminated,1);rejectedFamilies++;
 });
}

test('new input terminates the old calculation and captured callbacks cannot publish',()=>{
 const h=harness();h.controller.run(input());const old=h.calculations[0],message=old.onmessage!,error=old.onerror!,clone=old.onmessageerror!;h.controller.run(input(1));const before=h.controller.getSnapshot();message.call(old as unknown as Worker,{data:{bad:1}} as MessageEvent);error.call(old as unknown as Worker,{preventDefault(){throw Error('old callback executed');}} as unknown as ErrorEvent);clone.call(old as unknown as Worker,{} as MessageEvent);assert.equal(h.controller.getSnapshot(),before);assert.equal(old.terminated,1);h.controller.dispose();assert.equal(h.calculations[1].terminated,1);
});

test('cancel during raw copy prevents validation-worker creation and ready publication',async()=>{
 let resume:()=>void=()=>{};const h=harness({nodesPerTask:1,pause:()=>new Promise<void>(r=>{resume=r;})});h.controller.run(input());deliver(h.calculations[0]);h.controller.cancel();resume();await new Promise(r=>setTimeout(r,0));assert.equal(h.validations.length,0);assert.equal(h.controller.getSnapshot().status,'idle');assert(!h.states.some(s=>s.status==='ready'));
});

test('cancel during independent validation terminates it and ignores late receipt/errors',async()=>{
 const h=harness();h.controller.run(input());deliver(h.calculations[0]);await until(()=>h.validations.length>0);const p=h.validations[0],message=p.onmessage!,error=p.onerror!;const receipt=validate(p.sent[0]);h.controller.cancel();message.call(p as unknown as Worker,{data:receipt} as MessageEvent);error.call(p as unknown as Worker,{preventDefault(){throw Error('released validation callback');}} as unknown as ErrorEvent);assert.equal(p.terminated,1);assert.equal(h.controller.getSnapshot().status,'idle');
});

test('cancel during validated-result copy prevents a late ready state',async()=>{
 let admission=false,resume:()=>void=()=>{};const h=harness({nodesPerTask:1,pause:()=>admission?new Promise<void>(r=>{resume=r;}):Promise.resolve()});h.controller.run(input());deliver(h.calculations[0]);await until(()=>h.validations.length>0);const p=h.validations[0],receipt=validate(p.sent[0]);admission=true;p.emit(receipt);h.controller.cancel();resume();await new Promise(r=>setTimeout(r,0));assert.equal(h.controller.getSnapshot().status,'idle');assert.equal(p.terminated,1);assert(!h.states.some(s=>s.status==='ready'));
});

test('new run during verification cannot be overwritten by a stale success',async()=>{
 const h=harness();h.controller.run(input());deliver(h.calculations[0]);await until(()=>h.validations.length>0);const p=h.validations[0],late=p.onmessage!,receipt=validate(p.sent[0]);h.controller.run(input(1));late.call(p as unknown as Worker,{data:receipt} as MessageEvent);assert.equal(h.controller.getSnapshot().requestId,2);assert.equal(h.controller.getSnapshot().status,'pending');assert.equal(p.terminated,1);h.controller.dispose();
});

test('stale positive IDs are ignored but malformed current frames fail',()=>{
 for(const frame of[{version:2,requestId:1},{version:1,requestId:0},null,{version:1,requestId:1,phase:'started',kind:'wrong'}]){const h=harness();h.controller.run(input());h.calculations[0].emit({version:1,requestId:99,ok:true});assert.equal(h.controller.getSnapshot().status,'pending');h.calculations[0].emit(frame);assert.equal(h.controller.getSnapshot().status,'error');assert.equal(h.calculations[0].terminated,1);}
});

test('validation receipts must match request, kind, input key and success envelope',async()=>{
 for(const edit of[(v:any)=>{v.kind='wrong';},(v:any)=>{v.inputKey+=' ';},(v:any)=>{v.response.inputKey+=' ';},(v:any)=>{v.phase='started';},(v:any)=>{v.extra=0;},(v:any)=>{v.response.result=null;}]){
  const h=harness();h.controller.run(input());deliver(h.calculations[0]);await until(()=>h.validations.length>0);const p=h.validations[0],receipt=structuredClone(validate(p.sent[0]));edit(receipt);p.emit(receipt);await until(()=>h.controller.getSnapshot().status!=='pending');assert.equal(h.controller.getSnapshot().status,'error');assert.equal(p.terminated,1);
 }
});

test('invalid input does not execute a getter or start a worker',()=>{
 const h=harness();let calls=0;const p:any={kind:'spin-chain'};Object.defineProperty(p,'parameters',{get(){calls++;return{};}});h.controller.run(p);assert.equal(calls,0);assert.equal(h.calculations.length,0);assert.equal(h.controller.getSnapshot().status,'error');
});

test('factories, postMessage failures, worker errors and clone errors release current ports',async()=>{
 const h=harness();h.factories.calculate=()=>{throw Error('unavailable');};h.controller.run(input());assert.equal(h.controller.getSnapshot().status,'error');
 for(const mode of['post','error','clone'] as const){const h=harness();h.factories.calculate=()=>{const p=new Port();p.throwPost=mode==='post';h.calculations.push(p);return p;};h.controller.run(input());const p=h.calculations[0];if(mode==='error')p.onerror?.call(p as unknown as Worker,{preventDefault(){}} as unknown as ErrorEvent);if(mode==='clone')p.onmessageerror?.call(p as unknown as Worker,{} as MessageEvent);assert.equal(h.controller.getSnapshot().status,'error');assert.equal(p.terminated,1);}
 const v=harness();v.factories.validate=()=>{throw Error('validation unavailable');};v.controller.run(input());deliver(v.calculations[0]);await until(()=>v.controller.getSnapshot().status==='error');assert.equal(v.calculations[0].terminated,1);
});

test('reentrant notifications and factories cannot leak or publish an obsolete request',()=>{
 let c:Controller;let created=0;c=new Controller({calculate:()=>{created++;return new Port();},validate:()=>new Port()},s=>{if(s.status==='pending')c.cancel();});c.run(input());assert.equal(created,0);assert.equal(c.getSnapshot().status,'idle');
 const ports:Port[]=[];c=new Controller({calculate:()=>{const p=new Port();ports.push(p);if(ports.length===1)c.run(input(1));return p;},validate:()=>new Port()},()=>{});c.run(input());assert.equal(ports[0].terminated,1);assert.equal(ports[1].terminated,0);assert.equal(c.getSnapshot().requestId,2);c.dispose();assert.equal(ports[1].terminated,1);assert.equal(c.run(input()),null);
});

test('a released calculation error cannot interrupt a live validation',async()=>{
 const h=harness();h.controller.run(input());const p=h.calculations[0],error=p.onerror!,clone=p.onmessageerror!;deliver(p);await until(()=>h.validations.length>0);error.call(p as unknown as Worker,{preventDefault(){throw Error('stale');}} as unknown as ErrorEvent);clone.call(p as unknown as Worker,{} as MessageEvent);await finish(h);assert.equal(h.controller.getSnapshot().status,'ready');
});

test('task copies match synchronous owned semantics and preserve aliases without caller getters',async()=>{
 const alias={x:[1,2,'ok',null]},value={a:alias,b:alias};const actual:any=await copy(value,{pause:async()=>{},nodesPerTask:2});const expected:any=syncCopy(value);assert.deepEqual(actual,expected);assert.equal(actual.a,actual.b);assert(Object.isFrozen(actual.a.x));let calls=0;assert.deepEqual(await copy(new Proxy(value,{get(){calls++;throw Error('get');}}),{pause:async()=>{}}),actual);assert.equal(calls,0);
});

test('task copies reject accessors, prototypes, sparse/extra arrays, cycles and hostile reflection',async()=>{
 let calls=0;const accessor={};Object.defineProperty(accessor,'x',{get(){calls++;return 1;}});const cycle:any={};cycle.self=cycle;const symbol:any={};symbol[Symbol('x')]=0;const extra:any=[];extra.x=0;
 for(const v of[accessor,Object.create(null),Object.create({x:1}),[,,],extra,cycle,symbol,new Proxy({},{ownKeys(){throw Error('reflection');}}),NaN,Infinity,undefined,()=>0,'x'.repeat(8193)])await assert.rejects(copy(v,{pause:async()=>{}}));assert.equal(calls,0);
 const own:any={};Object.defineProperty(own,'__proto__',{value:{safe:true},enumerable:true});const saved:any=await copy(own,{pause:async()=>{}});assert(Object.hasOwn(saved,'__proto__'));assert.equal(Object.getPrototypeOf(saved),Object.prototype);
});

test('copy node/depth/array budgets and cancellation are enforced across yielded tasks',async()=>{
 for(const nodesPerTask of[0,2049,NaN])await assert.rejects(copy({}, {nodesPerTask}));let deep:any={};for(let i=0;i<65;i++)deep={child:deep};await assert.rejects(copy(deep,{pause:async()=>{}}));await assert.rejects(copy(Array(65537).fill(0),{pause:async()=>{}}));await assert.rejects(copy(Array.from({length:8},()=>Array(65000).fill(0)),{pause:async()=>{}}));
 let active=true,pauses=0;await assert.rejects(copy(Array(100).fill(0),{nodesPerTask:3,active:()=>active,pause:async()=>{pauses++;active=false;}}),ManyBodyCopyCancelled);assert.equal(pauses,1);await assert.rejects(copy({}, {active:()=>false}),ManyBodyCopyCancelled);
});

test('validation requests reject malformed ownership and response families before producing a receipt',()=>{
 for(const value of[null,{}, {version:1,requestId:0,input:input(),response:{}},{version:1,requestId:1,input:{},response:{}}])assert.equal(validate(value).phase,'rejected');
 const p:any={version:1,requestId:1,input:input(),response:{}};Object.defineProperty(p,'response',{get(){throw Error('getter must not run');}});assert.equal(validate(p).phase,'rejected');
});

test('main controller and factories exclude all numerical validators and import safely during SSR',async()=>{
 const entries=['src/lib/learning-many-body-controller.ts','src/lib/learning-many-body-worker-factory.ts','src/lib/learning-many-body-validation-worker-factory.ts'];
 const code=await build({entryPoints:entries,bundle:true,write:false,metafile:true,outdir:'unused',platform:'browser',format:'esm',target:'es2020'}),inputs=Object.keys(code.metafile!.inputs);
 assert(!inputs.some(p=>/learning-(spin-chain|fermion-fock|boson-fock|boson-coherent|mps|schmidt|tensor-roots|quantum-hilbert-core|ritz-interval|interval-transcendentals|many-body-(?:evaluator|validation-evaluator|.*-validation))\.ts$/.test(p)),inputs.join('\n'));
 const ssr=await build({stdin:{contents:"export {ManyBodyCalculationController} from './src/lib/learning-many-body-controller';export {createManyBodyWorker} from './src/lib/learning-many-body-worker-factory';export {createManyBodyValidationWorker} from './src/lib/learning-many-body-validation-worker-factory';",resolveDir:process.cwd()},bundle:true,write:false,platform:'browser',format:'iife',target:'es2020'});let workers=0;runInNewContext(ssr.outputFiles[0].text,{Worker(){workers++;throw Error('SSR worker');}});assert.equal(workers,0);
 writeFileSync(`${runtime}/main-thread-dependency-graph.json`,JSON.stringify({inputs,heavyNumericalEnginesExcluded:true,mathematicalValidatorsExcluded:true,ssrWorkersCreated:workers},null,2)+'\n');
});
test('default copy scheduling lets ordinary timers run between chunks',async()=>{
 let ticks=0;const timer=setInterval(()=>{ticks++;},0);
 try{const value:any=await copy(Array(20000).fill(1),{nodesPerTask:128});assert.equal(value.length,20000);assert(Object.isFrozen(value));assert(ticks>2,'ordinary timer queue must get execution time');}finally{clearInterval(timer);}
});

test('React hook renders on the server without creating workers or publishing a result',()=>{
 function Probe(){const s=useManyBodyCalculation(input());assert.equal(s.response,null);return createElement('span',null,s.status);}
 assert.equal(renderToString(createElement(Probe)),'<span>idle</span>');
});
after(()=>writeFileSync(`${runtime}/metrics.json`,JSON.stringify({validFamilies,rejectedFamilies},null,2)+'\n'));
