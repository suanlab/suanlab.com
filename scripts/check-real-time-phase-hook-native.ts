import assert from 'node:assert/strict';
import {mkdirSync,readFileSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {build} from 'esbuild';
import {chromium} from 'playwright';
import {realTimePhaseExamples,realTimePacketDefault} from '../src/lib/learning-real-time-phase-input';
const root='.runtime/learning/real-time-phase-worker';mkdirSync(root,{recursive:true});
type HookState={status:string;phase:string|null;response:unknown;error:string|null;retry:()=>void;cancel:()=>void};
type Harness=Window&{__initial:unknown;__hook:HookState;__set:(input:unknown,enabled:boolean)=>void;__unmount:()=>void;
 __control:{hidden:boolean;mode:string;created:number;terminated:number;calculationStarted:number;verificationStarted:number;active:Set<Worker>};
 __deadlineState:{status:string;error?:string};__beginDeadline:(input:unknown,target:string)=>void;__retryDeadline:()=>void;__disposeDeadline:()=>void};
const stable=(value:unknown):string=>Array.isArray(value)?`[${value.map(stable).join(',')}]`:value&&typeof value==='object'?`{${Object.keys(value).sort().map(k=>`${JSON.stringify(k)}:${stable((value as Record<string,unknown>)[k])}`).join(',')}}`:JSON.stringify(value);
async function main() {
 const codes={calculate:readFileSync(root+'/calculate.js','utf8'),validate:readFileSync(root+'/validate.js','utf8')};
 const contents=`import{createElement,useState}from'react';import{createRoot}from'react-dom/client';import{useRealTimePhaseCalculation}from'./src/components/learning/use-real-time-phase-calculation';import{RealTimePhaseCalculationController as Controller}from'./src/lib/learning-real-time-phase-controller';
 function Probe(){const[input,setInput]=useState(window.__initial),[enabled,setEnabled]=useState(true);window.__hook=useRealTimePhaseCalculation(input,enabled);window.__set=(v,on)=>{setInput(v);setEnabled(on);};return createElement('output',null,window.__hook.status);}const node=document.createElement('div');document.body.append(node);const root=createRoot(node);window.__unmount=()=>root.unmount();root.render(createElement(Probe));
 window.__beginDeadline=(input,target)=>{let mode=target;const controller=new Controller({calculate:()=>new Worker(mode==='calculate'?'busy-calculate.js':'learning-real-time-phase.worker.ts',{type:'module',name:'real-time-phase-calculation'}),validate:()=>new Worker(mode==='validate'?'busy-validate.js':'learning-real-time-phase-validation.worker.ts',{type:'module',name:'real-time-phase-validation'})},state=>window.__deadlineState=state,{},10000);window.__retryDeadline=()=>{mode='normal';controller.run(input);};window.__disposeDeadline=()=>controller.dispose();controller.run(input);};`;
 const bundle=await build({stdin:{contents,resolveDir:process.cwd(),loader:'tsx'},bundle:true,write:false,platform:'browser',format:'esm',target:'es2020',metafile:true});
 const inputs=Object.keys(bundle.metafile!.inputs);for(const file of ['learning-real-time-phase.ts','learning-real-time-phase-validation.ts','learning-quantum-hilbert-core.ts'])assert(!inputs.includes('src/lib/'+file));assert(!bundle.outputFiles[0].text.includes('BigInt('));
 writeFileSync(root+'/hook-native-dependency-graph.json',JSON.stringify(bundle.metafile,null,2)+'\n');
 const browser=await chromium.launch({headless:true}),errors:string[]=[];
 try {
  const context=await browser.newContext();await context.addInitScript({content:'globalThis.__name=fn=>fn;'});const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/__real_time_phase_hook__/',r=>r.fulfill({contentType:'text/html',body:'<!doctype html><html lang="en"><title>Real-time hook verification</title><body><label>Typing<input id="typing"></label></body></html>'}));
  await page.route('**/learning-real-time-phase.worker.ts',r=>r.fulfill({contentType:'text/javascript',body:codes.calculate}));
  await page.route('**/learning-real-time-phase-validation.worker.ts',r=>r.fulfill({contentType:'text/javascript',body:codes.validate}));
  for(const phase of ['calculate','validate'])await page.route('**/busy-'+phase+'.js',r=>r.fulfill({contentType:'text/javascript',body:`self.onmessage=e=>{self.postMessage({version:1,requestId:e.data.requestId,phase:'${phase==='calculate'?'started':'verification-started'}',kind:'real-time-phase'});while(true){}};`}));
  await page.goto('http://127.0.0.1:4175/__real_time_phase_hook__/');
  const small=realTimePhaseExamples.find(x=>x.id==='negative-mode-hand')!.input,large=realTimePhaseExamples.find(x=>x.id==='maximum-mixed-coprime')!.input;
  await page.evaluate(input=>{
   const w=window as unknown as Harness;w.__initial=input;const control={hidden:false,mode:'normal',created:0,terminated:0,calculationStarted:0,verificationStarted:0,active:new Set<Worker>()};w.__control=control;
   Object.defineProperty(document,'visibilityState',{configurable:true,get:()=>control.hidden?'hidden':'visible'});const Native=Worker;
   window.Worker=class extends Native {
    channel:string;
    constructor(url:string|URL,options?:WorkerOptions){super(url,options);this.channel=options?.name??'';control.created++;control.active.add(this);this.addEventListener('message',e=>{if(e.data.phase==='started')control.calculationStarted++;if(e.data.phase==='verification-started')control.verificationStarted++;});}
    postMessage(message:unknown,options?:StructuredSerializeOptions|Transferable[]){if(control.mode==='throw-calc'&&this.channel==='real-time-phase-calculation')throw Error('Intentional post failure');if(control.mode==='hang-calc'&&this.channel==='real-time-phase-calculation'||control.mode==='hang-validation'&&this.channel==='real-time-phase-validation')return;if(Array.isArray(options))super.postMessage(message,options);else super.postMessage(message,options);}
    terminate(){if(control.active.delete(this)){control.terminated++;super.terminate();}}
   };
  },small);
  await page.addScriptTag({type:'module',content:bundle.outputFiles[0].text});
  const wait=(status:string,phase:string|null=null)=>page.waitForFunction(({status,phase})=>{const s=(window as unknown as Harness).__hook;return s?.status===status&&(!phase||s.phase===phase);},{status,phase},{timeout:120000});
  const noResponse=async()=>assert.equal(await page.evaluate(()=>(window as unknown as Harness).__hook.response),null);
  const released=async()=>assert.equal(await page.evaluate(()=>(window as unknown as Harness).__control.active.size),0);
  const mode=(v:string)=>page.evaluate(v=>{(window as unknown as Harness).__control.mode=v;},v);
  const show=(visible:boolean)=>page.evaluate(visible=>{(window as unknown as Harness).__control.hidden=!visible;document.dispatchEvent(new Event('visibilitychange'));},visible);
  await wait('ready');await released();console.log('Initial hook ready');
  await page.evaluate(input=>(window as unknown as Harness).__set(input,true),large);await wait('pending','calculating');
  await page.locator('#typing').fill('Typing during actual maximum calculation');assert.equal(await page.evaluate(()=>(window as unknown as Harness).__hook.status),'pending');
  await page.evaluate(()=>(window as unknown as Harness).__hook.cancel());await wait('idle');await noResponse();await released();console.log('Cancelled active calculation');
  const before=await page.evaluate(()=>(window as unknown as Harness).__control.verificationStarted);
  await page.evaluate(()=>(window as unknown as Harness).__hook.retry());await wait('pending','validating');
  await page.waitForFunction(before=>(window as unknown as Harness).__control.verificationStarted>before,before,{timeout:120000});
  await page.locator('#typing').fill('Typing during actual maximum independent verification');assert.equal(await page.evaluate(()=>(window as unknown as Harness).__hook.status),'pending');
  await page.evaluate(()=>(window as unknown as Harness).__hook.cancel());await wait('idle');await noResponse();await released();console.log('Cancelled active verification after started signal');
  await page.waitForTimeout(100);assert.equal(await page.evaluate(()=>(window as unknown as Harness).__hook.status),'idle');
  await page.evaluate(()=>(window as unknown as Harness).__hook.retry());await wait('ready');await released();
  const result=await page.evaluate(()=>(window as unknown as Harness).__hook.response)as{result:unknown};
  const baseline=JSON.parse(readFileSync(root+'/node-baseline-hashes.json','utf8'))as string[],index=realTimePhaseExamples.findIndex(x=>x.id==='maximum-mixed-coprime');
  assert.equal(createHash('sha256').update(stable(result.result)).digest('hex'),baseline[index]);
  assert(await page.evaluate(()=>{const r=(window as unknown as Harness).__hook.response as{result:{exact:unknown;wave:unknown}};return Object.isFrozen(r)&&Object.isFrozen(r.result)&&Object.isFrozen(r.result.exact)&&Object.isFrozen(r.result.wave);}));console.log('Maximum retry ready andfrozen');
  await mode('hang-validation');await page.evaluate(input=>(window as unknown as Harness).__set(input,true),small);await wait('pending','validating');await noResponse();
  await mode('normal');await page.evaluate(input=>(window as unknown as Harness).__set(input,true),realTimePacketDefault);await wait('ready');await released();
  assert.equal(await page.evaluate(()=>((window as unknown as Harness).__hook.response as{result:{kind:string}}).result.kind),'packet');
  const count=await page.evaluate(()=>(window as unknown as Harness).__control.created),reordered=Object.fromEntries(Object.entries(realTimePacketDefault).reverse());
  await page.evaluate(input=>(window as unknown as Harness).__set(input,true),reordered);await page.waitForTimeout(50);await wait('ready');
  assert.equal(await page.evaluate(()=>(window as unknown as Harness).__control.created),count);
  await page.evaluate(input=>(window as unknown as Harness).__set({...input,time:'0'},true),small);await wait('error');await noResponse();await released();assert.equal(await page.evaluate(()=>(window as unknown as Harness).__hook.error),'invalid-input');
  assert.equal(await page.evaluate(()=>(window as unknown as Harness).__control.created),count);
  await page.evaluate(input=>(window as unknown as Harness).__set(input,false),small);await wait('idle');await noResponse();
  await mode('hang-calc');await page.evaluate(input=>(window as unknown as Harness).__set(input,true),small);await wait('pending','starting');await show(false);await wait('idle');await released();
  await mode('normal');await show(true);await wait('ready');await released();
  await mode('hang-validation');await page.evaluate(()=>(window as unknown as Harness).__hook.retry());await wait('pending','validating');await show(false);await wait('idle');await released();
  await mode('normal');await show(true);await wait('ready');await released();
  await mode('throw-calc');await page.evaluate(()=>(window as unknown as Harness).__hook.retry());await wait('error');await noResponse();await released();
  assert.equal(await page.evaluate(()=>(window as unknown as Harness).__hook.error),'worker-failed');
  await mode('normal');await page.evaluate(()=>(window as unknown as Harness).__hook.retry());await wait('ready');await released();
  await mode('hang-calc');await page.evaluate(()=>(window as unknown as Harness).__hook.retry());await wait('pending');await page.evaluate(()=>(window as unknown as Harness).__unmount());await released();await mode('normal');console.log('Hook replacement,visibility,error retry andunmount passed');
  for(const target of ['calculate','validate']) {
   const counter=await page.evaluate(target=>target==='calculate'?(window as unknown as Harness).__control.calculationStarted:(window as unknown as Harness).__control.verificationStarted,target);
   await page.evaluate(({input,target})=>(window as unknown as Harness).__beginDeadline(input,target),{input:small,target});
   await page.waitForFunction(({target,counter})=>{const c=(window as unknown as Harness).__control;return(target==='calculate'?c.calculationStarted:c.verificationStarted)>counter;},{target,counter});
   await page.locator('#typing').fill('Typing while actual '+target+' Worker spins');
   await page.waitForFunction(()=>(window as unknown as Harness).__deadlineState?.status==='error',undefined,{timeout:30000});
   assert.equal(await page.evaluate(()=>(window as unknown as Harness).__deadlineState.error),'timeout');await released();
   await page.evaluate(()=>(window as unknown as Harness).__retryDeadline());await page.waitForFunction(()=>(window as unknown as Harness).__deadlineState.status==='ready',undefined,{timeout:30000});
   await page.evaluate(()=>(window as unknown as Harness).__disposeDeadline());await released();console.log('Actual spinning '+target+' Worker deadline andretry passed');
  }
  const counts=await page.evaluate(()=>{const c=(window as unknown as Harness).__control;return{created:c.created,terminated:c.terminated,active:c.active.size,calculationStarted:c.calculationStarted,verificationStarted:c.verificationStarted};});
  assert.equal(counts.created,counts.terminated);assert.deepEqual(errors,[]);
  const report={...counts,actualModuleWorkers:true,maximumInput:'maximum-mixed-coprime',calculationStartedBeforeCancel:true,verificationStartedBeforeCancel:true,typingDuringBothActualPhases:true,noLateReadyAfterCancel:true,maximumRetryMatchesNodeProducer:true,mainThreadOwnedResultFrozen:true,newInputReplacesVerification:true,reorderedInputDoesNotRecalculate:true,invalidInputCreatesNoWorker:true,disabledHidesResult:true,hiddenCalculationAndValidationCancel:true,visibleResumes:true,errorRetry:true,pendingUnmount:true,actualSpinningWorkerDeadlines:2,deadlineRetry:true,deadlineMilliseconds:10000,productionDeadlineMilliseconds:120000,visibilityEventSimulation:true,physicalHiddenTabAutomation:false,intentionalPostMessageHoldsForVisibilityTests:true,productionNextAssetsVerified:false,readerRegistered:false,errors};
  writeFileSync(root+'/native-hook-lifecycle.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
 }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
