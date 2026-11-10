import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {build} from 'esbuild';
import {chromium} from 'playwright';
import {calculateSaddleFluctuation} from '../src/lib/learning-saddle-fluctuation';
import {saddleFluctuationExamples} from '../src/lib/learning-saddle-fluctuation-input';
const root='.runtime/learning/saddle-fluctuation-worker';
type HookState={status:string;phase:string|null;response:unknown;error:string|null;retry:()=>void;cancel:()=>void};
type Harness=Window&{__initial:unknown;__hook:HookState;__set:(input:unknown,enabled:boolean)=>void;__unmount:()=>void;__control:{hidden:boolean;mode:string;created:number;terminated:number;calculationPosted:number;validationPosted:number;active:Set<Worker>}};
async function main(){
 const codes={calculate:readFileSync(`${root}/calculate.js`,'utf8'),validate:readFileSync(`${root}/validate.js`,'utf8')};
 const bundle=await build({stdin:{contents:`import{createElement,useState}from'react';import{createRoot}from'react-dom/client';import{useSaddleFluctuationCalculation}from'./src/components/learning/use-saddle-fluctuation-calculation';function Probe(){const[input,setInput]=useState(window.__initial),[enabled,setEnabled]=useState(true);const state=useSaddleFluctuationCalculation(input,enabled);window.__hook=state;window.__set=(next,on)=>{setInput(next);setEnabled(on);};return createElement('output',{id:'hook-state'},state.status);}const node=document.createElement('div');document.body.append(node);const root=createRoot(node);window.__unmount=()=>root.unmount();root.render(createElement(Probe));`,resolveDir:process.cwd(),loader:'tsx'},bundle:true,write:false,platform:'browser',format:'esm',metafile:true});
 assert(!Object.keys(bundle.metafile!.inputs).some(p=>/learning-(?:saddle-fluctuation|graph-gauge|quantum-hilbert-core)(?:\.ts|-validation\.ts|-evaluator\.ts|-validation-evaluator\.ts)/.test(p)));writeFileSync(`${root}/hook-dependency-graph.json`,JSON.stringify(bundle.metafile,null,2)+'\n');
 const browser=await chromium.launch({headless:true}),errors:string[]=[];
 try{
  const context=await browser.newContext();await context.addInitScript({content:'globalThis.__name=fn=>fn;'});const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/__saddle_fluctuation_hook__/',r=>r.fulfill({contentType:'text/html',body:'<!doctype html><html lang="en"><title>Local Saddle fluctuation hook checks</title><body><label>Typing<input id="typing"></label></body></html>'}));
  await page.route('**/learning-saddle-fluctuation.worker.ts',r=>r.fulfill({contentType:'text/javascript',body:codes.calculate}));await page.route('**/learning-saddle-fluctuation-validation.worker.ts',r=>r.fulfill({contentType:'text/javascript',body:codes.validate}));
  await page.goto('http://127.0.0.1:4175/__saddle_fluctuation_hook__/');
  await page.evaluate(input=>{
   const w=window as unknown as Harness;w.__initial=input;const control={hidden:false,mode:'normal',created:0,terminated:0,calculationPosted:0,validationPosted:0,active:new Set<Worker>()};w.__control=control;Object.defineProperty(document,'visibilityState',{configurable:true,get:()=>control.hidden?'hidden':'visible'});const Native=Worker;
   window.Worker=class extends Native{
    channel:string;
    constructor(url:string|URL,options?:WorkerOptions){super(url,options);this.channel=options?.name??'';control.created++;control.active.add(this);}
    postMessage(message:unknown,options?:StructuredSerializeOptions|Transferable[]){if(control.mode==='throw-calc'&&this.channel==='saddle-fluctuation-calculation')throw Error('Intentional postMessage failure');if((control.mode==='hang-calc'&&this.channel==='saddle-fluctuation-calculation')||(control.mode==='hang-validation'&&this.channel==='saddle-fluctuation-validation'))return;if(this.channel==='saddle-fluctuation-calculation')control.calculationPosted++;if(this.channel==='saddle-fluctuation-validation')control.validationPosted++;if(Array.isArray(options))super.postMessage(message,options);else super.postMessage(message,options);}
    terminate(){if(control.active.delete(this)){control.terminated++;super.terminate();}}
   };
  },saddleFluctuationExamples[0].input);
  await page.addScriptTag({type:'module',content:bundle.outputFiles[0].text});
  const wait=(status:string,phase:string|null=null)=>page.waitForFunction(({status,phase})=>{const s=(window as unknown as Harness).__hook;return s?.status===status&&(!phase||s.phase===phase);},{status,phase},{timeout:120000});
  const visible=async(shown:boolean)=>page.evaluate(shown=>{(window as unknown as Harness).__control.hidden=!shown;document.dispatchEvent(new Event('visibilitychange'));},shown);
  const mode=async(value:string)=>page.evaluate(value=>{(window as unknown as Harness).__control.mode=value;},value);
  const noResponse=async()=>assert.equal(await page.evaluate(()=>(window as unknown as Harness).__hook.response),null);
  const released=async()=>assert.equal(await page.evaluate(()=>(window as unknown as Harness).__control.active.size),0);
  await wait('ready');await released();
  if(process.env.SADDLE_FLUCTUATION_NATIVE_FLOW==='active-cancel'){
   const input=saddleFluctuationExamples.find(c=>c.id==='mixed-coprime-denominators')!.input;
   await page.evaluate(input=>(window as unknown as Harness).__set(input,true),input);await wait('pending','calculating');
   await page.locator('#typing').fill('Typing during a dispatched maximum calculation');
   assert.equal(await page.locator('#typing').inputValue(),'Typing during a dispatched maximum calculation');
   assert.equal(await page.evaluate(()=>(window as unknown as Harness).__hook.status),'pending');
   await page.evaluate(()=>(window as unknown as Harness).__hook.cancel());await wait('idle');await noResponse();await released();
   const before=await page.evaluate(()=>(window as unknown as Harness).__control.validationPosted);
   await page.evaluate(()=>(window as unknown as Harness).__hook.retry());await wait('pending','validating');
   await page.waitForFunction(before=>(window as unknown as Harness).__control.validationPosted>before,before);
   await page.waitForTimeout(100);assert.equal(await page.evaluate(()=>(window as unknown as Harness).__hook.status),'pending');
   await page.locator('#typing').fill('Typing after maximum validation dispatch');
   assert.equal(await page.locator('#typing').inputValue(),'Typing after maximum validation dispatch');
   await page.evaluate(()=>(window as unknown as Harness).__hook.cancel());await wait('idle');await noResponse();await released();
   await page.waitForTimeout(100);assert.equal(await page.evaluate(()=>(window as unknown as Harness).__hook.status),'idle');
   await page.evaluate(()=>(window as unknown as Harness).__hook.retry());await wait('ready');await released();
   const response=await page.evaluate(()=>(window as unknown as Harness).__hook.response) as {result:unknown};
   assert.deepEqual(response.result,calculateSaddleFluctuation(saddleFluctuationExamples.find(c=>c.id==='mixed-coprime-denominators')!.input));
   await page.evaluate(()=>(window as unknown as Harness).__unmount());await released();
   const counts=await page.evaluate(()=>{const c=(window as unknown as Harness).__control;return{created:c.created,terminated:c.terminated,active:c.active.size,calculationPosted:c.calculationPosted,validationPosted:c.validationPosted};});
   assert.equal(counts.created,counts.terminated);assert.deepEqual(errors,[]);
   const report={...counts,maximumInput:'mixed-coprime-denominators',calculationStartedBeforeCancel:true,validationDispatchedBeforeCancel:true,typingDuringBothPhases:true,lateReadyAbsentAfterCancel:true,retryVerifiedAgainstNodeProducer:true,PythonNumericsVerifiedSeparately:true,intentionalPostMessageHolds:false,visibilityEventSimulation:true,physicalHiddenTabAutomation:false,actualModuleWorkers:true,productionNextAssetsVerified:false,readerRegistered:false,errors};
   writeFileSync(`${root}/native-active-cancellation.json`,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));return;
  }
  const large=saddleFluctuationExamples.find(c=>c.id==='mixed-coprime-denominators')!.input;await page.evaluate(input=>(window as unknown as Harness).__set(input,false),large);await wait('idle');await noResponse();
  await mode('hang-calc');await page.evaluate(input=>(window as unknown as Harness).__set(input,true),large);await wait('pending','starting');await noResponse();await visible(false);await wait('idle');await released();
  await mode('normal');await visible(true);await wait('pending','calculating');
  await page.locator('#typing').fill('Typing during the maximum exact calculation');
  assert.equal(await page.locator('#typing').inputValue(),'Typing during the maximum exact calculation');
  assert.equal(await page.evaluate(()=>(window as unknown as Harness).__hook.status),'pending');
  await wait('ready');
  await mode('hang-validation');await page.evaluate(()=>(window as unknown as Harness).__hook.retry());await wait('pending','validating');await noResponse();await visible(false);await wait('idle');await released();
  await mode('normal');await visible(true);await wait('ready');
  await mode('hang-validation');await page.evaluate(()=>(window as unknown as Harness).__hook.retry());await wait('pending','validating');
  await mode('normal');await page.evaluate(input=>(window as unknown as Harness).__set(input,true),saddleFluctuationExamples.find(c=>c.id==='double-hbar')!.input);await wait('ready');
  assert.equal(await page.evaluate(()=>(window as unknown as Harness).__hook.response&&( (window as unknown as Harness).__hook.response as {result:{input:{hbar:string}}}).result.input.hbar),'2');
  const beforeInvalid=await page.evaluate(()=>(window as unknown as Harness).__control.created);
  await page.evaluate(input=>(window as unknown as Harness).__set({...input,time:'0'},true),saddleFluctuationExamples[0].input);await wait('error');await noResponse();await released();
  assert.equal(await page.evaluate(()=>(window as unknown as Harness).__control.created),beforeInvalid);
  assert.equal(await page.evaluate(()=>(window as unknown as Harness).__hook.error),'invalid-input');
  const reordered=Object.fromEntries(Object.entries(saddleFluctuationExamples.find(c=>c.id==='double-hbar')!.input).reverse());
  await page.evaluate(input=>(window as unknown as Harness).__set(input,true),reordered);await wait('ready');await released();
  await mode('throw-calc');await page.evaluate(()=>(window as unknown as Harness).__hook.retry());await wait('error');await noResponse();await released();
  await mode('normal');await page.evaluate(()=>(window as unknown as Harness).__hook.retry());await wait('ready');await released();
  await mode('hang-calc');await page.evaluate(()=>(window as unknown as Harness).__hook.retry());await wait('pending');await page.evaluate(()=>(window as unknown as Harness).__unmount());await released();
  const counts=await page.evaluate(()=>{const c=(window as unknown as Harness).__control;return{created:c.created,terminated:c.terminated,active:c.active.size};});assert.equal(counts.created,counts.terminated);assert.deepEqual(errors,[]);
  const report={...counts,initialReady:true,maximumTypingResponsive:true,invalidInputReportsErrorWithoutWorker:true,reorderedFieldsRemainReady:true,disabledHidesOldResult:true,hiddenStartingCancelled:true,hiddenValidationCancelled:true,visibleResumes:true,newInputReplacesVerification:true,retry:true,errorRetryVerified:true,pendingUnmount:true,visibilityEventSimulation:true,physicalHiddenTabAutomation:false,actualModuleWorkers:true,intentionalPostMessageHolds:true,productionNextAssetsVerified:false,readerRegistered:false,errors};writeFileSync(`${root}/native-hook.json`,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
 }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
