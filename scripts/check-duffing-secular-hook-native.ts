import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {build} from 'esbuild';
import {chromium} from 'playwright';
import {duffingSecularExamples} from '../src/lib/learning-duffing-secular-input';
const root='.runtime/learning/duffing-secular-worker';
type HookState={status:string;phase:string|null;response:unknown;retry:()=>void};
type Harness=Window&{__initial:unknown;__hook:HookState;__set:(input:unknown,enabled:boolean)=>void;__unmount:()=>void;__control:{hidden:boolean;mode:string;created:number;terminated:number;active:Set<Worker>}};
async function main(){
 const codes={calculate:readFileSync(`${root}/calculate.js`,'utf8'),validate:readFileSync(`${root}/validate.js`,'utf8')};
 const bundle=await build({stdin:{contents:`import{createElement,useState}from'react';import{createRoot}from'react-dom/client';import{useDuffingSecularCalculation}from'./src/components/learning/use-duffing-secular-calculation';function Probe(){const[input,setInput]=useState(window.__initial),[enabled,setEnabled]=useState(true);const state=useDuffingSecularCalculation(input,enabled);window.__hook=state;window.__set=(next,on)=>{setInput(next);setEnabled(on);};return createElement('output',{id:'hook-state'},state.status);}const node=document.createElement('div');document.body.append(node);const root=createRoot(node);window.__unmount=()=>root.unmount();root.render(createElement(Probe));`,resolveDir:process.cwd(),loader:'tsx'},bundle:true,write:false,platform:'browser',format:'esm',metafile:true});
 assert(!Object.keys(bundle.metafile!.inputs).some(p=>/learning-(?:duffing-secular|graph-gauge|quantum-hilbert-core)(?:\.ts|-validation\.ts|-evaluator\.ts|-validation-evaluator\.ts)/.test(p)));writeFileSync(`${root}/hook-dependency-graph.json`,JSON.stringify(bundle.metafile,null,2)+'\n');
 const browser=await chromium.launch({headless:true}),errors:string[]=[];
 try{
  const context=await browser.newContext();await context.addInitScript({content:'globalThis.__name=fn=>fn;'});const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/__duffing_secular_hook__/',r=>r.fulfill({contentType:'text/html',body:'<!doctype html><html lang="en"><title>Local perturbation hook checks</title><body></body></html>'}));
  await page.route('**/learning-duffing-secular.worker.ts',r=>r.fulfill({contentType:'text/javascript',body:codes.calculate}));await page.route('**/learning-duffing-secular-validation.worker.ts',r=>r.fulfill({contentType:'text/javascript',body:codes.validate}));
  await page.goto('http://127.0.0.1:4175/__duffing_secular_hook__/');
  await page.evaluate(input=>{
   const w=window as unknown as Harness;w.__initial=input;const control={hidden:false,mode:'normal',created:0,terminated:0,active:new Set<Worker>()};w.__control=control;Object.defineProperty(document,'visibilityState',{configurable:true,get:()=>control.hidden?'hidden':'visible'});const Native=Worker;
   window.Worker=class extends Native{
    channel:string;
    constructor(url:string|URL,options?:WorkerOptions){super(url,options);this.channel=options?.name??'';control.created++;control.active.add(this);}
    postMessage(message:unknown,options?:StructuredSerializeOptions|Transferable[]){if(control.mode==='throw-calc'&&this.channel==='duffing-secular-calculation')throw Error('Intentional postMessage failure');if((control.mode==='hang-calc'&&this.channel==='duffing-secular-calculation')||(control.mode==='hang-validation'&&this.channel==='duffing-secular-validation'))return;if(Array.isArray(options))super.postMessage(message,options);else super.postMessage(message,options);}
    terminate(){if(control.active.delete(this)){control.terminated++;super.terminate();}}
   };
  },duffingSecularExamples[0].input);
  await page.addScriptTag({type:'module',content:bundle.outputFiles[0].text});
  const wait=(status:string,phase:string|null=null)=>page.waitForFunction(({status,phase})=>{const s=(window as unknown as Harness).__hook;return s?.status===status&&(!phase||s.phase===phase);},{status,phase},{timeout:30000});
  const visible=async(shown:boolean)=>page.evaluate(shown=>{(window as unknown as Harness).__control.hidden=!shown;document.dispatchEvent(new Event('visibilitychange'));},shown);
  const mode=async(value:string)=>page.evaluate(value=>{(window as unknown as Harness).__control.mode=value;},value);
  const noResponse=async()=>assert.equal(await page.evaluate(()=>(window as unknown as Harness).__hook.response),null);
  const released=async()=>assert.equal(await page.evaluate(()=>(window as unknown as Harness).__control.active.size),0);
  await wait('ready');await released();
  const large=duffingSecularExamples.find(c=>c.id==='maximum-coprime')!.input;await page.evaluate(input=>(window as unknown as Harness).__set(input,false),large);await wait('idle');await noResponse();
  await mode('hang-calc');await page.evaluate(input=>(window as unknown as Harness).__set(input,true),large);await wait('pending','starting');await noResponse();await visible(false);await wait('idle');await released();
  await mode('normal');await visible(true);await wait('ready');
  await mode('hang-validation');await page.evaluate(()=>(window as unknown as Harness).__hook.retry());await wait('pending','validating');await noResponse();await visible(false);await wait('idle');await released();
  await mode('normal');await visible(true);await wait('ready');
  await mode('hang-validation');await page.evaluate(()=>(window as unknown as Harness).__hook.retry());await wait('pending','validating');
  await mode('normal');await page.evaluate(input=>(window as unknown as Harness).__set(input,true),duffingSecularExamples.find(c=>c.id==='harmonic')!.input);await wait('ready');
  assert.equal(await page.evaluate(()=>(window as unknown as Harness).__hook.response&&( (window as unknown as Harness).__hook.response as {result:{mode:string}}).result.mode),'harmonic');
  await mode('throw-calc');await page.evaluate(()=>(window as unknown as Harness).__hook.retry());await wait('error');await noResponse();await released();
  await mode('normal');await page.evaluate(()=>(window as unknown as Harness).__hook.retry());await wait('ready');await released();
  await mode('hang-calc');await page.evaluate(()=>(window as unknown as Harness).__hook.retry());await wait('pending');await page.evaluate(()=>(window as unknown as Harness).__unmount());await released();
  const counts=await page.evaluate(()=>{const c=(window as unknown as Harness).__control;return{created:c.created,terminated:c.terminated,active:c.active.size};});assert.equal(counts.created,counts.terminated);assert.deepEqual(errors,[]);
  const report={...counts,initialReady:true,disabledHidesOldResult:true,hiddenStartingCancelled:true,hiddenValidationCancelled:true,visibleResumes:true,newInputReplacesVerification:true,retry:true,errorRetryVerified:true,pendingUnmount:true,visibilityEventSimulation:true,physicalHiddenTabAutomation:false,actualModuleWorkers:true,intentionalPostMessageHolds:true,productionNextAssetsVerified:false,readerRegistered:false,errors};writeFileSync(`${root}/native-hook.json`,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
 }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
