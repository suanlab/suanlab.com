import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {build} from 'esbuild';
import {chromium} from 'playwright';
import {quantumInformationPresets} from '../src/lib/learning-quantum-information-presets';
const runtime='.runtime/learning/quantum-information-workers';
type HookState={status:string;phase:string|null;response:unknown;retry:()=>void};
type Harness=Window&{__initial:unknown;__hook:HookState;__set:(input:unknown,enabled:boolean)=>void;__unmount:()=>void;__control:{hidden:boolean;created:number;terminated:number;active:Set<Worker>}};
async function main(){
 const calculate=readFileSync(`${runtime}/calculate.js`,'utf8'),validate=readFileSync(`${runtime}/validate.js`,'utf8');
 const result=await build({stdin:{contents:`import{createElement,useState}from'react';import{createRoot}from'react-dom/client';import{useInformationCalculation}from'./src/components/learning/use-information-calculation';function Probe(){const[input,setInput]=useState(window.__initial),[enabled,setEnabled]=useState(true);const state=useInformationCalculation(input,enabled);window.__hook=state;window.__set=(next,on)=>{setInput(next);setEnabled(on);};return createElement('output',{id:'hook-state'},state.status);}const node=document.createElement('div');document.body.append(node);const root=createRoot(node);window.__unmount=()=>root.unmount();root.render(createElement(Probe));`,resolveDir:process.cwd(),loader:'tsx'},bundle:true,write:false,platform:'browser',format:'esm',metafile:true});
 const inputs=Object.keys(result.metafile!.inputs);assert(!inputs.some(p=>/learning-quantum-hilbert-core|learning-quantum-information-(?:response|calculation|.*validator|evaluator)/.test(p)));
 writeFileSync(`${runtime}/hook-dependency-graph.json`,JSON.stringify(result.metafile,null,2)+'\n');
 const browser=await chromium.launch({headless:true}),errors:string[]=[];
 try{
  const context=await browser.newContext();await context.addInitScript({content:'globalThis.__name=(fn)=>fn;'});const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/__information_hook_harness__/',r=>r.fulfill({contentType:'text/html',body:'<!doctype html><html lang="en"><title>Information hook check</title><body></body></html>'}));
  await page.route('**/learning-quantum-information.worker.ts',r=>r.fulfill({contentType:'text/javascript',body:calculate}));
  await page.route('**/learning-quantum-information-validation.worker.ts',r=>r.fulfill({contentType:'text/javascript',body:validate}));
  await page.goto('http://127.0.0.1:4175/__information_hook_harness__/');
  const first=Object.values(quantumInformationPresets)[0][0].input,maximum=Object.values(quantumInformationPresets).flat().find(p=>p.input.kind==='bb84')!.input;
  assert(maximum.kind==='bb84');const large={...maximum,attackFraction:'1/99999999999999999999999999999999',sampleSize:256};
  await page.evaluate(input=>{
   const w=window as unknown as Harness;w.__initial=input;const control={hidden:false,created:0,terminated:0,active:new Set<Worker>()};w.__control=control;
   Object.defineProperty(document,'visibilityState',{configurable:true,get:()=>control.hidden?'hidden':'visible'});
   const NativeWorker=window.Worker;
   window.Worker=class extends NativeWorker{constructor(url:string|URL,options?:WorkerOptions){super(url,options);control.created++;control.active.add(this);}terminate(){if(control.active.delete(this)){control.terminated++;super.terminate();}}};
  },first);
  await page.addScriptTag({type:'module',content:result.outputFiles[0].text});
  const wait=async(status:string,phase:string|null=null)=>page.waitForFunction(({status,phase})=>{const h=(window as unknown as Harness).__hook;return h?.status===status&&(!phase||h.phase===phase);},{status,phase},{timeout:30000});
  await wait('ready');assert(await page.evaluate(()=>(window as unknown as Harness).__hook.response!==null));
  await page.evaluate(input=>(window as unknown as Harness).__set(input,false),large);await wait('idle');assert.equal(await page.evaluate(()=>(window as unknown as Harness).__hook.response),null);
  await page.evaluate(input=>(window as unknown as Harness).__set(input,true),large);await wait('pending','calculating');
  assert.equal(await page.evaluate(()=>(window as unknown as Harness).__hook.response),null);
  await page.evaluate(()=>{(window as unknown as Harness).__control.hidden=true;document.dispatchEvent(new Event('visibilitychange'));});await wait('idle');assert.equal(await page.evaluate(()=>(window as unknown as Harness).__control.active.size),0);
  await page.evaluate(()=>{(window as unknown as Harness).__control.hidden=false;document.dispatchEvent(new Event('visibilitychange'));});await wait('ready');
  await page.evaluate(()=>(window as unknown as Harness).__hook.retry());await wait('pending','validating');
  await page.evaluate(()=>{(window as unknown as Harness).__control.hidden=true;document.dispatchEvent(new Event('visibilitychange'));});await wait('idle');assert.equal(await page.evaluate(()=>(window as unknown as Harness).__control.active.size),0);
  await page.evaluate(()=>{(window as unknown as Harness).__control.hidden=false;document.dispatchEvent(new Event('visibilitychange'));});await wait('ready');
  await page.evaluate(()=>(window as unknown as Harness).__hook.retry());await wait('pending');await page.evaluate(()=>(window as unknown as Harness).__unmount());
  const counts=await page.evaluate(()=>{const c=(window as unknown as Harness).__control;return {created:c.created,terminated:c.terminated,active:c.active.size};});assert.equal(counts.active,0);assert.equal(counts.created,counts.terminated);assert.deepEqual(errors,[]);
  const report={...counts,initialReady:true,disabledInputHidesOldResponse:true,hiddenCalculationCancelled:true,hiddenValidationCancelled:true,visibleResumes:true,retry:true,unmountStopsWorkers:true,visibilityChangeEventsTested:true,readerRegistered:false,pageErrors:errors};
  writeFileSync(`${runtime}/hook-native-verification.json`,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
 }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
