/** Native module worker + React hook QA in an isolated in-memory harness.
 * This is not an actual Next reader/bundle or a completed teaching UI.
 */
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {mkdirSync,writeFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {build} from 'esbuild';
import {chromium,type Page} from 'playwright';
import {correlationKinds as kinds,correlationInputKey,type CorrelationInput} from '../src/lib/learning-correlations-protocol';
const destination='.runtime/learning/correlations-worker';
const normal:CorrelationInput[]=[
 {kind:kinds[0],parameters:{couplingK:-.5,fieldOne:0,fieldTwo:0,probeOne:1,probeTwo:1}},
 {kind:kinds[1],parameters:{spinCount:8,couplingK:-.5}},
 {kind:kinds[2],parameters:{sampleCount:8,correlationRho:-.5}},
];
const options={minimumPrecision:8192,maximumPrecision:8192};
const heavy:CorrelationInput[]=[
 {kind:kinds[0],parameters:{couplingK:Number.MIN_VALUE,fieldOne:.8,fieldTwo:0,probeOne:1,probeTwo:1},options},
 {kind:kinds[1],parameters:{spinCount:128,couplingK:Number.MIN_VALUE},options},
 {kind:kinds[2],parameters:{sampleCount:512,correlationRho:.9999999999999999},options},
];
const entry=`import React from 'react';
import {createRoot} from 'react-dom/client';
import {useCorrelationCalculation} from './src/components/learning/use-correlations-calculation';
import {correlationInputKey} from './src/lib/learning-correlations-protocol';
const api=window.corProbe={errors:[],status:'idle'};
function Probe({input,enabled}){
 const state=useCorrelationCalculation(input,enabled);
 Object.assign(api,{status:state.status,phase:state.phase,error:state.error,resultKey:state.response?.inputKey??null,resultKind:state.response?.kind??null,retry:state.retry,cancel:state.cancel});
 if(state.response&&state.response.inputKey!==correlationInputKey(input))api.errors.push('stale render');
 return <output data-status={state.status}>{state.status}</output>;
}
function App(){
 const [input,setInput]=React.useState(${JSON.stringify(normal[0])}),[mounted,setMounted]=React.useState(true),[enabled,setEnabled]=React.useState(true);
 api.run=setInput;api.setMounted=setMounted;api.setEnabled=setEnabled;
 return mounted?<Probe input={input} enabled={enabled}/>:<output data-unmounted>unmounted</output>;
}
createRoot(document.getElementById('root')).render(<React.StrictMode><App/></React.StrictMode>);`;
interface Probe {status:string;phase:string|null;resultKey:string|null;resultKind:string|null;error:string|null;errors:string[];run:(input:CorrelationInput|null)=>void;retry:()=>void;cancel:()=>void;setMounted:(mounted:boolean)=>void;setEnabled:(enabled:boolean)=>void}
async function status(page:Page,value:string){await page.waitForFunction(value=>(window as unknown as {corProbe:Probe}).corProbe?.status===value,value);}
async function run(page:Page,input:CorrelationInput){await page.evaluate(input=>(window as unknown as {corProbe:Probe}).corProbe.run(input),input);}
async function ready(page:Page,input:CorrelationInput){
 await status(page,'ready');
 const r=await page.evaluate(()=>{const p=(window as unknown as {corProbe:Probe}).corProbe;return {kind:p.resultKind,key:p.resultKey,error:p.error,errors:p.errors};});
 assert.equal(r.kind,input.kind);assert.equal(r.key,correlationInputKey(input));assert.equal(r.error,null);assert.deepEqual(r.errors,[]);
}
async function main(){
 mkdirSync(destination,{recursive:true});
 const [app,worker]=await Promise.all([
  build({stdin:{contents:entry,loader:'jsx',resolveDir:resolve('.'),sourcefile:'correlations-probe.jsx'},bundle:true,write:false,format:'esm',platform:'browser',target:'es2022',metafile:true,logLevel:'silent'}),
  build({entryPoints:['src/lib/learning-correlations.worker.ts'],bundle:true,write:false,format:'esm',platform:'browser',target:'es2022',metafile:true,logLevel:'silent'}),
 ]);
 const appInputs=Object.keys(app.metafile!.inputs),numerical=['learning-binary-correlations.ts','learning-periodic-ising-ring.ts','learning-ar1-average.ts','learning-binary-products.ts','learning-interval-transcendentals.ts'];
 for(const name of numerical)assert.ok(!appInputs.some(p=>p.endsWith(name)),name);
 let failWorker=false;
 const server=createServer((request,response)=>{
  response.setHeader('Cache-Control','no-store');
  if(request.url==='/app.js'){response.setHeader('Content-Type','text/javascript');response.end(app.outputFiles[0].text);}
  else if(request.url==='/learning-correlations.worker.ts'){
   if(failWorker){response.writeHead(404);response.end('intentional test failure');return;}
   response.setHeader('Content-Type','text/javascript');response.end(worker.outputFiles[0].text);
  }else if(request.url==='/'){response.setHeader('Content-Type','text/html');response.end('<!doctype html><html lang="en"><title>Internal worker QA</title><div id="root"></div><script type="module" src="/app.js"></script></html>');}
  else{response.writeHead(404);response.end();}
 });
 await new Promise<void>(resolve=>server.listen(0,'127.0.0.1',resolve));
 const address=server.address();assert.ok(address&&typeof address==='object');
 let browser:Awaited<ReturnType<typeof chromium.launch>>|undefined;
 try{
  browser=await chromium.launch({headless:true,args:['--no-sandbox']});const page=await browser.newPage();const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
  await page.addInitScript(()=>{
   const stats={created:0,terminated:0,started:0,live:0};(window as unknown as {workerStats:typeof stats}).workerStats=stats;
   const NativeWorker=window.Worker;
   window.Worker=class extends NativeWorker{
    private done=false;
    constructor(url:string|URL,options?:WorkerOptions){super(url,options);stats.created++;stats.live++;this.addEventListener('message',e=>{if(e.data?.phase==='started')stats.started++;});}
    terminate(){if(!this.done){this.done=true;stats.terminated++;stats.live--;}super.terminate();}
   };
  });
  await page.goto(`http://127.0.0.1:${address.port}/`);await ready(page,normal[0]);
  const started=async(input:CorrelationInput)=>{await run(page,input);await page.waitForFunction(()=>(window as unknown as {corProbe:Probe}).corProbe.phase==='calculating');};
  for(let i=0;i<3;i++){
   await started(heavy[i]);await page.evaluate(()=>(window as unknown as {corProbe:Probe}).corProbe.cancel());await status(page,'idle');
   assert.equal(await page.evaluate(()=>(window as unknown as {workerStats:{live:number}}).workerStats.live),0);
   await run(page,normal[i]);await ready(page,normal[i]);
   const before=await page.evaluate(()=>(window as unknown as {workerStats:{created:number}}).workerStats.created);
   await page.evaluate(()=>(window as unknown as {corProbe:Probe}).corProbe.retry());
   await page.waitForFunction(before=>(window as unknown as {workerStats:{created:number}}).workerStats.created>before&&(window as unknown as {corProbe:Probe}).corProbe.status==='ready',before);
   await ready(page,normal[i]);
  }
  await started(heavy[1]);await run(page,normal[2]);await ready(page,normal[2]);
  await started(heavy[0]);await page.evaluate(()=>(window as unknown as {corProbe:Probe}).corProbe.setEnabled(false));await status(page,'idle');
  await page.waitForFunction(()=>(window as unknown as {workerStats:{live:number}}).workerStats.live===0);
  assert.equal(await page.evaluate(()=>(window as unknown as {corProbe:Probe}).corProbe.resultKey),null);
  await run(page,normal[0]);await page.evaluate(()=>(window as unknown as {corProbe:Probe}).corProbe.setEnabled(true));await ready(page,normal[0]);
  await started(heavy[2]);await page.evaluate(()=>(window as unknown as {corProbe:Probe}).corProbe.run(null));await status(page,'idle');
  await page.waitForFunction(()=>(window as unknown as {workerStats:{live:number}}).workerStats.live===0);
  await run(page,normal[2]);await ready(page,normal[2]);
  failWorker=true;await page.evaluate(()=>(window as unknown as {corProbe:Probe}).corProbe.retry());await status(page,'error');assert.equal(await page.evaluate(()=>(window as unknown as {corProbe:Probe}).corProbe.resultKey),null);
  failWorker=false;await page.evaluate(()=>(window as unknown as {corProbe:Probe}).corProbe.retry());await ready(page,normal[2]);
  await started(heavy[1]);await page.evaluate(()=>(window as unknown as {corProbe:Probe}).corProbe.setMounted(false));await page.locator('[data-unmounted]').waitFor();
  assert.equal(await page.evaluate(()=>(window as unknown as {workerStats:{live:number}}).workerStats.live),0);
  await page.evaluate(input=>{const p=(window as unknown as {corProbe:Probe}).corProbe;p.run(input);p.setMounted(true);},normal[0]);await ready(page,normal[0]);
  const stats=await page.evaluate(()=>(window as unknown as {workerStats:{created:number;terminated:number;started:number;live:number}}).workerStats);assert.equal(stats.live,0);assert.equal(stats.created,stats.terminated);assert.deepEqual(errors,[]);
  const report={scope:'isolated native browser worker and actual React hook; not Next reader or Next worker asset',families:3,realStartedCancellationCases:3,retryCases:3,replacementVerified:true,disabledCancellationVerified:true,nullInputCancellationVerified:true,assetFailureRecovered:true,unmountCancellationVerified:true,strictModeRemountVerified:true,numericalModelsExcludedFromMainBundle:true,workers:stats,staleRenders:await page.evaluate(()=>(window as unknown as {corProbe:Probe}).corProbe.errors),errors};
  writeFileSync(`${destination}/browser-verification.json`,JSON.stringify(report,null,2)+'\n');writeFileSync(`${destination}/bundle-metafiles.json`,JSON.stringify({app:app.metafile,worker:worker.metafile},null,2)+'\n');console.log(JSON.stringify(report));
 }finally{if(browser)await browser.close();await new Promise<void>((resolve,reject)=>server.close(error=>error?reject(error):resolve()));}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
