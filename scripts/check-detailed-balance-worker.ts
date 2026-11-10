/** Native module worker + React hook QA in an isolated in-memory harness.
 * This is not an actual Next reader/bundle or a completed teaching UI.
 */
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {mkdirSync,writeFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {build} from 'esbuild';
import {chromium,type Page} from 'playwright';
import {detailedBalanceKinds as kinds,detailedBalanceInputKey,type DetailedBalanceInput,type DetailedBalanceSuccess} from '../src/lib/learning-detailed-balance-protocol';
const destination='.runtime/learning/detailed-balance-worker';
const normal:DetailedBalanceInput[]=[
 {kind:kinds[0],parameters:{forwardWeight:2,backwardWeight:1,stayWeight:1,initialWeights:[1,0,0],stepCount:12}},
 {kind:kinds[1],parameters:{targetWeights:[1,2,4],forwardWeight:2,backwardWeight:1,stayWeight:1}},
 {kind:kinds[2],parameters:{forwardNumerator:1,backwardNumerator:2,transitionDenominator:4,initialOneNumerator:0,initialDenominator:1,stepCount:12}},
];
const heavy:DetailedBalanceInput[]=[
 {kind:kinds[0],parameters:{forwardWeight:1000000,backwardWeight:999999,stayWeight:1000000,initialWeights:[1,0,0],stepCount:512}},
 {kind:kinds[1],parameters:{targetWeights:[1000000,999999,999983],forwardWeight:1000000,backwardWeight:999999,stayWeight:1}},
 {kind:kinds[2],parameters:{forwardNumerator:500000,backwardNumerator:500001,transitionDenominator:1000000,initialOneNumerator:0,initialDenominator:1,stepCount:512}},
];
const entry=`import React from 'react';
import {createRoot} from 'react-dom/client';
import {useDetailedBalanceCalculation} from './src/components/learning/use-detailed-balance-calculation';
import {detailedBalanceInputKey} from './src/lib/learning-detailed-balance-protocol';
const api=window.dbProbe={errors:[],status:'idle'};
function Probe({input,enabled}){
 const state=useDetailedBalanceCalculation(input,enabled);
 Object.assign(api,{status:state.status,phase:state.phase,error:state.error,resultKey:state.response?.inputKey??null,resultKind:state.response?.kind??null,resultRequestId:state.response?.requestId??null,result:state.response??null,retry:state.retry,cancel:state.cancel});
 if(state.response&&state.response.inputKey!==detailedBalanceInputKey(input))api.errors.push('stale render');
 return <output data-status={state.status}>{state.status}</output>;
}
function App(){
 const [input,setInput]=React.useState(${JSON.stringify(normal[0])}),[mounted,setMounted]=React.useState(true),[enabled,setEnabled]=React.useState(true);
 api.run=setInput;api.setMounted=setMounted;api.setEnabled=setEnabled;
 return mounted?<Probe input={input} enabled={enabled}/>:<output data-unmounted>unmounted</output>;
}
createRoot(document.getElementById('root')).render(<React.StrictMode><App/></React.StrictMode>);`;
interface Probe {status:string;phase:string|null;resultKey:string|null;resultKind:string|null;error:string|null;errors:string[];cancelAtStartedKey:string|null;resultRequestId:number|null;result:DetailedBalanceSuccess|null;run:(input:DetailedBalanceInput|null)=>void;retry:()=>void;cancel:()=>void;setMounted:(mounted:boolean)=>void;setEnabled:(enabled:boolean)=>void}
async function status(page:Page,value:string){await page.waitForFunction(value=>(window as unknown as {dbProbe:Probe}).dbProbe?.status===value,value);}
async function run(page:Page,input:DetailedBalanceInput){await page.evaluate(input=>(window as unknown as {dbProbe:Probe}).dbProbe.run(input),input);}
async function ready(page:Page,input:DetailedBalanceInput){
 await page.waitForFunction(expected=>{const p=(window as unknown as {dbProbe:Probe}).dbProbe;return p?.status==='ready'&&p.resultKey===expected;},detailedBalanceInputKey(input),{timeout:60000});
 const r=await page.evaluate(()=>{const p=(window as unknown as {dbProbe:Probe}).dbProbe;return {kind:p.resultKind,key:p.resultKey,error:p.error,errors:p.errors};});
 assert.equal(r.kind,input.kind);assert.equal(r.key,detailedBalanceInputKey(input));assert.equal(r.error,null);assert.deepEqual(r.errors,[]);
}
async function main(){
 mkdirSync(destination,{recursive:true});
 const [app,worker]=await Promise.all([
  build({stdin:{contents:entry,loader:'jsx',resolveDir:resolve('.'),sourcefile:'detailed-balance-probe.jsx'},bundle:true,write:false,format:'esm',platform:'browser',target:'es2022',metafile:true,logLevel:'silent'}),
  build({entryPoints:['src/lib/learning-detailed-balance.worker.ts'],bundle:true,write:false,format:'esm',platform:'browser',target:'es2022',metafile:true,logLevel:'silent'}),
 ]);
 const appInputs=Object.keys(app.metafile!.inputs),numerical=['learning-stationary-cycle.ts','learning-hastings-proposal.ts','learning-two-state-mixing.ts','learning-finite-markov-exact.ts','learning-binary-products.ts','learning-rational-gcd.ts'];
 for(const name of numerical)assert.ok(!appInputs.some(p=>p.endsWith(name)),name);
 let failWorker=false;
 const server=createServer((request,response)=>{
  response.setHeader('Cache-Control','no-store');
  if(request.url==='/app.js'){response.setHeader('Content-Type','text/javascript');response.end(app.outputFiles[0].text);}
  else if(request.url==='/learning-detailed-balance.worker.ts'){
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
   const stats={created:0,terminated:0,started:0,live:0,ackCancelled:0,maxResponseHandlingMs:0};(window as unknown as {workerStats:typeof stats}).workerStats=stats;
   const NativeWorker=window.Worker;
   window.Worker=class extends NativeWorker{
    private done=false;private inputKey='';
    constructor(url:string|URL,options?:WorkerOptions){super(url,options);stats.created++;stats.live++;this.addEventListener('message',e=>{
     if(e.data?.phase==='started'){
      stats.started++;const api=(window as unknown as {dbProbe:Probe}).dbProbe;
      if(api?.cancelAtStartedKey===this.inputKey){api.cancelAtStartedKey=null;queueMicrotask(()=>{api.cancel();stats.ackCancelled++;});}
     }else if(e.data?.ok===true){const start=performance.now();queueMicrotask(()=>{stats.maxResponseHandlingMs=Math.max(stats.maxResponseHandlingMs,performance.now()-start);});}
    });}
    postMessage(value:unknown,optionsOrTransfer:Transferable[]|StructuredSerializeOptions=[]){this.inputKey=JSON.stringify((value as {input?:unknown}).input);if(Array.isArray(optionsOrTransfer))super.postMessage(value,optionsOrTransfer);else super.postMessage(value,optionsOrTransfer);}
    terminate(){if(!this.done){this.done=true;stats.terminated++;stats.live--;}super.terminate();}
   };
  });
  await page.goto(`http://127.0.0.1:${address.port}/`);await ready(page,normal[0]);
  const started=async(input:DetailedBalanceInput)=>{await run(page,input);await page.waitForFunction(()=>(window as unknown as {dbProbe:Probe}).dbProbe.phase==='calculating');};
  for(let i=0;i<3;i++){
   if(i===1){
    const before=await page.evaluate(()=>(window as unknown as {workerStats:{ackCancelled:number}}).workerStats.ackCancelled);
    await page.evaluate(({input,expected})=>{const p=(window as unknown as {dbProbe:Probe}).dbProbe;p.cancelAtStartedKey=expected;p.run(input);},{input:heavy[i],expected:detailedBalanceInputKey(heavy[i])});
    await page.waitForFunction(before=>(window as unknown as {workerStats:{ackCancelled:number}}).workerStats.ackCancelled>before,before);await status(page,'idle');
   }else{await started(heavy[i]);await page.evaluate(()=>(window as unknown as {dbProbe:Probe}).dbProbe.cancel());await status(page,'idle');}
   assert.equal(await page.evaluate(()=>(window as unknown as {workerStats:{live:number}}).workerStats.live),0);
   await run(page,normal[i]);await ready(page,normal[i]);
   const priorRequest=await page.evaluate(()=>(window as unknown as {dbProbe:Probe}).dbProbe.resultRequestId);
   const before=await page.evaluate(()=>(window as unknown as {workerStats:{created:number}}).workerStats.created);
   await page.evaluate(()=>(window as unknown as {dbProbe:Probe}).dbProbe.retry());
   await page.waitForFunction(({before,priorRequest})=>(window as unknown as {workerStats:{created:number}}).workerStats.created>before&&(window as unknown as {dbProbe:Probe}).dbProbe.status==='ready'&&(window as unknown as {dbProbe:Probe}).dbProbe.resultRequestId!==priorRequest,{before,priorRequest});
   await ready(page,normal[i]);
  }
  await started(heavy[0]);await run(page,normal[2]);await ready(page,normal[2]);
  await started(heavy[0]);await page.evaluate(()=>(window as unknown as {dbProbe:Probe}).dbProbe.setEnabled(false));await status(page,'idle');
  await page.waitForFunction(()=>(window as unknown as {workerStats:{live:number}}).workerStats.live===0);
  assert.equal(await page.evaluate(()=>(window as unknown as {dbProbe:Probe}).dbProbe.resultKey),null);
  await run(page,normal[0]);await page.evaluate(()=>(window as unknown as {dbProbe:Probe}).dbProbe.setEnabled(true));await ready(page,normal[0]);
  await started(heavy[2]);await page.evaluate(()=>(window as unknown as {dbProbe:Probe}).dbProbe.run(null));await status(page,'idle');
  await page.waitForFunction(()=>(window as unknown as {workerStats:{live:number}}).workerStats.live===0);
  await run(page,normal[2]);await ready(page,normal[2]);
  failWorker=true;await page.evaluate(()=>(window as unknown as {dbProbe:Probe}).dbProbe.retry());await status(page,'error');assert.equal(await page.evaluate(()=>(window as unknown as {dbProbe:Probe}).dbProbe.resultKey),null);
  failWorker=false;await page.evaluate(()=>(window as unknown as {dbProbe:Probe}).dbProbe.retry());await ready(page,normal[2]);
  await started(heavy[0]);await page.evaluate(()=>(window as unknown as {dbProbe:Probe}).dbProbe.setMounted(false));await page.locator('[data-unmounted]').waitFor();
  assert.equal(await page.evaluate(()=>(window as unknown as {workerStats:{live:number}}).workerStats.live),0);
  await page.evaluate(input=>{const p=(window as unknown as {dbProbe:Probe}).dbProbe;p.run(input);p.setMounted(true);},normal[0]);await ready(page,normal[0]);
  for(const input of [heavy[0],heavy[2]]){await run(page,input);await ready(page,input);}
  const checks:DetailedBalanceInput[]=[
   {kind:kinds[1],parameters:{targetWeights:[1,2,4],forwardWeight:1,backwardWeight:0,stayWeight:0}},
   {kind:kinds[2],parameters:{forwardNumerator:0,backwardNumerator:0,transitionDenominator:1,initialOneNumerator:1,initialDenominator:3,stepCount:3}},
   {kind:kinds[2],parameters:{forwardNumerator:1,backwardNumerator:1,transitionDenominator:1,initialOneNumerator:0,initialDenominator:1,stepCount:3}},
   {kind:kinds[2],parameters:{forwardNumerator:0,backwardNumerator:1,transitionDenominator:1,initialOneNumerator:1,initialDenominator:1,stepCount:3}},
  ];
  const semanticChecks=[];
  for(const input of checks){
   await run(page,input);await ready(page,input);
   const response=await page.evaluate(()=>(window as unknown as {dbProbe:Probe}).dbProbe.result);assert.ok(response);
   if(response.kind===kinds[1]){const result=response.result;assert.equal(result.corrected.acceptance[0][2],null);assert.equal(result.corrected.acceptance[0][1]!.resolution,'structural-zero');assert.equal(result.corrected.stationaryDistribution,null);}
   else{assert.ok(response.kind===kinds[2]);const result=response.result,p=response.input.parameters;
    if(p.forwardNumerator===0&&p.backwardNumerator===0){assert.equal(result.stationaryDistribution,null);assert.equal(result.current.totalVariationFromStationary,null);assert.ok(result.limitDistribution);}
    else if(p.forwardNumerator===1){assert.equal(result.mixingAtQuarter.status,'never-periodic');assert.equal(result.limitDistribution,null);assert.equal(result.cesaroAverage[1].value,.5);}
    else{assert.equal(result.stationaryTimeReversal![1],null);assert.equal(result.stationaryLagOneCorrelation,null);assert.equal(result.stationaryIndicatorVariance!.resolution,'structural-zero');}
   }
   semanticChecks.push({kind:input.kind,parameters:input.parameters,verified:true});
  }
  const stats=await page.evaluate(()=>(window as unknown as {workerStats:{created:number;terminated:number;started:number;live:number;ackCancelled:number;maxResponseHandlingMs:number}}).workerStats);assert.equal(stats.live,0);assert.equal(stats.created,stats.terminated);assert.deepEqual(errors,[]);
  const report={scope:'isolated native browser worker and actual React hook; not production Next reader/asset. Short Hastings cancellation is ACK-linked; CPU interruption timing is not claimed for it',families:3,realStartedCancellationCases:3,visibleCalculatingCancellationFamilies:[kinds[0],kinds[2]],startedAckLinkedCancellationFamilies:[kinds[1]],largeResultCases:2,semanticChecks,retryCases:3,replacementVerified:true,disabledCancellationVerified:true,nullInputCancellationVerified:true,assetFailureRecovered:true,unmountCancellationVerified:true,strictModeRemountVerified:true,numericalModelsExcludedFromMainBundle:true,workers:stats,staleRenders:await page.evaluate(()=>(window as unknown as {dbProbe:Probe}).dbProbe.errors),errors};
  writeFileSync(`${destination}/browser-verification.json`,JSON.stringify(report,null,2)+'\n');writeFileSync(`${destination}/bundle-metafiles.json`,JSON.stringify({app:app.metafile,worker:worker.metafile},null,2)+'\n');console.log(JSON.stringify(report));
 }finally{if(browser)await browser.close();await new Promise<void>((resolve,reject)=>server.close(error=>error?reject(error):resolve()));}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
