/** Native browser transport and controller lifecycle, without registering readers. */
import assert from 'node:assert/strict';
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {build} from 'esbuild';
import {chromium} from 'playwright';
import {hilbertKinds,isHilbertSuccessForInput,type HilbertInput} from '../src/lib/learning-hilbert-protocol';
async function main(){
 const base=process.env.HILBERT_WORKER_BASE_URL??'http://localhost:4175',root='.runtime/learning/hilbert-worker';mkdirSync(root,{recursive:true});
 const workerBundle=await build({entryPoints:['src/lib/learning-hilbert.worker.ts'],bundle:true,write:false,format:'esm',platform:'browser',target:'es2020'});
 const controllerBundle=await build({entryPoints:['src/lib/learning-hilbert-controller.ts'],bundle:true,write:false,format:'esm',platform:'browser',target:'es2020',metafile:true});
 const graph=Object.keys(controllerBundle.metafile!.inputs);assert(!graph.some(p=>/learning-(quantum-state|wavefunction-l2|quantum-hilbert-core)\.ts|entropy-production-math|interval-transcendentals/.test(p)));writeFileSync(`${root}/browser-dependency-graph.json`,JSON.stringify(graph,null,2)+'\n');
 const inputs=hilbertKinds.flatMap((kind,i)=>JSON.parse(readFileSync(`scripts/fixtures/${['quantum-state','wavefunction-l2'][i]}-reference.json`,'utf8')).cases.map((c:{input:HilbertInput['parameters']})=>({kind,parameters:c.input}as HilbertInput)));
 const browser=await chromium.launch({headless:true}),report={nativeWorkerCases:0,terminatedAfterStarted:0,lifecycle:null as unknown,pageErrors:[]as string[],nextAssetsVerified:false,readerVerified:false};
 try{
  const page=await browser.newPage();await page.addInitScript({content:'globalThis.__name = (fn) => fn;'});page.on('pageerror',e=>report.pageErrors.push(e.message));
  await page.route('**/__hilbert-worker-verification.js',r=>r.fulfill({status:200,contentType:'text/javascript',body:workerBundle.outputFiles[0].text}));
  await page.route('**/__hilbert-controller-verification.js',r=>r.fulfill({status:200,contentType:'text/javascript',body:controllerBundle.outputFiles[0].text}));
  await page.route('**/__hilbert-broken-worker.js',r=>r.fulfill({status:200,contentType:'text/javascript',body:'throw new Error("intentional native Worker failure");'}));
  await page.goto(base+'/book/interactive/');
  for(let i=0;i<inputs.length;i++){
   const input=inputs[i],id=i+1;
   const messages=await page.evaluate(async({input,id})=>new Promise<unknown[]>((resolve,reject)=>{
    const worker=new Worker('/__hilbert-worker-verification.js',{type:'module'}),messages:unknown[]=[];const timer=setTimeout(()=>{worker.terminate();reject(Error('native worker timeout'));},20000);
    worker.onerror=e=>{clearTimeout(timer);worker.terminate();reject(Error(e.message));};worker.onmessage=e=>{messages.push(e.data);if(messages.length===2){clearTimeout(timer);worker.terminate();resolve(messages);}};worker.postMessage({version:1,requestId:id,input});
   }),{input,id});assert.deepEqual(messages[0],{version:1,requestId:id,phase:'started',kind:input.kind});assert(isHilbertSuccessForInput(messages[1],input,id),`native ${input.kind} case${i}`);report.nativeWorkerCases++;
  }
  const representatives=[inputs[0],inputs[24]];
  for(const input of representatives){
   await page.evaluate(async input=>new Promise<void>((resolve,reject)=>{
    const worker=new Worker('/__hilbert-worker-verification.js',{type:'module'}),timer=setTimeout(()=>{worker.terminate();reject(Error('native start timeout'));},15000);
    worker.onerror=e=>{clearTimeout(timer);worker.terminate();reject(Error(e.message));};worker.onmessage=e=>{clearTimeout(timer);worker.onmessage=null;worker.terminate();if(e.data.phase==='started')resolve();else reject(Error('missing start phase'));};worker.postMessage({version:1,requestId:1001,input});
   }),input);report.terminatedAfterStarted++;
  }
  report.lifecycle=await page.evaluate(async inputs=>{
   const controllerPath='/__hilbert-controller-verification.js';const {HilbertCalculationController:Controller}=await import(/* webpackIgnore: true */ controllerPath);
   let created=0,terminated=0;const live=new Set<Worker>(),states:string[]=[];let broken=false;
   const factory=()=>{const w=new Worker(broken?'/__hilbert-broken-worker.js':'/__hilbert-worker-verification.js',{type:'module'});created++;live.add(w);const terminate=w.terminate.bind(w);w.terminate=()=>{if(live.delete(w)){terminated++;terminate();}};return w;};
   let receive:(state:any)=>void=()=>{};const c=new Controller(factory,(state:any)=>{states.push(state.status);receive(state);});
   const wait=(run:()=>void,accept:(s:any)=>boolean)=>new Promise<any>((resolve,reject)=>{const timer=setTimeout(()=>{c.dispose();reject(Error('controller timeout'));},20000);receive=s=>{if(accept(s)){clearTimeout(timer);receive=()=>{};resolve(s);}};run();});
   for(let i=0;i<inputs.length;i++){
    await wait(()=>c.run(inputs[i]),s=>s.status==='pending'&&s.phase==='calculating');c.cancel();if(c.getSnapshot().status!=='idle'||live.size!==0)throw Error('cancel failed');
   }
   c.run(inputs[0]);const ready=await wait(()=>c.run(inputs[1]),s=>s.status==='ready');if(ready.input.kind!==inputs[1].kind||live.size!==0)throw Error('replacement result misattributed');
   broken=true;await wait(()=>c.run(inputs[0]),s=>s.status==='error');if(live.size!==0)throw Error('error leaked worker');
   broken=false;await wait(()=>c.run(inputs[1]),s=>s.status==='ready');
   c.run(inputs[0]);c.dispose();if(live.size!==0||created!==terminated)throw Error('dispose leaked worker');
   return{created,terminated,live:live.size,states,cancelCases:2,replacement:true,nativeFailureAndRetry:true,disposed:true};
  },representatives);
  assert.deepEqual(report.pageErrors,[]);writeFileSync(`${root}/native-worker.json`,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
 }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
