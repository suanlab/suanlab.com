import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {build} from 'esbuild';
import {chromium} from 'playwright';
import {realTimePhaseExamples} from '../src/lib/learning-real-time-phase-input';
const root='.runtime/learning/real-time-phase-worker';
type Harness=Window&{__fault:string;__state:{status:string;error?:string};__ports:Map<Worker,string>;__counts:{created:number;terminated:number};__run:(input:unknown,fault:string)=>void;__retry:()=>void;__dispose:()=>void};
async function main() {
 const bundle=await build({stdin:{contents:`import{RealTimePhaseCalculationController as Controller}from'./src/lib/learning-real-time-phase-controller';window.__run=(input,fault)=>{window.__fault=fault;const c=new Controller({calculate:()=>new Worker(window.__fault==='unavailable'?'http://[':window.__fault==='async-calc'?'throw.js':window.__fault==='message-calc'?'idle.js':'calculate.js',{type:'module',name:'calc'}),validate:()=>new Worker(window.__fault==='async-validate'?'throw.js':window.__fault==='message-validate'?'idle.js':'validate.js',{type:'module',name:'validate'})},s=>window.__state=s);window.__retry=()=>{window.__fault='normal';c.run(input);};window.__dispose=()=>c.dispose();c.run(input);};`,resolveDir:process.cwd(),loader:'ts'},bundle:true,write:false,platform:'browser',format:'esm',target:'es2020',metafile:true});
 assert(!Object.keys(bundle.metafile!.inputs).includes('src/lib/learning-real-time-phase-validation.ts'));
 const codes={calculate:readFileSync(root+'/calculate.js','utf8'),validate:readFileSync(root+'/validate.js','utf8')},browser=await chromium.launch({headless:true}),errors:string[]=[];
 try {
  const context=await browser.newContext();await context.addInitScript({content:'globalThis.__name=fn=>fn;'});const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/__real_time_phase_errors__/',r=>r.fulfill({contentType:'text/html',body:'<!doctype html><html lang="en"><title>Worker error recovery</title><body></body></html>'}));
  for(const kind of ['calculate','validate']as const)await page.route(`**/__real_time_phase_errors__/${kind}.js`,r=>r.fulfill({contentType:'text/javascript',body:codes[kind]}));
  await page.route('**/__real_time_phase_errors__/throw.js',r=>r.fulfill({contentType:'text/javascript',body:'throw new Error("Intentional native Worker failure");'}));
  await page.route('**/__real_time_phase_errors__/idle.js',r=>r.fulfill({contentType:'text/javascript',body:'self.onmessage=()=>{};'}));
  await page.goto('http://127.0.0.1:4175/__real_time_phase_errors__/');
  await page.evaluate(()=>{
   const w=window as unknown as Harness;w.__ports=new Map();w.__counts={created:0,terminated:0};const Native=Worker;
   window.Worker=class extends Native {
    channel:string;
    constructor(url:string|URL,options?:WorkerOptions){super(url,options);this.channel=options?.name??'';w.__ports.set(this,this.channel);w.__counts.created++;}
    postMessage(value:unknown,options?:StructuredSerializeOptions|Transferable[]){if(w.__fault==='clone-calc'&&this.channel==='calc'){super.postMessage(()=>null);return;}if(Array.isArray(options))super.postMessage(value,options);else super.postMessage(value,options);}
    terminate(){if(w.__ports.delete(this)){w.__counts.terminated++;super.terminate();}}
   };
  });
  await page.addScriptTag({type:'module',content:bundle.outputFiles[0].text});const rows=[];
  for(const fault of ['async-calc','async-validate','clone-calc','message-calc','message-validate','unavailable']) {
   console.log('Checking native error: '+fault);
   await page.evaluate(({input,fault})=>(window as unknown as Harness).__run(input,fault),{input:realTimePhaseExamples.find(x=>x.id==='negative-mode-hand')!.input,fault});
   if(fault.startsWith('message-')) {
    const channel=fault==='message-calc'?'calc':'validate';
    await page.waitForFunction(channel=>Array.from((window as unknown as Harness).__ports.values()).includes(channel),channel);
    await page.evaluate(channel=>{const w=window as unknown as Harness;const port=Array.from(w.__ports.entries()).find(([,name])=>name===channel)![0];port.dispatchEvent(new MessageEvent('messageerror'));},channel);
   }
   try{await page.waitForFunction(()=>(window as unknown as Harness).__state?.status==='error',undefined,{timeout:30000});}
   catch(error){writeFileSync(root+'/native-error-probe-failure.json',JSON.stringify(await page.evaluate(()=>{const w=window as unknown as Harness;return{fault:w.__fault,state:w.__state,ports:Array.from(w.__ports.values()),counts:w.__counts};}),null,2)+'\n');throw error;}
   const error=await page.evaluate(()=>(window as unknown as Harness).__state.error);
   assert.equal(error,fault==='unavailable'?'unavailable':fault.startsWith('message-')?'invalid-response':'worker-failed');
   assert.equal(await page.evaluate(()=>(window as unknown as Harness).__ports.size),0);
   await page.evaluate(()=>(window as unknown as Harness).__retry());await page.waitForFunction(()=>(window as unknown as Harness).__state.status==='ready',undefined,{timeout:30000});
   await page.evaluate(()=>(window as unknown as Harness).__dispose());assert.equal(await page.evaluate(()=>(window as unknown as Harness).__ports.size),0);rows.push({fault,error,retryReady:true});
  }
  const counts=await page.evaluate(()=>(window as unknown as Harness).__counts);assert.equal(counts.created,counts.terminated);assert.deepEqual(errors,[]);
  const report={rows,...counts,actualModuleWorkers:true,actualAsynchronousWorkerErrors:2,actualNativeCloneFailure:1,actualInvalidURLConstructorFailure:1,syntheticMessageErrorEventsOnNativePorts:2,allRetriesReady:true,errors};
  writeFileSync(root+'/native-errors.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
 }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
