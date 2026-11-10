/** Native Worker transport smoke test; does not register readers or test Next-built assets. */
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { build } from 'esbuild';
import { chromium } from 'playwright';
import { thermalKinds, isThermalSuccessForInput, type ThermalInput } from '../src/lib/learning-thermal-protocol';
async function main() {
 const base=process.env.THERMAL_WORKER_BASE_URL??'http://localhost:4175';
 const bundle=await build({entryPoints:['src/lib/learning-thermal.worker.ts'],bundle:true,write:false,format:'esm',platform:'browser',target:'es2020'});
 const inputs=thermalKinds.flatMap((kind,i)=>JSON.parse(readFileSync(`scripts/fixtures/${i===0?'driven':'reservoir'}-two-state-thermal-reference.json`,'utf8')).cases.map((c:{input:ThermalInput['parameters']})=>({kind,parameters:c.input} as ThermalInput)));
 const browser=await chromium.launch({headless:true});const report={nativeWorkerCases:0,terminatedAfterStarted:0,pageErrors:[] as string[],nextAssetsVerified:false,readerVerified:false};
 try {
 const page=await browser.newPage();page.on('pageerror',e=>report.pageErrors.push(e.message));
 await page.route('**/__thermal-worker-verification.js',route=>route.fulfill({status:200,contentType:'text/javascript',body:bundle.outputFiles[0].text}));
 await page.goto(base+'/book/interactive/');
 for(let i=0;i<inputs.length;i++){
 const input=inputs[i],id=i+1;
 const messages=await page.evaluate(async({input,id})=>new Promise<unknown[]>((resolve,reject)=>{
 const worker=new Worker('/__thermal-worker-verification.js',{type:'module'}),messages:unknown[]=[];
 const timer=setTimeout(()=>{worker.terminate();reject(Error('native worker timeout'));},15000);
 worker.onerror=e=>{clearTimeout(timer);worker.terminate();reject(Error(e.message));};
 worker.onmessage=e=>{messages.push(e.data);if(messages.length===2){clearTimeout(timer);worker.terminate();resolve(messages);}};
 worker.postMessage({version:1,requestId:id,input});
 }),{input,id});
 assert.deepEqual(messages[0],{version:1,requestId:id,phase:'started',kind:input.kind});assert(isThermalSuccessForInput(messages[1],input,id));report.nativeWorkerCases++;
 }
 for(const input of [inputs[0],inputs[49]]){
 await page.evaluate(async input=>new Promise<void>((resolve,reject)=>{
 const worker=new Worker('/__thermal-worker-verification.js',{type:'module'});const timer=setTimeout(()=>{worker.terminate();reject(Error('native start timeout'));},15000);
 worker.onerror=e=>{clearTimeout(timer);worker.terminate();reject(Error(e.message));};
 worker.onmessage=e=>{clearTimeout(timer);worker.onmessage=null;worker.terminate();if(e.data.phase==='started')resolve();else reject(Error('missing start phase'));};worker.postMessage({version:1,requestId:1001,input});
 }),input);report.terminatedAfterStarted++;
 }
 assert.deepEqual(report.pageErrors,[]);const dir='.runtime/learning/thermal-worker-integration';mkdirSync(dir,{recursive:true});writeFileSync(`${dir}/native-worker.json`,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
 }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
