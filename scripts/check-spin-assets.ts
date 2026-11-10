/** Verify the registered reader's generated Worker against the validated engine. */
import assert from 'node:assert/strict';
import {readFileSync,writeFileSync,readdirSync} from 'node:fs';
import {chromium} from 'playwright';
import {spinKinds,isSpinSuccessForInput,type SpinInput} from '../src/lib/learning-spin-protocol';
import {evaluateSpinRequest} from '../src/lib/learning-spin-evaluator';
import {isMasterObservation} from '../src/lib/learning-master-equation-protocol';
import type {PartitionObservation as Q} from '../src/lib/learning-partition-exact-core';
let mathematicalObservations=0;
function compareNative(actual:unknown,expected:unknown,path='response'){
 if(isMasterObservation(expected)){
  assert(isMasterObservation(actual),path);const a=actual as Q,b=expected as Q;
  for(const key of ['bounds','sign','value','resolution','logResolution']as const)assert.deepEqual(a[key],b[key],path+'.'+key);
  if(typeof b.logMagnitude==='number'){assert(typeof a.logMagnitude==='number');assert(Math.abs(a.logMagnitude-b.logMagnitude)<=1e-12*Math.max(1,Math.abs(b.logMagnitude)),path+'.logMagnitude');}else assert.equal(a.logMagnitude,b.logMagnitude);
  mathematicalObservations++;return;
 }
 if(Array.isArray(expected)){assert(Array.isArray(actual),path);assert.equal(actual.length,expected.length,path);expected.forEach((x,i)=>compareNative(actual[i],x,path+'.'+i));return;}
 if(expected&&typeof expected==='object'){assert(actual&&typeof actual==='object',path);assert.deepEqual(Object.keys(actual).sort(),Object.keys(expected).sort(),path);for(const[k,v]of Object.entries(expected))compareNative((actual as Record<string,unknown>)[k],v,path+'.'+k);return;}
 assert.equal(actual,expected,path);
}
async function main(){
 const root='.runtime/learning/quantum-spin-integration',source=JSON.parse(readFileSync(root+'/native-reader.json','utf8')),assets=source.workerURLs as string[];assert.equal(assets.length,1);assert(assets[0].includes('/_next/static/'));
 const factoryFiles=readdirSync('out/_next/static/chunks').filter(f=>f.endsWith('.js')).filter(f=>readFileSync('out/_next/static/chunks/'+f,'utf8').includes('name:"spin-calculation"'));assert.equal(factoryFiles.length,1);assert(readFileSync('out/_next/static/chunks/'+factoryFiles[0],'utf8').includes('type:void 0,name:"spin-calculation"'));
 const references=['half','ladder'].map(f=>JSON.parse(readFileSync(`scripts/fixtures/spin-${f}-reference.json`,'utf8')).cases);
 const inputs=spinKinds.flatMap((kind,i)=>references[i].map((c:{input:SpinInput['parameters']})=>({kind,parameters:c.input} as SpinInput)));
 const browser=await chromium.launch({headless:true,executablePath:process.env.SUANLAB_BROWSER_PATH||'/usr/bin/google-chrome',args:['--no-sandbox']}),errors:string[]=[];try{
 const page=await browser.newPage();await page.addInitScript({content:'globalThis.__name=(fn)=>fn;'});page.on('pageerror',e=>errors.push(e.message));await page.goto('http://localhost:4175/book/interactive/');
 for(const [index,input] of inputs.entries()){
 const id=index+1,messages=await page.evaluate(async({input,id,asset})=>new Promise<unknown[]>((resolve,reject)=>{const worker=new Worker(asset,{name:'spin-calculation'}),messages:unknown[]=[];const timer=setTimeout(()=>{worker.terminate();reject(Error('production worker timeout'));},20000);worker.onerror=e=>{clearTimeout(timer);worker.terminate();reject(Error(e.message));};worker.onmessage=e=>{messages.push(e.data);if(messages.length===2){clearTimeout(timer);worker.terminate();resolve(messages);}};worker.postMessage({version:1,requestId:id,input});}),{input,id,asset:assets[0]});
 assert.deepEqual(messages[0],{version:1,requestId:id,phase:'started',kind:input.kind});assert(isSpinSuccessForInput(messages[1],input,id));compareNative(messages[1],evaluateSpinRequest({version:1,requestId:id,input}));
 }
 assert.deepEqual(errors,[]);const report={productionWorkerAssets:assets,compiledFactoryFile:factoryFiles[0],workerModeMatchedToCompiledFactory:'classic',nativePreparations:inputs.length,mathematicalObservations,allPortsTerminated:true,errors};writeFileSync(root+'/production-worker.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
 }finally{await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
