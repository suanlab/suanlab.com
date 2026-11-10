import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {mkdirSync,readFileSync,writeFileSync} from 'node:fs';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {chromium} from 'playwright';
import {thermalFieldExamples} from '../src/lib/learning-thermal-field-input';
import {evaluateThermalFieldRequest} from '../src/lib/learning-thermal-field-evaluator';
const root='.runtime/learning/thermal-field-reader-integration',candidate=process.env.THERMAL_READER_CANDIDATE??'/tmp/suanlab-thermal-field-reader-candidate-20261111';mkdirSync(root,{recursive:true});
const stable=(v:unknown):string=>Array.isArray(v)?`[${v.map(stable).join(',')}]`:v&&typeof v==='object'?`{${Object.keys(v).sort().map(k=>`${JSON.stringify(k)}:${stable((v as Record<string,unknown>)[k])}`).join(',')}}`:JSON.stringify(v);
async function main(){
 const python=spawnSync('python3',['scripts/learning/verify-thermal-field-foundation.py'],{encoding:'utf8',timeout:120000,maxBuffer:8000000});assert.equal(python.status,0,python.stderr);
 const reference=JSON.parse(python.stdout) as {cases:{input:unknown;expected:Record<string,unknown>}[]},cases=[...reference.cases.map(x=>x.input),...thermalFieldExamples.map(x=>x.input)];
 const outputs=cases.map((input,i)=>{const response=evaluateThermalFieldRequest({version:1,requestId:i+101,input});if(!response.ok)assert.fail(JSON.stringify(response));return response;});
 reference.cases.forEach((row,i)=>{const exact=outputs[i].result.exact;for(const[k,v]of Object.entries(row.expected))assert.deepEqual(exact[k as keyof typeof exact],v,`${i}:${k}`);});
 const server=createServer((request,response)=>{try{const relative=decodeURIComponent(new URL(request.url!,'http://localhost').pathname);if(relative.includes('..'))throw Error('Path');const file=path.join(candidate,'out',relative.endsWith('/')?relative+'index.html':relative);const mime:Record<string,string>={'.js':'text/javascript','.css':'text/css','.woff2':'font/woff2','.svg':'image/svg+xml','.png':'image/png','.jpg':'image/jpeg'};response.setHeader('Content-Type',mime[path.extname(file)]??'text/html');response.end(readFileSync(file));}catch{response.statusCode=404;response.end('Not found');}});
 await new Promise<void>(r=>server.listen(0,'127.0.0.1',r));const address=server.address();assert(address&&typeof address!=='string');const base=`http://127.0.0.1:${address.port}`,browser=await chromium.launch({headless:true}),errors:string[]=[],assets=new Set<string>();
 try{
  const context=await browser.newContext();await context.addInitScript({content:"globalThis.__name=fn=>fn;globalThis.__factoryRecords=[];localStorage.setItem('suanlab-language','en');const Native=Worker;globalThis.Worker=class extends Native{constructor(url,options){super(url,options);__factoryRecords.push({url:String(url),options});}};"});
  context.on('request',request=>{const u=request.url();if(u.startsWith(base+'/_next/static/')&&u.endsWith('.js'))assets.add(new URL(u).pathname);});
  const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));await page.goto(base+'/book/online/physics-math-for-ai/finite-field-measure-and-thermal-free-energy/');await page.locator('[data-thermal-status="ready"]').waitFor({timeout:120000});
  const records=await page.evaluate(()=>((window as unknown as{__factoryRecords:{url:string;options?:WorkerOptions}[]}).__factoryRecords));assert.equal(records.length,2);assert(records[0].url.startsWith(base+'/_next/static/')&&records[1].url.startsWith(base+'/_next/static/'));
  const batch=await page.evaluate(async({cases,records})=>{
   const stable=(v:unknown):string=>Array.isArray(v)?`[${v.map(stable).join(',')}]`:v&&typeof v==='object'?`{${Object.keys(v).sort().map(k=>`${JSON.stringify(k)}:${stable((v as Record<string,unknown>)[k])}`).join(',')}}`:JSON.stringify(v);
   let created=0,closed=0,started=0;
   function request(index:number,value:unknown):Promise<unknown>{return new Promise((resolve,reject)=>{const spec=records[index],worker=new Worker(spec.url,spec.options);created++;const timer=setTimeout(()=>{worker.terminate();closed++;reject(Error('Production Worker timeout'));},30000);const finish=(result:unknown,error?:unknown)=>{clearTimeout(timer);worker.terminate();closed++;error?reject(error):resolve(result);};worker.onmessage=e=>{if(e.data.phase==='started'){started++;return;}finish(e.data);};worker.onerror=e=>{e.preventDefault();finish(null,Error(e.message));};worker.addEventListener('messageerror',()=>finish(null,Error('Decode error')));worker.postMessage(value);});}
   const rows=[];let first:unknown;
   for(const[i,input]of cases.entries()){
    const requestId=i+101,response=await request(0,{version:1,requestId,input}) as {ok:boolean;result:unknown};if(!response.ok)throw Error(JSON.stringify(response));if(i===0)first=response;
    const accepted=await request(1,{version:1,requestId,input,response}) as {phase:string;response:{result:unknown}};if(accepted.phase!=='validated')throw Error(JSON.stringify(accepted));
    if(stable(response.result)!==stable(accepted.response.result))throw Error('Gate changed the result');
    const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(stable(response.result))))).map(x=>x.toString(16).padStart(2,'0')).join('');rows.push({requestId,hash:digest});
   }
   for(const kind of ['energy','interval','attribution']){
    const response=structuredClone(first) as {requestId:number;result:{exact:{thermalEnergy:string};observations:{logPartition:{lower:string;upper:string}}}};
    if(kind==='energy')response.result.exact.thermalEnergy='999';if(kind==='interval')response.result.observations.logPartition={lower:'0',upper:'0'};if(kind==='attribution')response.requestId=102;
    const result=await request(1,{version:1,requestId:101,input:cases[0],response}) as {phase:string};if(result.phase!=='rejected')throw Error('Accepted production mutation '+kind);
   }
   return {rows,created,closed,started,rejectedMutations:3};
  },{cases,records});
  batch.rows.forEach((row,i)=>assert.equal(row.hash,createHash('sha256').update(stable(outputs[i].result)).digest('hex'),`production ${i}`));assert.equal(batch.created,batch.closed);assert.equal(batch.started,66);assert.deepEqual(errors,[]);
  const hashes=Object.fromEntries([...assets].map(relative=>[relative,createHash('sha256').update(readFileSync(path.join(candidate,'out',decodeURIComponent(relative)))).digest('hex')]));
  writeFileSync(root+'/production-workers.json',JSON.stringify({status:'actual-next-assets-verified',cases:cases.length,independentFractionModels:52,examples:14,records:records.map(r=>({...r,url:new URL(r.url).pathname})),batch,servedScriptHashes:hashes,errors},null,2)+'\n');console.log(JSON.stringify({productionRequests:cases.length,ports:batch.created,closed:batch.closed,rejectedMutations:3}));
 }finally{await browser.close();await new Promise<void>((resolve,reject)=>server.close(e=>e?reject(e):resolve()));}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
