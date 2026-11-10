/** Synthetic decode events on actual module Workers; no damaged-clone claim. */
import assert from 'node:assert/strict';
import { readFileSync, readdirSync, mkdirSync, writeFileSync } from 'node:fs';
import { build } from 'esbuild';
import { chromium } from 'playwright';
import inputs from './fixtures/worker-messageerror-inputs.json';
async function main(){
  const files=readdirSync('src/lib').filter(x=>/^learning-.*-controller\.ts$/.test(x));
  const cases=files.map((file,i)=>{
    const base=file.replace('-controller.ts',''),source=readFileSync('src/lib/'+file,'utf8'),protocol=readFileSync('src/lib/'+base+'-protocol.ts','utf8');
    return {base,alias:'Controller'+i,className:source.match(/export class (\w+)/)![1],dual:source.includes('this.factories.calculate()'),tag:protocol.match(/\bkind\s*:\s*'([^']+)'/)?.[1],input:(inputs as Record<string,unknown>)[base]};
  });
  const imports=cases.map(c=>`import{${c.className} as ${c.alias}}from'./src/lib/${c.base}-controller';`).join('\n');
  const entries=cases.map(c=>`{...${JSON.stringify(c)},Constructor:${c.alias}}`).join(',');
  const code=`${imports}
window.__audit=async()=>{
 const check=(ok,label)=>{if(!ok)throw Error(label);};
 let created=0,terminated=0;const rows=[];
 const nativePort=()=>{const port=new Worker('idle.js',{type:'module'});created++;const live=new Set();
   const add=port.addEventListener.bind(port),remove=port.removeEventListener.bind(port),terminate=port.terminate.bind(port);
   port.addEventListener=(type,callback,...args)=>{if(type==='messageerror')live.add(callback);add(type,callback,...args);};
   port.removeEventListener=(type,callback,...args)=>{if(type==='messageerror')live.delete(callback);remove(type,callback,...args);};
   port.terminate=()=>{check(live.size===0,'listener leaked at terminate');terminated++;terminate();};return port;};
 for(const spec of [${entries}]){
   const calc=[],validation=[];const factory=ports=>()=>{const p=nativePort();ports.push(p);return p;};
   const c=new spec.Constructor(spec.dual?{calculate:factory(calc),validate:factory(validation)}:factory(calc),()=>{}, {pause:async()=>{}});
   try{
     c.run(spec.input);check(c.getSnapshot().status==='pending',spec.base+' pending');
     calc[0].dispatchEvent(new MessageEvent('messageerror'));check(c.getSnapshot().error==='invalid-response',spec.base+' calculation error');
     c.run(spec.input);calc[0].dispatchEvent(new MessageEvent('messageerror'));check(c.getSnapshot().status==='pending',spec.base+' stale event');
     c.cancel();check(c.getSnapshot().status==='idle',spec.base+' cancel');
     if(spec.dual){const id=c.run(spec.input),s=c.getSnapshot();const response={version:1,requestId:id,ok:true,kind:s.input.kind??spec.tag,input:s.input,inputKey:s.inputKey,result:{input:s.input}};
       if(spec.base==='learning-qubit')response.observations={schemaVersion:1,coordinates:[],states:[]};
       calc[2].dispatchEvent(new MessageEvent('message',{data:response}));
       for(let i=0;i<100&&!validation.length&&c.getSnapshot().status==='pending';i++)await new Promise(r=>setTimeout(r,1));
       check(validation.length===1,spec.base+' validation port');validation[0].dispatchEvent(new MessageEvent('messageerror'));
       check(c.getSnapshot().error==='invalid-response',spec.base+' validation error');}
     rows.push({controller:spec.base,calculationError:true,retryAndCancel:true,validationError:spec.dual});
   }finally{c.dispose();}
 }
 check(created===terminated,'unreleased ports');return {rows,created,terminated,actualModuleWorkers:true,syntheticMessageErrorEvents:true,syntheticValidationHandoff:true,actualDecodeFailureInduced:false};};`;
  const bundle=await build({stdin:{contents:code,resolveDir:process.cwd(),loader:'js'},bundle:true,write:false,platform:'browser',format:'esm',target:'es2020',metafile:true});
  const browser=await chromium.launch({headless:true});const errors:string[]=[];
  try{
    const page=await browser.newPage();page.on('pageerror',e=>errors.push(e.message));await page.addInitScript('globalThis.__name=fn=>fn;');
    await page.route('**/__worker_event_audit__/',r=>r.fulfill({contentType:'text/html',body:'<!doctype html><title>Worker event audit</title>'}));
    await page.route('**/__worker_event_audit__/idle.js',r=>r.fulfill({contentType:'text/javascript',body:'self.onmessage=()=>{};'}));
    await page.goto('http://127.0.0.1:4204/__worker_event_audit__/');await page.addScriptTag({type:'module',content:bundle.outputFiles[0].text});
    await page.waitForFunction('typeof window.__audit === "function"');
    const result=await page.evaluate('window.__audit()') as {rows:unknown[];created:number;terminated:number};assert.equal(result.rows.length,46);assert.deepEqual(errors,[]);
    const root='.runtime/learning/worker-messageerror-compatibility';mkdirSync(root,{recursive:true});
    const report={...result,browserVersion:browser.version(),errors};writeFileSync(root+'/native-controllers.json',JSON.stringify(report,null,2)+'\n');
    console.log(JSON.stringify({controllers:result.rows.length,created:result.created,terminated:result.terminated,errors}));
  }finally{await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
