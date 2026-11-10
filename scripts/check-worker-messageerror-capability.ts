/** Read-only browser capability probe. Synthetic events test registration,
 * not a deliberately corrupted structured-clone payload.
 */
import assert from 'node:assert/strict';
import {mkdirSync,writeFileSync} from 'node:fs';
import {chromium} from 'playwright';
async function main() {
 const browser=await chromium.launch({headless:true});
 try {
  const page=await browser.newPage();await page.evaluate('globalThis.__name=fn=>fn;');
  const result=await page.evaluate(()=>{
   const url=URL.createObjectURL(new Blob(['self.onmessage=()=>{};'],{type:'text/javascript'})),worker=new Worker(url,{type:'module'});
   try {
    const idlPropertyPresent='onmessageerror'in worker;let propertyEvents=0,listenerEvents=0;
    worker.onmessageerror=()=>propertyEvents++;const listener=()=>listenerEvents++;
    worker.addEventListener('messageerror',listener);worker.dispatchEvent(new MessageEvent('messageerror'));
    const afterFirst={propertyEvents,listenerEvents};worker.removeEventListener('messageerror',listener);worker.dispatchEvent(new MessageEvent('messageerror'));
    return{idlPropertyPresent,afterFirst,afterDetach:{propertyEvents,listenerEvents}};
   }finally{worker.terminate();URL.revokeObjectURL(url);}
  });
  assert.equal(result.afterFirst.listenerEvents,1);assert.equal(result.afterDetach.listenerEvents,1);
  const report={...result,browserVersion:browser.version(),actualModuleWorker:true,syntheticRegistrationEvents:true,actualDecodeFailureInduced:false};
  mkdirSync('.runtime/learning/worker-eventtarget-capability',{recursive:true});writeFileSync('.runtime/learning/worker-eventtarget-capability/probe.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
 }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
