import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {writeFileSync} from 'node:fs';
import {thermalKinds} from '../src/lib/learning-thermal';
async function main(){
 const browser=await chromium.launch({headless:true}),base=process.env.THERMAL_READER_BASE_URL??'http://localhost:4175';
 const report={motionCases:0,focusPauses:0,offscreenPauses:0,globalPauses:0,reducedPauses:0,wrongFamilyRestoresRejected:0,workerFailureRetries:0};
 try{
 for(const kind of thermalKinds){
 const context=await browser.newContext({viewport:{width:1600,height:1000},reducedMotion:'no-preference'});await context.addInitScript(()=>{localStorage.setItem('suanlab-language','en');localStorage.setItem('suanlab-motion-paused','false');});const page=await context.newPage();
 const url=`${base}/book/online/physics-math-for-ai/${kind}/`;await page.goto(url);
 const lab=page.locator(`[data-thermal-lab="${kind}"]`),result=lab.locator('[data-thermal-result]'),play=lab.getByRole('button',{name:'Play stages',exact:true});
 const ready=()=>page.waitForFunction(kind=>document.querySelector(`[data-thermal-lab="${kind}"]`)?.getAttribute('data-thermal-state')==='ready',kind);await ready();
 await play.scrollIntoViewIfNeeded();await page.waitForFunction(kind=>{const lab=document.querySelector(`[data-thermal-lab="${kind}"]`);return [...lab!.querySelectorAll('button')].some(b=>b.textContent==='Play stages'&&!b.disabled);},kind);
 await play.click();await page.waitForFunction(kind=>document.querySelector(`[data-thermal-lab="${kind}"] [data-thermal-result]`)?.getAttribute('data-thermal-playing')==='true',kind);
 await lab.locator('[data-thermal-record]').focus();assert.equal(await result.getAttribute('data-thermal-playing'),'false');report.focusPauses++;
 await play.click();await page.evaluate(()=>scrollTo(0,0));await page.waitForFunction(kind=>document.querySelector(`[data-thermal-lab="${kind}"] [data-thermal-result]`)?.getAttribute('data-thermal-playing')==='false',kind);report.offscreenPauses++;
 await page.getByRole('button',{name:'Pause site motion',exact:true}).first().click();await page.waitForFunction(()=>document.documentElement.dataset.siteMotion==='paused');await play.scrollIntoViewIfNeeded();assert(await play.isDisabled());report.globalPauses++;
 await page.getByRole('button',{name:'Play site motion',exact:true}).first().click();await page.emulateMedia({reducedMotion:'reduce'});await play.scrollIntoViewIfNeeded();assert(await play.isDisabled());report.reducedPauses++;report.motionCases++;
 // Valid data for the other family must not switch this reader's model.
 const other=thermalKinds.find(k=>k!==kind)!;await page.goto(`${base}/book/online/physics-math-for-ai/${other}/`);await page.waitForFunction(kind=>document.querySelector(`[data-thermal-lab="${kind}"]`)?.getAttribute('data-thermal-state')==='ready',other);
 await page.evaluate(({kind,other})=>localStorage.setItem(`suanlab-thermal-${kind}`,localStorage.getItem(`suanlab-thermal-${other}`)!),{kind,other});await page.goto(url);await ready();assert.equal(JSON.parse((await lab.getAttribute('data-thermal-current'))!).kind,kind);report.wrongFamilyRestoresRejected++;
 // Abort only the dedicated Worker asset. The reader must offer retry and recover.
 const workerRequests:string[]=[];page.on('worker',worker=>workerRequests.push(worker.url()));
 await lab.locator('fieldset.ai-presets button').nth(1).click();await ready();assert(workerRequests.length>0);const target=workerRequests.at(-1)!;
 await page.route(target,route=>route.abort());await lab.locator('fieldset.ai-presets button').first().click();await page.waitForFunction(kind=>document.querySelector(`[data-thermal-lab="${kind}"]`)?.getAttribute('data-thermal-state')==='error',kind);
 await page.unroute(target);await lab.getByRole('button',{name:'Retry calculation',exact:true}).click();await ready();report.workerFailureRetries++;
 await context.close();console.log(`PASS lifecycle ${kind}`);
 }
 writeFileSync('.runtime/learning/thermal-interface/native-lifecycle.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
 }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
