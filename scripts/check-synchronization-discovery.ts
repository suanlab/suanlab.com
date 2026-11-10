import {chromium} from 'playwright';
import assert from 'node:assert/strict';
import {writeFileSync} from 'node:fs';
import {synchronizationKinds as kinds} from '../src/lib/learning-synchronization';
async function main(){const browser=await chromium.launch({headless:true}),report={languages:0,newCards:0,libraryEntries:0,readerLinks:0,onlineBookExcluded:0,errors:[]as string[]};try{
 for(const language of ['ko','en']){const c=await browser.newContext({viewport:{width:1600,height:1000}});await c.addInitScript(language=>{localStorage.setItem('suanlab-language',language);localStorage.setItem('suanlab-motion-paused','true');},language);const p=await c.newPage();p.on('pageerror',e=>report.errors.push(e.message));await p.goto('http://localhost:4175/book/interactive/');await p.locator('main a[href="/book/online/physics-math-for-ai/"]').click();await p.waitForURL('**/book/online/physics-math-for-ai/');report.libraryEntries++;
 for(const kind of kinds){const link=p.locator(`main a.ai-unit-card[href="/book/online/physics-math-for-ai/${kind}/"]`);await link.waitFor({state:'attached'});assert.equal(await link.locator(`[data-synchronization-cover="${kind}"]`).count(),1);report.newCards++;const target=await link.getAttribute('href');assert.equal((await p.request.get('http://localhost:4175'+target)).status(),200);report.readerLinks++;}
 await p.goto('http://localhost:4175/book/online/');assert.equal(await p.locator('main a[href*="physics-math-for-ai"]').count(),0);assert(await p.locator('main a[href^="/book/online/"]').count());report.onlineBookExcluded++;report.languages++;await c.close();
 }assert.deepEqual(report.errors,[]);writeFileSync('.runtime/learning/synchronization-interface/discovery.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));}finally{await browser.close();}}
main().catch(e=>{console.error(e);process.exitCode=1;});
