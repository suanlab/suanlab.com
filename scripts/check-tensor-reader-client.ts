/** Verify client-side reader changes release production native Workers. */
import assert from 'node:assert/strict';
import {writeFileSync} from 'node:fs';
import {chromium} from 'playwright';
import {tensorKinds as kinds} from '../src/lib/learning-tensor';
async function main(){const browser=await chromium.launch({headless:true,executablePath:process.env.SUANLAB_BROWSER_PATH||'/usr/bin/google-chrome',args:['--no-sandbox','--enable-unsafe-swiftshader','--disable-gpu-rasterization']});try{
 const context=await browser.newContext({viewport:{width:1600,height:1000},reducedMotion:'reduce'});
 await context.addInitScript({content:`globalThis.__name=(fn)=>fn;localStorage.setItem('suanlab-language','en');const Native=Worker,s=window.tensorClientStats={created:0,terminated:0,urls:[],live:new Set()};window.Worker=class extends Native{ended=false;constructor(...args){super(...args);s.created++;s.urls.push(String(args[0]));s.live.add(this);}terminate(){if(!this.ended){this.ended=true;s.terminated++;s.live.delete(this);}super.terminate();}};`});
 const page=await context.newPage(),errors:string[]=[],book='/book/online/physics-math-for-ai/';page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://localhost:4175'+book+kinds[0]+'/');
 const ready=async(kind:string)=>{await page.locator(`[data-tensor-lab="${kind}"][data-tensor-state="ready"]`).waitFor();await page.locator(`[data-tensor-concept="${kind}"][data-tensor-concept-state="ready"]`).waitFor();};await ready(kinds[0]);
 let moves=0;for(const kind of [kinds[1],kinds[0],kinds[1]]){await page.locator(`main .ai-reader-bottom a[href="${book}${kind}/"]`).click();await page.waitForURL('**'+book+kind+'/');await ready(kind);if(kind===kinds[1]){const sphere=page.locator('[data-tensor-lab] [data-hilbert-bloch]').first();await sphere.scrollIntoViewIfNeeded();await page.waitForFunction(el=>el?.getAttribute('data-hilbert-bloch-status')==='3d',await sphere.elementHandle());assert(await page.locator('canvas').count()>0);}else{assert.equal(await page.locator('canvas').count(),0);}assert.equal(await page.locator(`[data-tensor-result="${kind}"]`).count(),2);moves++;}
 const assets=await page.evaluate(()=>[...new Set((window as any).tensorClientStats.urls)]as string[]);for(const asset of assets)await context.route(asset,async r=>{await new Promise(resolve=>setTimeout(resolve,750));await r.continue().catch(()=>{});});
 await page.locator('[data-tensor-lab] fieldset.ai-presets button').nth(13).click();await page.waitForFunction(()=>(window as any).tensorClientStats.live.size>0);const liveBeforeMove=await page.evaluate(()=>(window as any).tensorClientStats.live.size);assert(liveBeforeMove>0);await page.locator(`main .ai-reader-bottom a[href="${book}${kinds[0]}/"]`).evaluate((a:HTMLAnchorElement)=>a.click());await page.waitForURL('**'+book+kinds[0]+'/');await ready(kinds[0]);moves++;
 await page.locator('main .ai-reader-top a').first().click();await page.waitForURL('**'+book);await page.waitForFunction(()=>(window as any).tensorClientStats.live.size===0);
 assert.equal(await page.locator('canvas').count(),0);
 const stats=await page.evaluate(()=>{const s=(window as any).tensorClientStats;return{created:s.created,terminated:s.terminated,live:s.live.size};});assert(stats.created>0);assert.equal(stats.created,stats.terminated);assert.deepEqual(errors,[]);
 const report={clientReaderMoves:moves,pendingNativeWorkerMove:true,liveBeforeMove,...stats,errors};writeFileSync('.runtime/learning/quantum-tensor-integration/client-navigation.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));await context.close();
 }finally{await browser.close();}}
main().catch(e=>{console.error(e);process.exitCode=1;});
