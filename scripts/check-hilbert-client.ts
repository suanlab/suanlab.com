/** Verify client-side reader changes release source-built native Workers. */
import assert from 'node:assert/strict';
import {writeFileSync} from 'node:fs';
import {chromium} from 'playwright';
import {hilbertKinds as kinds} from '../src/lib/learning-hilbert';
async function main(){const browser=await chromium.launch({headless:true,args:['--no-sandbox','--enable-unsafe-swiftshader','--disable-gpu-rasterization']});try{
 const context=await browser.newContext({viewport:{width:1600,height:1000},reducedMotion:'reduce'});
 await context.addInitScript({content:`globalThis.__name=(fn)=>fn;localStorage.setItem('suanlab-language','en');const Native=Worker,s=window.hilbertClientStats={created:0,terminated:0,live:new Set()};window.Worker=class extends Native{ended=false;constructor(...args){super(...args);s.created++;s.live.add(this);}terminate(){if(!this.ended){this.ended=true;s.terminated++;s.live.delete(this);}super.terminate();}};`});
 const page=await context.newPage(),errors:string[]=[],book='/book/online/physics-math-for-ai/';page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://localhost:4175'+book+kinds[0]+'/');
 const ready=async(kind:string)=>{await page.locator(`[data-hilbert-lab="${kind}"][data-hilbert-state="ready"]`).waitFor();await page.locator(`[data-hilbert-concept="${kind}"][data-hilbert-concept-state="ready"]`).waitFor();};await ready(kinds[0]);const sphere=page.locator('[data-hilbert-lab] [data-hilbert-bloch]');await sphere.scrollIntoViewIfNeeded();await page.waitForFunction(el=>el?.getAttribute('data-hilbert-bloch-status')==='3d',await sphere.elementHandle());assert(await page.locator('canvas').count()>0);
 let moves=0;for(const kind of [kinds[1],kinds[0],kinds[1]]){await page.locator(`main .ai-reader-bottom a[href="${book}${kind}/"]`).click();await page.waitForURL('**'+book+kind+'/');await ready(kind);assert.equal(await page.locator(`[data-hilbert-result="${kind}"]`).count(),2);moves++;}
 await page.locator('main .ai-reader-top a').first().click();await page.waitForURL('**'+book);await page.waitForFunction(()=>(window as any).hilbertClientStats.live.size===0);
 assert.equal(await page.locator('canvas').count(),0);
 const stats=await page.evaluate(()=>{const s=(window as any).hilbertClientStats;return{created:s.created,terminated:s.terminated,live:s.live.size};});assert(stats.created>0);assert.equal(stats.created,stats.terminated);assert.deepEqual(errors,[]);
 const report={clientReaderMoves:moves,...stats,errors};writeFileSync('.runtime/learning/hilbert-integration/client-navigation.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));await context.close();
 }finally{await browser.close();}}
main().catch(e=>{console.error(e);process.exitCode=1;});
