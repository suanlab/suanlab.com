/** Verify a real client-side chapter transition terminates active numerical work. */
import assert from 'node:assert/strict';import {writeFileSync} from 'node:fs';import {chromium} from 'playwright';
import {phaseKinds as kinds,phasePresetInput as preset} from '../src/lib/learning-phase-transitions';
import {phaseInputKey as key} from '../src/lib/learning-phase-transitions-protocol';
async function main(){const browser=await chromium.launch({headless:true,args:['--no-sandbox']}),errors:string[]=[];try{
 const context=await browser.newContext({viewport:{width:1440,height:900},reducedMotion:'reduce'});await context.addInitScript(kind=>{
  localStorage.setItem('suanlab-language','en');localStorage.setItem(`suanlab-phase-${kind}`,JSON.stringify({version:1,value:{kind,parameters:{spinCount:128,couplingK:2,fieldB:Number.MIN_VALUE},options:{minimumPrecision:8192,maximumPrecision:8192}}}));
  const Native=window.Worker,live=new Set<Worker>(),stats={created:0,terminated:0,heavyStarted:false,heavyTerminated:false};Object.assign(window,{phaseUnmountLive:live,phaseUnmountStats:stats});window.Worker=class extends Native{
   request:unknown=null;
   constructor(...args:ConstructorParameters<typeof Worker>){super(...args);stats.created++;live.add(this);this.addEventListener('message',e=>{const r=this.request as {input?:{parameters?:{spinCount?:number};options?:{minimumPrecision?:number}}};if(e.data?.phase==='started'&&r?.input?.parameters?.spinCount===128&&r.input.options?.minimumPrecision===8192)stats.heavyStarted=true;});}
   postMessage(value:unknown){this.request=value;super.postMessage(value);}
   terminate(){const r=this.request as {input?:{parameters?:{spinCount?:number};options?:{minimumPrecision?:number}}};if(live.delete(this)){stats.terminated++;if(r?.input?.parameters?.spinCount===128&&r.input.options?.minimumPrecision===8192)stats.heavyTerminated=true;}super.terminate();}
  };
 },kinds[0]);const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));await page.goto(`http://localhost:4175/book/online/physics-math-for-ai/${kinds[0]}/`);await page.waitForFunction(()=>(window as unknown as {phaseUnmountStats:{heavyStarted:boolean}}).phaseUnmountStats.heavyStarted,{},{timeout:60000});const origin=await page.evaluate(()=>performance.timeOrigin);
 await page.locator(`.ai-reader-bottom a[href="/book/online/physics-math-for-ai/${kinds[1]}/"]`).click();await page.waitForURL(`**/${kinds[1]}/`);assert.equal(await page.evaluate(()=>performance.timeOrigin),origin,'chapter change must be client-side');
 await page.waitForFunction(k=>document.querySelector('[data-phase-lab]')?.getAttribute('data-phase-current')===k,key(preset(kinds[1],0)),{timeout:60000});await page.locator('[data-phase-concept-state="ready"]').waitFor({timeout:60000});await page.waitForFunction(()=>(window as unknown as {phaseUnmountLive:Set<Worker>}).phaseUnmountLive.size===0);assert.equal(await page.locator(`[data-phase-result="${kinds[0]}"]`).count(),0);const stats=await page.evaluate(()=>(window as unknown as {phaseUnmountStats:{created:number;terminated:number;heavyTerminated:boolean}}).phaseUnmountStats);assert.ok(stats.heavyTerminated);assert.equal(stats.created,stats.terminated);assert.deepEqual(errors,[]);
 const report={scope:'actual Next client-side chapter transition',heavyStarted:true,heavyTerminated:stats.heavyTerminated,clientSideNavigation:true,staleResults:0,workers:stats,errors};writeFileSync('.runtime/learning/phase-transitions-integration/unmount-verification.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report));await context.close();
 }finally{await browser.close();}}
main().catch(error=>{console.error(error);process.exitCode=1;});
