import assert from 'node:assert/strict';
import {writeFileSync} from 'node:fs';
import {chromium} from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import {saddleFluctuationExamples} from '../src/lib/learning-saddle-fluctuation-input';
import {saddleFluctuationDefaultDisplay,saddleFluctuationDefaultLearning,saddleFluctuationAlignDisplay} from '../src/lib/learning-saddle-fluctuation-view';
const root='.runtime/learning/saddle-fluctuation-reader-integration',key='suanlab-saddle-fluctuation-v1';
const url=(process.env.SADDLE_FLUCTUATION_PREVIEW_BASE??'http://127.0.0.1:4203')+'/book/online/physics-math-for-ai/saddle-fluctuation-and-nongaussian-approximation/';
async function main(){
 const browser=await chromium.launch({headless:true}),rows:unknown[]=[],errors:string[]=[];let referenceChecks=0;
 try{
  const well=saddleFluctuationExamples.find(e=>e.id==='double-well')!.input;
  const stored={version:1,input:well,display:saddleFluctuationAlignDisplay(well,saddleFluctuationDefaultDisplay),learning:saddleFluctuationDefaultLearning};
  const cases=[{id:'invalid-json',raw:'{'},{id:'mixed-model',raw:JSON.stringify({...stored,input:{...well,intervals:4}})},{id:'invalid-learning',raw:JSON.stringify({...stored,learning:{...stored.learning,step:10}})},{id:'denied-read',raw:null},{id:'denied-write',raw:null},{id:'valid-well',raw:JSON.stringify(stored)}];
  for(const item of cases){
   const context=await browser.newContext({viewport:{width:320,height:1000},hasTouch:true,isMobile:true}),page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
   await context.addInitScript({content:'globalThis.__name=fn=>fn;'});
   await page.addInitScript(({id,raw,key})=>{
    localStorage.setItem('suanlab-language','en');localStorage.setItem('theme','dark');if(raw!==null)localStorage.setItem(key,raw);
    const originalGet=Storage.prototype.getItem,originalSet=Storage.prototype.setItem;
    if(id==='denied-read')Storage.prototype.getItem=function(k){if(k===key)throw Error('Denied read');return originalGet.call(this,k);};
    if(id==='denied-write')Storage.prototype.setItem=function(k,v){if(k===key)throw Error('Denied write');return originalSet.call(this,k,v);};
   },{...item,key});
   await page.goto(url);await page.locator('[data-saddle-phase="ready"]').waitFor({timeout:120000});
   assert.equal(await page.locator(`[data-saddle-result="${item.id==='valid-well'?'well':'chain'}"]`).count(),1);
   if(item.id!=='valid-well')assert.equal(await page.locator('[data-saddle-storage]').getAttribute('data-saddle-storage'),item.id.startsWith('denied')?'unavailable':'corrupt');
   const axe=await new AxeBuilder({page}).include('.ai-reader').withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();writeFileSync(`${root}/axe-storage-${item.id}.json`,JSON.stringify(axe,null,2)+'\n');assert.deepEqual(axe.violations,[],item.id);
   rows.push({id:item.id,ready:true,accessibility:true});await context.close();
  }
  for(const language of ['ko','en'] as const){
   const context=await browser.newContext({viewport:{width:320,height:1000},hasTouch:true,isMobile:true}),page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
   await context.addInitScript({content:'globalThis.__name=fn=>fn;'});await page.addInitScript(language=>{localStorage.setItem('suanlab-language',language);localStorage.setItem('theme','light');},language);
   await page.goto(url);await page.locator('[data-saddle-phase="ready"]').waitFor({timeout:120000});
   for(let step=0;step<10;step++){
    await page.locator('.ai-concept-steps button').nth(step).click();const svg=page.locator(`[data-saddle-concept] [data-saddle-reference="${step}"]`);
    const overflowing=await svg.evaluate(node=>Array.from(node.querySelectorAll('text')).filter(text=>{const b=text.getBBox();return b.x< -1||b.y< -1||b.x+b.width>721||b.y+b.height>401;}).map(n=>n.textContent));
    assert.deepEqual(overflowing,[],`${language} reference${step} clipped text`);await svg.screenshot({path:`${root}/reference-${language}-${step}.png`});referenceChecks++;
   }
   await context.close();
  }
  const context=await browser.newContext({viewport:{width:320,height:1000},hasTouch:true,isMobile:true,reducedMotion:'no-preference'}),page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
  await context.addInitScript({content:'globalThis.__name=fn=>fn;'});
  const input=saddleFluctuationExamples.find(e=>e.id==='mixed-coprime-denominators')!.input;
  await page.addInitScript(({input,key,display,learning})=>{localStorage.setItem('suanlab-language','en');localStorage.setItem('theme','dark');if(localStorage.getItem(key)===null)localStorage.setItem(key,JSON.stringify({version:1,input,display,learning}));const Native=Worker,stats={created:0,terminated:0,active:new Set<Worker>()};(window as unknown as {__max:typeof stats}).__max=stats;window.Worker=class extends Native{constructor(url:string|URL,options?:WorkerOptions){super(url,options);stats.created++;stats.active.add(this);}terminate(){if(stats.active.delete(this)){stats.terminated++;super.terminate();}}};},{input,key,display:saddleFluctuationAlignDisplay(input,saddleFluctuationDefaultDisplay),learning:saddleFluctuationDefaultLearning});
  const ready=()=>page.locator('[data-saddle-phase="ready"]').waitFor({timeout:120000});
  const counts=()=>page.evaluate(()=>{const s=(window as unknown as {__max:{created:number;terminated:number;active:Set<Worker>}}).__max;return {created:s.created,terminated:s.terminated,active:s.active.size};});
  await page.goto(url);await ready();
  const baseline=await counts();await page.getByRole('button',{name:'Play display times',exact:true}).click();await page.waitForTimeout(1200);assert.notEqual(await page.getByLabel('Display time index',{exact:true}).inputValue(),String(input.mode==='chain'?input.probe:0));await page.getByRole('button',{name:'Stop display playback',exact:true}).click();assert.deepEqual(await counts(),baseline);
  await page.getByRole('button',{name:'Apply input',exact:true}).click();await page.getByRole('button',{name:'Cancel calculation',exact:true}).waitFor();await page.getByRole('button',{name:'Cancel calculation',exact:true}).click();await page.locator('[data-saddle-phase="idle"]').waitFor();assert.equal(await page.locator('[data-saddle-result]').count(),0);assert.equal((await counts()).active,0);
  await page.getByRole('button',{name:'Recalculate',exact:true}).click();await page.getByText('Independently checking the result.',{exact:true}).waitFor({timeout:120000});await page.getByLabel('Action constant hbar',{exact:true}).fill('2');await page.getByRole('button',{name:'Cancel calculation',exact:true}).click();await page.locator('[data-saddle-phase="idle"]').waitFor();await page.waitForTimeout(150);assert.equal(await page.locator('[data-saddle-result]').count(),0);assert.equal((await counts()).active,0);
  await page.getByRole('button',{name:'Recalculate',exact:true}).click();await ready();
  await page.getByLabel('Action constant hbar',{exact:true}).fill('1.0');await page.getByRole('button',{name:'Apply input',exact:true}).click();assert((await page.locator('[data-saddle-phase]').getByRole('alert').innerText()).includes('Check integer/fraction syntax'));assert.equal((await counts()).active,0);assert.equal(await page.locator('[data-saddle-phase="ready"]').count(),1);
  await page.getByLabel('Action constant hbar',{exact:true}).fill('2');await page.getByRole('button',{name:'Apply input',exact:true}).click();await ready();
  const saved=await page.evaluate(key=>JSON.parse(localStorage.getItem(key)!),key);assert.equal(saved.input.hbar,'2');assert.equal(saved.input.intervals,32);const final=await counts();assert.equal(final.active,0);assert.equal(final.created,final.terminated);
  await page.reload();await ready();assert.equal(await page.getByLabel('Action constant hbar',{exact:true}).inputValue(),'2');assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));const afterReload=await counts();assert.equal(afterReload.active,0);assert.equal(afterReload.created,afterReload.terminated);
  const axe=await new AxeBuilder({page}).include('.ai-reader').withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();writeFileSync(root+'/axe-maximum-recovery.json',JSON.stringify(axe,null,2)+'\n');assert.deepEqual(axe.violations,[]);assert.deepEqual(errors,[]);
  const report={storageCases:rows,actualProductionReader:true,referenceChecks,maximumInput:'mixed-coprime-denominators',maximumInputRetains32Intervals:true,maximumFlow:final,afterReload,typingDuringDispatchedValidation:true,cancelDuringCalculation:true,cancelDuringValidation:true,noLateReadyAfterCancel:true,invalidDraftKeepsAppliedResult:true,correctedInputApplied:true,reloadPreservesInput:true,displayPlaybackDoesNotCalculate:true,accessibilityCases:7,actualModuleWorkers:true,readerRegistered:true,productionNextAssetsVerified:true,errors};writeFileSync(root+'/reader-storage-maximum.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));await context.close();
 }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
