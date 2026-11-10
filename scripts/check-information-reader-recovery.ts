import assert from 'node:assert/strict';
import {writeFileSync} from 'node:fs';
import {chromium,type Page} from 'playwright';
import {informationTeaching} from '../src/data/information-teaching';
import {learningUnits} from '../src/data/interactive-book';
const root='.runtime/learning/quantum-information-reader-integration',base=process.env.INFORMATION_PREVIEW_BASE??'http://127.0.0.1:4176';
const instrumentation=`globalThis.__name=fn=>fn;window.__informationMode='normal';window.__informationWorkers={created:0,terminated:0};const Native=Worker;window.Worker=class extends Native{constructor(...args){super(...args);this.closed=false;window.__informationWorkers.created++;}postMessage(frame){if(window.__informationMode==='hang')return;if(window.__informationMode==='fail')throw Error('QA postMessage failure');super.postMessage(frame);}terminate(){if(!this.closed){this.closed=true;window.__informationWorkers.terminated++;}super.terminate();}};`;
async function main(){const browser=await chromium.launch({headless:true}),states:unknown[]=[],errors:string[]=[];
 const phase=(page:Page,value:string)=>page.locator('[data-information-phase="'+value+'"]').waitFor({timeout:60000});
 const released=async(page:Page)=>{const c=await page.evaluate(()=>(window as unknown as {__informationWorkers:{created:number;terminated:number}}).__informationWorkers);assert.equal(c.created,c.terminated);return c;};
 const mode=(page:Page,value:string)=>page.evaluate(v=>{(window as unknown as {__informationMode:string}).__informationMode=v;},value);
 try{
  for(const item of informationTeaching){
   const context=await browser.newContext({viewport:{width:1440,height:1000},reducedMotion:'reduce'});await context.addInitScript({content:instrumentation});await context.addInitScript(()=>localStorage.setItem('suanlab-language','en'));
   const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));const route='/book/online/physics-math-for-ai/'+item.unit.id+'/';await page.goto(base+route);await phase(page,'ready');
   const prereqs=page.locator('.ai-prerequisite-links a');assert.equal(await prereqs.count(),item.unit.prerequisiteIds.length);
   for(const[index,id]of item.unit.prerequisiteIds.entries())assert.equal(await prereqs.nth(index).getAttribute('href'),'/book/online/physics-math-for-ai/'+id+'/');
   const position=learningUnits.findIndex(unit=>String(unit.id)===item.unit.id),nav=page.locator('.ai-reader-bottom a');
   assert.equal(await nav.first().getAttribute('href'),'/book/online/physics-math-for-ai/'+learningUnits[position-1].id+'/');
   assert.equal(await nav.last().getAttribute('href'),position===learningUnits.length-1?'/book/online/physics-math-for-ai/':'/book/online/physics-math-for-ai/'+learningUnits[position+1].id+'/');
   const examples=page.getByLabel('All examples',{exact:true}),values=await examples.locator('option').evaluateAll(options=>options.map(o=>(o as HTMLOptionElement).value).filter(v=>v!=='custom'));
   await mode(page,'hang');await examples.selectOption(values[1]);await phase(page,'pending');assert.equal(await page.locator('[data-information-result]').count(),0);await page.getByRole('button',{name:'Cancel calculation',exact:true}).click();await released(page);
   await mode(page,'normal');await page.getByRole('button',{name:'Recalculate',exact:true}).click();await phase(page,'ready');
   await mode(page,'fail');await examples.selectOption(values[0]);await phase(page,'error');assert.equal(await page.locator('[data-information-result]').count(),0);await released(page);
   await mode(page,'normal');await page.getByRole('button',{name:'Recalculate',exact:true}).click();await phase(page,'ready');
   await mode(page,'hang');await examples.selectOption(values[1]);await phase(page,'pending');await page.locator('.ai-reader-top a').first().click();await page.locator('.ai-book-overview').waitFor();assert.equal(await page.locator('[data-information-lab]').count(),0);const workers=await released(page);
   states.push({kind:item.unit.id,prerequisites:item.unit.prerequisiteIds.length,previousNextLinks:true,cancelAndRecalculate:true,postMessageFailureAndRetry:true,pendingUnmount:true,workers});await context.close();
   writeFileSync(root+'/reader-recovery-progress.json',JSON.stringify({states,errors},null,2)+'\n');console.log(JSON.stringify({cases:states.length,kind:item.unit.id}));
  }
  assert.equal(states.length,9);assert.deepEqual(errors,[]);writeFileSync(root+'/reader-recovery.json',JSON.stringify({states,errors,productionWorkerAssets:true,intentionalFaultInjection:true},null,2)+'\n');
 }finally{await browser.close();}}
main().catch(error=>{console.error(error);process.exitCode=1;});
