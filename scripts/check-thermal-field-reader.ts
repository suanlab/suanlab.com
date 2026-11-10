/** Static Next export QA. All responses,including Worker chunks,come from the candidate out. */
import assert from 'node:assert/strict';
import {createServer} from 'node:http';
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import path from 'node:path';
import {chromium} from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import {thermalFieldReader} from '../src/data/thermal-field-reader';
import {thermalFieldTeachingId as id} from '../src/lib/learning-thermal-field-teaching';
import {emptyThermalState,thermalFieldStorageKey} from '../src/lib/learning-thermal-field-state';
const root='.runtime/learning/thermal-field-reader-integration';mkdirSync(root,{recursive:true});
const candidate=process.env.THERMAL_READER_CANDIDATE??'/tmp/suanlab-thermal-field-reader-candidate-20261111';
async function main(){
 const server=createServer((request,response)=>{try{const relative=decodeURIComponent(new URL(request.url!,'http://localhost').pathname);if(relative.includes('..'))throw Error('Path');const full=path.join(candidate,'out',relative.endsWith('/')?relative+'index.html':relative),mime:Record<string,string>={'.js':'text/javascript','.css':'text/css','.woff2':'font/woff2','.svg':'image/svg+xml','.png':'image/png','.jpg':'image/jpeg','.jpeg':'image/jpeg','.webp':'image/webp','.ico':'image/x-icon','.txt':'text/plain'};response.setHeader('Content-Type',mime[path.extname(full)]??'text/html');response.end(readFileSync(full));}catch{response.statusCode=404;response.end('Not found');}});
 await new Promise<void>(r=>server.listen(0,'127.0.0.1',r));const address=server.address();assert(address&&typeof address!=='string');const base=`http://127.0.0.1:${address.port}`,url=base+'/book/online/physics-math-for-ai/'+id+'/';
 const browser=await chromium.launch({headless:true,args:['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader']}),errors:string[]=[],rows=[];
 try{
  for(const language of ['ko','en'] as const)for(const theme of ['light','dark'])for(const width of [320,1440]){
   const context=await browser.newContext({viewport:{width,height:950},reducedMotion:'reduce'});
   await context.addInitScript({content:`globalThis.__name=fn=>fn;localStorage.setItem('suanlab-language',${JSON.stringify(language)});localStorage.setItem('theme',${JSON.stringify(theme)});const Native=Worker,live=new Set();globalThis.__ports={created:0,closed:0,active:0,urls:[]};globalThis.Worker=class extends Native{constructor(url,options){super(url,options);live.add(this);__ports.created++;__ports.active=live.size;__ports.urls.push(String(url));}terminate(){if(live.delete(this)){__ports.closed++;__ports.active=live.size;}super.terminate();}};`});
   const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
   const ready=async()=>{await page.locator('[data-thermal-status="ready"]').waitFor({timeout:120000});await page.waitForFunction(()=>((window as unknown as{__ports:{active:number}}).__ports.active===0));};
   const t=(ko:string,en:string)=>language==='ko'?ko:en;
   assert.equal((await page.goto(url))!.status(),200);await ready();await page.waitForFunction(l=>document.querySelector('.ai-reader')?.getAttribute('lang')===l,language);
   assert.equal(await page.locator('h1').innerText(),thermalFieldReader.unit.title[language]);assert.equal(await page.locator('.ai-prose h2').count(),21);assert.equal(await page.locator('.ai-question').count(),8);assert.equal(await page.locator('fieldset.information-question').count(),0);
   for(const[i,step]of thermalFieldReader.guide.steps.entries()){await page.locator('[data-thermal-concept] .ai-concept-steps button').nth(i).click();assert.equal(await page.locator('[data-thermal-concept] .information-step h3').innerText(),step.title[language]);}
   for(const[i,q]of thermalFieldReader.unit.questions.entries()){const field=page.locator('.ai-question').nth(i),radios=field.locator('input[type=radio]');await radios.nth((q.answer+1)%3).check();await field.getByRole('button').click();assert((await field.locator('.ai-answer strong').innerText()).startsWith(t('다시 생각해보세요','Try again.')));await radios.nth(q.answer).check();await field.getByRole('button').click();assert((await field.locator('.ai-answer').innerText()).includes(q.explanation[language]));}
   for(const[i,g]of thermalFieldReader.guide.experiments.entries()){await page.getByLabel(t('안내 실험','Guided experiment'),{exact:true}).selectOption(String(i));await page.getByRole('button',{name:t('해설 보기','Show explanation'),exact:true}).click();assert.equal(await page.locator('[data-thermal-guide-explanation]').innerText(),g.explain[language]);}
   const lab=page.locator('[data-thermal-status]');
   await lab.getByLabel(t('전체 예제','All examples'),{exact:true}).selectOption('two-dimensional');await ready();
   await lab.getByLabel(t('관찰할 행렬','Matrix to inspect'),{exact:true}).selectOption('precision');await lab.getByLabel(t('그림의 척도','Plot scale'),{exact:true}).selectOption('signed-log');await lab.locator('[data-thermal-cell="0-0"]').focus();await page.keyboard.press('ArrowRight');
   await lab.getByLabel(t('온도 T','Temperature T'),{exact:true}).fill('1e-');
   await page.locator('.ai-completion input').check();
   await page.waitForFunction(k=>{const raw=localStorage.getItem(k);return raw&&JSON.parse(raw).model.draft.temperature==='1e-';},thermalFieldStorageKey);
   await page.reload();await ready();await page.waitForFunction(l=>document.querySelector('.ai-reader')?.getAttribute('lang')===l,language);
   assert.equal(await lab.getByLabel(t('온도 T','Temperature T'),{exact:true}).inputValue(),'1e-');assert.equal(await lab.getByLabel(t('관찰할 행렬','Matrix to inspect'),{exact:true}).inputValue(),'precision');assert.equal(await lab.getByLabel(t('그림의 척도','Plot scale'),{exact:true}).inputValue(),'signed-log');assert((await lab.locator('[data-thermal-selected]').innerText()).includes('[0,1]'));assert(await page.locator('.ai-completion input').isChecked());
   for(const[i,q]of thermalFieldReader.unit.questions.entries())assert(await page.locator('.ai-question').nth(i).locator('input[type=radio]').nth(q.answer).isChecked());
   assert.equal(await page.locator('[data-thermal-concept]').getAttribute('data-thermal-concept'),'5');
   if(language==='en')assert(!/[가-힣]/.test(await page.locator('.ai-reader').innerText()));assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
   const axe=await new AxeBuilder({page}).include('.ai-reader').analyze();assert.deepEqual(axe.violations.map(v=>({id:v.id,nodes:v.nodes.map(n=>n.target)})),[]);
   if(language==='en'&&theme==='light'&&width===1440)await page.locator('[data-thermal-concept]').screenshot({path:root+'/reader-concept-en.png'});
   const ports=await page.evaluate(()=>((window as unknown as{__ports:{created:number;closed:number;active:number;urls:string[]}}).__ports));assert.equal(ports.active,0);assert.equal(ports.created,ports.closed);assert(ports.urls.every(u=>u.startsWith(base+'/_next/static/')&&u.endsWith('.js')),'Nonproduction Worker URL');
   rows.push({language,theme,width,restoredDraft:true,restoredSelection:true,restoredStudy:true,axeViolations:0,ports,portCounterScope:'final document after reload'});await context.close();
  }
  for(const variant of ['corrupt','denied']){
   const context=await browser.newContext({viewport:{width:320,height:950},reducedMotion:'reduce'});
   await context.addInitScript({content:variant==='corrupt'?`localStorage.setItem(${JSON.stringify(thermalFieldStorageKey)},'{');`:`Object.defineProperty(Storage.prototype,'getItem',{value(){throw Error('blocked')}});Object.defineProperty(Storage.prototype,'setItem',{value(){throw Error('blocked')}});`});
   const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));await page.goto(url);await page.locator('[data-thermal-status="ready"]').waitFor({timeout:120000});assert.equal(await page.locator('[data-thermal-storage]').getAttribute('data-thermal-storage'),variant==='corrupt'?'corrupt':'unavailable');await context.close();
  }
  const initialized=structuredClone(emptyThermalState({steps:6,guides:6,questions:8,options:3}));initialized.input={...initialized.input,temperature:'3'};initialized.draft={...initialized.draft,temperature:'pending'};initialized.display={kind:'precision',scale:'signed-log',selected:3,flat:false};
  const context=await browser.newContext({reducedMotion:'reduce'});await context.addInitScript({content:`globalThis.__thermalWrites=[];globalThis.__thermalRequests=[];const NativeWorker=Worker;globalThis.Worker=class extends NativeWorker{postMessage(value){if(value?.input?.temperature)__thermalRequests.push(value.input.temperature);super.postMessage(value)}};const set=Storage.prototype.setItem;set.call(localStorage,'suanlab-language','en');set.call(localStorage,${JSON.stringify(thermalFieldStorageKey)},${JSON.stringify(JSON.stringify({version:1,model:initialized}))});Storage.prototype.setItem=function(k,v){if(k===${JSON.stringify(thermalFieldStorageKey)})__thermalWrites.push(JSON.parse(v).model.input.temperature);return set.call(this,k,v)};`});const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));await page.goto(url);await page.locator('[data-thermal-status="ready"]').waitFor({timeout:120000});assert.equal(await page.getByLabel('Temperature T',{exact:true}).inputValue(),'pending');assert((await page.locator('[data-thermal-selected]').innerText()).includes('[1,1]'));assert.equal((await page.evaluate(()=>((window as unknown as{__thermalWrites:string[]}).__thermalWrites)))[0],'3');assert.equal((await page.evaluate(()=>((window as unknown as{__thermalRequests:string[]}).__thermalRequests)))[0],'3');await context.close();
  const life=await browser.newContext({reducedMotion:'reduce'});
  await life.addInitScript({content:"globalThis.__name=fn=>fn;localStorage.setItem('suanlab-language','en');const Native=Worker,active=new Set();globalThis.__ports={created:0,closed:0,active:0};globalThis.Worker=class extends Native{constructor(url,options){super(url,options);active.add(this);__ports.created++;__ports.active=active.size;}terminate(){if(active.delete(this)){__ports.closed++;__ports.active=active.size;}super.terminate();}};"});
  await life.route(rows[0].ports.urls[0],async route=>{const response=await route.fetch();await new Promise(r=>setTimeout(r,2000));try{await route.fulfill({response});}catch{/* A cancelled Worker can abandon its unchanged asset request. */}});
  const lp=await life.newPage();lp.on('pageerror',e=>errors.push(e.message));await lp.goto(url);
  await lp.locator('[data-thermal-status="pending"]').waitFor();await lp.getByRole('button',{name:'Cancel calculation',exact:true}).click();await lp.locator('[data-thermal-status="idle"]').waitFor();
  await lp.getByRole('button',{name:'Recalculate',exact:true}).click();await lp.locator('[data-thermal-status="ready"]').waitFor({timeout:120000});
  await lp.getByLabel('All examples',{exact:true}).selectOption('two-dimensional');await lp.locator('[data-thermal-status="pending"]').waitFor();
  await lp.evaluate(()=>{Object.defineProperty(document,'visibilityState',{configurable:true,get:()=> 'hidden'});document.dispatchEvent(new Event('visibilitychange'));});
  await lp.locator('[data-thermal-status="idle"]').waitFor();assert.equal(await lp.locator('[data-thermal-result]').count(),0);
  await lp.evaluate(()=>{Object.defineProperty(document,'visibilityState',{configurable:true,get:()=> 'visible'});document.dispatchEvent(new Event('visibilitychange'));});
  await lp.locator('[data-thermal-status="ready"]').waitFor({timeout:120000});
  const lifePorts=await lp.evaluate(()=>((window as unknown as{__ports:{created:number;closed:number;active:number}}).__ports));assert.equal(lifePorts.active,0);assert.equal(lifePorts.created,lifePorts.closed);await life.close();
  assert.deepEqual(errors,[]);writeFileSync(root+'/reader-browser.json',JSON.stringify({status:'actual-static-reader-verified',rows,corruptState:true,deniedStorage:true,restoreBeforeWrite:true,restoreBeforeCalculation:true,cancelAndRetry:true,simulatedVisibility:true,lifePorts,productionWorkerAssets:true,errors},null,2)+'\n');console.log(JSON.stringify({readerStates:rows.length,storageCases:3,errors}));
 }finally{await browser.close();await new Promise<void>((resolve,reject)=>server.close(e=>e?reject(e):resolve()));}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
