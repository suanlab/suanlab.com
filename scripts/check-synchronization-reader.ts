/** Verify actual exported reader assets and native controls; no intercepted worker responses. */
import {chromium} from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import assert from 'node:assert/strict';
import {mkdirSync,writeFileSync} from 'node:fs';
import {synchronizationSteps} from '../src/lib/learning-synchronization-concepts';
import {synchronizationKinds,synchronizationPresets,synchronizationPresetInput,synchronizationInputKey} from '../src/lib/learning-synchronization';
const base=process.env.THERMAL_READER_BASE_URL??'http://localhost:4175',root='.runtime/learning/synchronization-interface';
async function main(){
 mkdirSync(root,{recursive:true});const browser=await chromium.launch({headless:true});
 const report={readerCases:0,presetSelections:0,conceptSteps:0,fixedSelections:0,predictions:0,guideSelections:0,unitQuestions:0,invalidEdits:0,validEdits:0,reloads:0,axeCases:0,sliderSelections:0,keyboardPans:0,touchPans:0,menuCases:0,workersCreated:0,workersTerminated:0,workerURLs:[] as string[],screenshots:[] as string[],failureRecoveryCases:0,pageErrors:[] as string[]};
 try{
 for(const language of ['en','ko'] as const)for(const theme of ['light','dark'])for(const width of [1600,390]){
 const context=await browser.newContext({viewport:{width,height:1000},colorScheme:theme as 'light'|'dark',reducedMotion:'reduce',hasTouch:width===390});
 const init=({language,theme}:{language:string;theme:string})=>{
 localStorage.setItem('suanlab-language',language);localStorage.setItem('theme',theme);localStorage.setItem('suanlab-motion-paused','true');
 const Native=Worker,s=(window as any).synchronizationWorkers={created:0,terminated:0,urls:[] as string[],live:new Set()};
 window.Worker=class extends Native{ended=false;tracked=false;constructor(...args:ConstructorParameters<typeof Worker>){super(...args);this.tracked=args[1]?.name==='synchronization-calculation';if(this.tracked){s.created++;s.urls.push(String(args[0]));s.live.add(this);}}terminate(){if(this.tracked&&!this.ended){this.ended=true;s.terminated++;s.live.delete(this);}super.terminate();}};
 };await context.addInitScript({content:'globalThis.__name=(fn)=>fn;('+init.toString()+')('+JSON.stringify({language,theme})+')'});
 const page=await context.newPage();page.setDefaultTimeout(30000);page.on('pageerror',e=>report.pageErrors.push(e.message));
 const flushWorkers=async()=>{await page.waitForFunction(()=>((window as any).synchronizationWorkers?.live.size??0)===0);const stats=await page.evaluate(()=>{const s=(window as any).synchronizationWorkers;const result={created:s.created,terminated:s.terminated,urls:s.urls};s.created=0;s.terminated=0;s.urls=[];return result;});assert.equal(stats.created,stats.terminated);report.workersCreated+=stats.created;report.workersTerminated+=stats.terminated;report.workerURLs.push(...stats.urls);};
 for(const kind of synchronizationKinds){
 await page.goto(`${base}/book/online/physics-math-for-ai/${kind}/`);await page.waitForFunction(l=>document.documentElement.lang===l,language);
 const lab=page.locator(`[data-synchronization-lab="${kind}"]`),concept=page.locator('.ai-intuition');
 const ready=async()=>{await page.waitForFunction(k=>{const lab=document.querySelector(`[data-synchronization-lab="${k}"]`);return lab?.getAttribute('data-synchronization-state')==='ready'&&lab.getAttribute('data-synchronization-input')===lab.getAttribute('data-synchronization-current');},kind);};await ready();
 await concept.locator('[data-synchronization-concept-state="ready"]').waitFor();
 assert.equal(await lab.locator('[data-synchronization-field]').count(),kind===synchronizationKinds[0]?15:kind===synchronizationKinds[1]?13:10);await page.locator('main a[href="/book/interactive/"]').first().waitFor();
 if(width===1600){const parent=page.locator('header nav a[href="/book/"]');await parent.focus();await parent.press('ArrowDown');assert.equal(await page.getByRole('menuitem',{name:'Interactive Book',exact:true}).getAttribute('aria-current'),'page');assert.equal(await page.getByRole('menuitem',{name:'Online Book',exact:true}).getAttribute('aria-current'),null);await page.keyboard.press('Escape');}else{await page.getByRole('button',{name:language==='ko'?'메뉴 열기':'Open menu',exact:true}).click();const dialog=page.getByRole('dialog');assert.equal(await dialog.getByRole('link',{name:'Interactive Book',exact:true}).getAttribute('aria-current'),'page');await dialog.getByRole('button',{name:language==='ko'?'닫기':'Close',exact:true}).click();}report.menuCases++;
 const coverage=theme==='light'&&width===1600&&language==='en';
 if(coverage){const buttons=lab.locator('fieldset.ai-presets button');assert.equal(await buttons.count(),synchronizationPresets[kind].length);for(let i=0;i<synchronizationPresets[kind].length;i++){
 await buttons.nth(i).click();await ready();assert.equal(await lab.getAttribute('data-synchronization-current'),synchronizationInputKey(synchronizationPresetInput(kind,i)));report.presetSelections++;
 }}
 if((theme==='light'&&width===1600)||(theme==='dark'&&width===390)){
 const steps=concept.locator('.ai-concept-steps button');for(let i=0;i<3;i++){
 await steps.nth(i).click();await concept.locator(`[data-synchronization-step="${i}"][data-synchronization-concept-state="ready"]`).waitFor();report.conceptSteps++;
 const fixed=concept.getByRole('group',{name:language==='ko'?'고정 준비 선택':'Choose a fixed preparation'}).locator('button');
 for(let n=0;n<await fixed.count();n++){await fixed.nth(n).click();assert.equal(await concept.locator('[data-synchronization-result]').count(),1);report.fixedSelections++;}
 await concept.locator('.ai-question input').nth(synchronizationSteps[kind][i].answer).check();await concept.getByRole('button',{name:language==='ko'?'예측 확인':'Check prediction',exact:true}).click();await concept.locator('.ai-answer-correct').waitFor();report.predictions++;
 }}
 const guide=lab.getByRole('group',{name:language==='ko'?'안내 실험 선택':'Choose a guided experiment'}).locator('button');for(let i=0;i<3;i++){await guide.nth(i).click();await lab.locator('.ai-guided-lab .ai-lab-actions button').first().click();await ready();report.guideSelections++;}
 const temp=lab.locator('[data-synchronization-field="timeNumerator"]'),key=await lab.getAttribute('data-synchronization-input');await temp.fill('1.00000000000000001');await temp.press('Tab');assert.equal(await temp.getAttribute('aria-invalid'),'true');assert.equal(await lab.getAttribute('data-synchronization-input'),key);report.invalidEdits++;
 await temp.fill('3');await temp.press('Tab');await ready();assert.equal(await temp.getAttribute('aria-invalid'),'false');assert.equal(JSON.parse((await lab.getAttribute('data-synchronization-current'))!).parameters.timeNumerator,3);report.validEdits++;
 const saved=await lab.getAttribute('data-synchronization-current');await flushWorkers();await page.reload();await ready();assert.equal(await lab.getAttribute('data-synchronization-current'),saved);report.reloads++;
 const slider=lab.getByRole('slider');await slider.focus();await slider.press('Home');assert.equal(await lab.locator('[data-synchronization-result]').getAttribute('data-synchronization-record'),'0');await slider.press('End');report.sliderSelections+=2;
 assert(await lab.getByRole('button',{name:language==='ko'?'시간 재생':'Play time',exact:true}).isDisabled());
 if(width===390){const region=lab.locator('[data-synchronization-diagram]').first();await region.scrollIntoViewIfNeeded();await region.focus();await page.keyboard.press('ArrowRight');await page.waitForFunction(el=>el!.scrollLeft>0,await region.elementHandle());report.keyboardPans++;await region.evaluate(el=>{el.scrollLeft=0;});const box=(await region.boundingBox())!;const cdp=await context.newCDPSession(page),x=box.x+Math.min(box.width-20,250),y=box.y+Math.min(box.height-20,100);await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x,y}]});for(let j=1;j<=5;j++)await cdp.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x:x-j*30,y}]});await cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});await page.waitForFunction(el=>el!.scrollLeft>0,await region.elementHandle());await cdp.detach();report.touchPans++;}
 const quiz=page.locator('.ai-quiz .ai-question');for(let n=0;n<3;n++){await quiz.nth(n).locator('input[type=radio]').nth(synchronizationSteps[kind][n].answer).check();await quiz.nth(n).getByRole('button',{name:language==='ko'?'답과 해설 확인':'Check answer and explanation',exact:true}).click();await quiz.nth(n).locator('.ai-answer-correct').waitFor();report.unitQuestions++;}
 const geometry=await page.locator('main').evaluate(main=>({overflow:document.documentElement.scrollWidth>innerWidth+1,nonfinite:[...main.querySelectorAll('svg [cx],svg [cy],svg path[d]')].some(n=>/NaN|Infinity/.test([...n.attributes].map(a=>a.value).join(' '))),hangul:/[가-힣]/.test((main as HTMLElement).innerText)}));assert(!geometry.overflow,JSON.stringify({language,theme,width,kind,geometry}));assert(!geometry.nonfinite);if(language==='en')assert(!geometry.hangul);
 const a11y=await new AxeBuilder({page}).include('main').withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();assert.deepEqual(a11y.violations.map(v=>({id:v.id,nodes:v.nodes.map(n=>n.target)})),[]);report.axeCases++;
 if((language==='en'&&theme==='light'&&width===1600)||(language==='ko'&&theme==='dark'&&width===390)){const file=`${root}/${kind}-${language}-${theme}-${width}.png`;await lab.locator('[data-synchronization-result]').scrollIntoViewIfNeeded();await page.screenshot({path:file});report.screenshots.push(file);}
 await flushWorkers();report.readerCases++;console.log(`PASS ${kind} ${language} ${theme} ${width}`);
 }
 await page.goto(`${base}/book/online/`);assert.equal(await page.locator('main a[href*="physics-math-for-ai"]').count(),0);assert(await page.locator('main a[href^="/book/online/"]').count());await page.goto(`${base}/book/interactive/`);await page.waitForFunction(()=>((window as any).synchronizationWorkers?.live.size??0)===0);
 await context.close();
 }
 const recovery=await browser.newContext({viewport:{width:1600,height:1000}});await recovery.addInitScript({content:'globalThis.__name=(fn)=>fn;localStorage.setItem("suanlab-language","en");'});const rp=await recovery.newPage();rp.on('pageerror',e=>report.pageErrors.push(e.message));
 await rp.goto(`${base}/book/online/physics-math-for-ai/${synchronizationKinds[0]}/`);const recoveryLab=rp.locator('[data-synchronization-lab]');await recoveryLab.locator('[data-synchronization-result]').waitFor();
 const assetURLs=[...new Set(report.workerURLs)];for(const asset of assetURLs)await recovery.route(asset,route=>route.fulfill({status:200,contentType:'text/javascript',body:'throw new Error("intentional reader worker failure");'}));
 await recoveryLab.locator('fieldset.ai-presets button').nth(1).click();await rp.locator('[data-synchronization-state="error"]').waitFor();assert.equal(await recoveryLab.locator('[data-synchronization-result]').count(),0);for(const asset of assetURLs)await recovery.unroute(asset);
 await recoveryLab.getByRole('button',{name:'Retry calculation',exact:true}).click();await rp.locator('[data-synchronization-state="ready"]').waitFor();await recoveryLab.locator('[data-synchronization-result]').waitFor();report.failureRecoveryCases++;await recovery.close();
 const xml=await(await fetch(base+'/sitemap.xml')).text();for(const kind of synchronizationKinds)assert(xml.includes('/physics-math-for-ai/'+kind+'/'));assert.deepEqual(report.pageErrors,[]);report.workerURLs=[...new Set(report.workerURLs)];assert(report.workerURLs.length>0&&report.workersCreated>=report.readerCases*2);assert(report.workerURLs.every(u=>u.includes('/_next/static/')));writeFileSync(`${root}/native-reader.json`,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
 }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
