/** Verify actual exported reader assets and native controls; no intercepted worker responses. */
import {chromium} from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import assert from 'node:assert/strict';
import {mkdirSync,writeFileSync} from 'node:fs';
import {hilbertSteps} from '../src/lib/learning-hilbert-concepts';
import {hilbertKinds,hilbertPresets,hilbertPresetInput,hilbertInputKey} from '../src/lib/learning-hilbert';
const base=process.env.HILBERT_READER_BASE_URL??'http://localhost:4175',root='.runtime/learning/hilbert-integration';
async function main(){
 mkdirSync(root,{recursive:true});const browser=await chromium.launch({headless:true,args:['--no-sandbox','--enable-unsafe-swiftshader','--disable-gpu-rasterization']});
 const report={readerCases:0,presetSelections:0,conceptSteps:0,fixedSelections:0,predictions:0,guideSelections:0,unitQuestions:0,invalidEdits:0,validEdits:0,reloads:0,axeCases:0,sliderSelections:0,keyboardPans:0,touchPans:0,menuCases:0,workersCreated:0,workersTerminated:0,workerURLs:[] as string[],screenshots:[] as string[],failureRecoveryCases:0,threeDCases:0,webglLossRecoveries:0,pageErrors:[] as string[]};
 try{
 for(const language of ['en','ko'] as const)for(const theme of ['light','dark'])for(const width of [1600,390]){
 const context=await browser.newContext({viewport:{width,height:1000},colorScheme:theme as 'light'|'dark',reducedMotion:'reduce',hasTouch:width===390});
 const init=({language,theme}:{language:string;theme:string})=>{
 localStorage.setItem('suanlab-language',language);localStorage.setItem('theme',theme);localStorage.setItem('suanlab-motion-paused','true');
 const Native=Worker,s=(window as any).hilbertWorkers={created:0,terminated:0,urls:[] as string[],live:new Set()};
 window.Worker=class extends Native{ended=false;tracked=false;constructor(...args:ConstructorParameters<typeof Worker>){super(...args);this.tracked=args[1]?.name==='hilbert-calculation';if(this.tracked){s.created++;s.urls.push(String(args[0]));s.live.add(this);}}terminate(){if(this.tracked&&!this.ended){this.ended=true;s.terminated++;s.live.delete(this);}super.terminate();}};
 };await context.addInitScript({content:'globalThis.__name=(fn)=>fn;('+init.toString()+')('+JSON.stringify({language,theme})+')'});
 const page=await context.newPage();page.setDefaultTimeout(30000);page.on('pageerror',e=>report.pageErrors.push(e.message));
 const flushWorkers=async()=>{await page.waitForFunction(()=>((window as any).hilbertWorkers?.live.size??0)===0);const stats=await page.evaluate(()=>{const s=(window as any).hilbertWorkers;const result={created:s.created,terminated:s.terminated,urls:s.urls};s.created=0;s.terminated=0;s.urls=[];return result;});assert.equal(stats.created,stats.terminated);report.workersCreated+=stats.created;report.workersTerminated+=stats.terminated;report.workerURLs.push(...stats.urls);};
 for(const kind of hilbertKinds){
 await page.goto(`${base}/book/online/physics-math-for-ai/${kind}/`);await page.waitForFunction(l=>document.documentElement.lang===l,language);
 const lab=page.locator(`[data-hilbert-lab="${kind}"]`),concept=page.locator('.ai-intuition');
 const ready=async()=>{await page.waitForFunction(k=>{const lab=document.querySelector(`[data-hilbert-lab="${k}"]`);return lab?.getAttribute('data-hilbert-state')==='ready'&&lab.getAttribute('data-hilbert-input')===lab.getAttribute('data-hilbert-current');},kind);};await ready();
 await concept.locator('[data-hilbert-concept-state="ready"]').waitFor();
 assert.equal(await lab.locator('[data-hilbert-field]').count(),kind===hilbertKinds[0]?16:15);await page.locator('main a[href="/book/interactive/"]').first().waitFor();
 if(width===1600){const parent=page.locator('header nav a[href="/book/"]');await parent.focus();await parent.press('ArrowDown');assert.equal(await page.getByRole('menuitem',{name:'Interactive Book',exact:true}).getAttribute('aria-current'),'page');assert.equal(await page.getByRole('menuitem',{name:'Online Book',exact:true}).getAttribute('aria-current'),null);await page.keyboard.press('Escape');}else{await page.getByRole('button',{name:language==='ko'?'메뉴 열기':'Open menu',exact:true}).click();const dialog=page.getByRole('dialog');assert.equal(await dialog.getByRole('link',{name:'Interactive Book',exact:true}).getAttribute('aria-current'),'page');await dialog.getByRole('button',{name:language==='ko'?'닫기':'Close',exact:true}).click();}report.menuCases++;
 const coverage=theme==='light'&&width===1600&&language==='en';
 if(coverage){const buttons=lab.locator('fieldset.ai-presets button');assert.equal(await buttons.count(),hilbertPresets[kind].length);for(let i=0;i<hilbertPresets[kind].length;i++){
 await buttons.nth(i).click();await ready();assert.equal(await lab.getAttribute('data-hilbert-current'),hilbertInputKey(hilbertPresetInput(kind,i)));report.presetSelections++;
 }}
 if((theme==='light'&&width===1600)||(theme==='dark'&&width===390)){
 const steps=concept.locator('.ai-concept-steps button');for(let i=0;i<3;i++){
 await steps.nth(i).click();await concept.locator(`[data-hilbert-step="${i}"][data-hilbert-concept-state="ready"]`).waitFor();report.conceptSteps++;
 const fixed=concept.getByRole('group',{name:language==='ko'?'고정 준비 선택':'Choose a fixed preparation'}).locator('button');
 for(let n=0;n<await fixed.count();n++){await fixed.nth(n).click();assert.equal(await concept.locator('[data-hilbert-result]').count(),1);report.fixedSelections++;}
 await concept.locator('.ai-question input').nth(hilbertSteps[kind][i].answer).check();await concept.getByRole('button',{name:language==='ko'?'예측 확인':'Check prediction',exact:true}).click();await concept.locator('.ai-answer-correct').waitFor();report.predictions++;
 }}
 const guide=lab.getByRole('group',{name:language==='ko'?'안내 실험 선택':'Choose a guided experiment'}).locator('button');for(let i=0;i<3;i++){await guide.nth(i).click();await lab.locator('.ai-guided-lab .ai-lab-actions button').first().click();await ready();report.guideSelections++;}
 const inputField=kind===hilbertKinds[0]?'phaseA':'cutoffDegree';const temp=lab.locator(`[data-hilbert-field="${inputField}"]`),key=await lab.getAttribute('data-hilbert-input');await temp.fill('1.00000000000000001');await temp.press('Tab');assert.equal(await temp.getAttribute('aria-invalid'),'true');assert.equal(await lab.getAttribute('data-hilbert-input'),key);report.invalidEdits++;
 await temp.fill('3');await temp.press('Tab');await ready();assert.equal(await temp.getAttribute('aria-invalid'),'false');assert.equal(JSON.parse((await lab.getAttribute('data-hilbert-current'))!).parameters[inputField],3);report.validEdits++;
 const saved=await lab.getAttribute('data-hilbert-current');await flushWorkers();await page.reload();await ready();assert.equal(await lab.getAttribute('data-hilbert-current'),saved);report.reloads++;
 if(kind===hilbertKinds[1]){
 const wave=lab.locator('[data-hilbert-record]'),slider=wave.getByRole('slider');await slider.focus();await slider.press('Home');assert.equal(await wave.getAttribute('data-hilbert-record'),'0');await slider.press('End');report.sliderSelections+=2;assert(await wave.getByRole('button',{name:language==='ko'?'그림 스캔':'Scan plot',exact:true}).isDisabled());
 }else{
 const sphere=lab.locator('[data-hilbert-bloch]');await sphere.scrollIntoViewIfNeeded();await page.waitForFunction(el=>el?.getAttribute('data-hilbert-bloch-status')==='3d',await sphere.elementHandle());report.threeDCases++;
 await sphere.getByRole('button',{name:language==='ko'?'정면':'Front',exact:true}).click();await sphere.getByRole('button',{name:language==='ko'?'위에서':'Top',exact:true}).click();await sphere.getByRole('button',{name:language==='ko'?'2D 보기':'Show 2D',exact:true}).click();assert.equal(await sphere.getAttribute('data-hilbert-bloch-status'),'flat');await sphere.getByRole('button',{name:language==='ko'?'3D 보기':'Show 3D',exact:true}).click();await page.waitForFunction(el=>el?.getAttribute('data-hilbert-bloch-status')==='3d',await sphere.elementHandle());
 if(theme==='light'&&width===1600){await sphere.locator('canvas').evaluate((canvas:HTMLCanvasElement)=>{canvas.getContext('webgl2')?.getExtension('WEBGL_lose_context')?.loseContext();});await page.waitForFunction(el=>el?.getAttribute('data-hilbert-bloch-status')==='2d',await sphere.elementHandle());await sphere.getByRole('button',{name:language==='ko'?'3D 다시 시도':'Retry 3D',exact:true}).click();await page.waitForFunction(el=>el?.getAttribute('data-hilbert-bloch-status')==='3d',await sphere.elementHandle());report.webglLossRecoveries++;}
 if((language==='en'&&theme==='light'&&width===1600)||(language==='ko'&&theme==='dark'&&width===390)){const path=`${root}/bloch-${language}-${theme}-${width}.png`;await sphere.screenshot({path});report.screenshots.push(path);}
 }
 if(width===390){const region=lab.locator('[data-hilbert-diagram]').first();await region.scrollIntoViewIfNeeded();await region.focus();await page.keyboard.press('ArrowRight');await page.waitForFunction(el=>el!.scrollLeft>0,await region.elementHandle());report.keyboardPans++;await region.evaluate(el=>{el.scrollLeft=0;});const box=(await region.boundingBox())!;const cdp=await context.newCDPSession(page),x=box.x+Math.min(box.width-20,250),y=box.y+Math.min(box.height-20,100);await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x,y}]});for(let j=1;j<=5;j++)await cdp.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x:x-j*30,y}]});await cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});await page.waitForFunction(el=>el!.scrollLeft>0,await region.elementHandle());await cdp.detach();report.touchPans++;}
 const quiz=page.locator('.ai-quiz .ai-question');for(let n=0;n<3;n++){await quiz.nth(n).locator('input[type=radio]').nth(hilbertSteps[kind][n].answer).check();await quiz.nth(n).getByRole('button',{name:language==='ko'?'답과 해설 확인':'Check answer and explanation',exact:true}).click();await quiz.nth(n).locator('.ai-answer-correct').waitFor();report.unitQuestions++;}
 const geometry=await page.locator('main').evaluate(main=>({overflow:document.documentElement.scrollWidth>innerWidth+1,nonfinite:[...main.querySelectorAll('svg [cx],svg [cy],svg path[d]')].some(n=>/NaN|Infinity/.test([...n.attributes].map(a=>a.value).join(' '))),hangul:/[가-힣]/.test((main as HTMLElement).innerText)}));assert(!geometry.overflow,JSON.stringify({language,theme,width,kind,geometry}));assert(!geometry.nonfinite);if(language==='en')assert(!geometry.hangul);
 const a11y=await new AxeBuilder({page}).include('main').withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();assert.deepEqual(a11y.violations.map(v=>({id:v.id,nodes:v.nodes.map(n=>n.target)})),[]);report.axeCases++;
 if((language==='en'&&theme==='light'&&width===1600)||(language==='ko'&&theme==='dark'&&width===390)){const file=`${root}/${kind}-${language}-${theme}-${width}.png`;await lab.locator('[data-hilbert-result]').scrollIntoViewIfNeeded();await page.screenshot({path:file});report.screenshots.push(file);}
 await flushWorkers();report.readerCases++;console.log(`PASS ${kind} ${language} ${theme} ${width}`);
 }
 await page.goto(`${base}/book/online/`);assert.equal(await page.locator('main a[href*="physics-math-for-ai"]').count(),0);assert(await page.locator('main a[href^="/book/online/"]').count());await page.goto(`${base}/book/interactive/`);await page.waitForFunction(()=>((window as any).hilbertWorkers?.live.size??0)===0);
 await context.close();
 }
 const recovery=await browser.newContext({viewport:{width:1600,height:1000}});await recovery.addInitScript({content:'globalThis.__name=(fn)=>fn;localStorage.setItem("suanlab-language","en");'});const rp=await recovery.newPage();rp.on('pageerror',e=>report.pageErrors.push(e.message));
 for(const kind of hilbertKinds){
 await rp.goto(`${base}/book/online/physics-math-for-ai/${kind}/`);const recoveryLab=rp.locator('[data-hilbert-lab]');await recoveryLab.locator('[data-hilbert-result]').waitFor();
 const assetURLs=[...new Set(report.workerURLs)];for(const asset of assetURLs)await recovery.route(asset,route=>route.fulfill({status:200,contentType:'text/javascript',body:'throw new Error("intentional reader worker failure");'}));
 await recoveryLab.locator('fieldset.ai-presets button').nth(1).click();await rp.locator('[data-hilbert-state="error"]').waitFor();assert.equal(await recoveryLab.locator('[data-hilbert-result]').count(),0);for(const asset of assetURLs)await recovery.unroute(asset);
 await recoveryLab.getByRole('button',{name:'Retry calculation',exact:true}).click();await rp.locator('[data-hilbert-state="ready"]').waitFor();await recoveryLab.locator('[data-hilbert-result]').waitFor();report.failureRecoveryCases++;}
 await recovery.close();
 const xml=await(await fetch(base+'/sitemap.xml')).text();for(const kind of hilbertKinds)assert(xml.includes('/physics-math-for-ai/'+kind+'/'));assert.deepEqual(report.pageErrors,[]);report.workerURLs=[...new Set(report.workerURLs)];assert(report.workerURLs.length>0&&report.workersCreated>=report.readerCases*2);assert(report.workerURLs.every(u=>u.includes('/_next/static/')));writeFileSync(`${root}/native-reader.json`,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
 }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
