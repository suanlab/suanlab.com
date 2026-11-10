import assert from 'node:assert/strict';
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {chromium} from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import {saddleFluctuationExamples} from '../src/lib/learning-saddle-fluctuation-input';
import {saddleFluctuationQuestions,saddleFluctuationSteps,saddleFluctuationExperiments} from '../src/data/saddle-fluctuation-teaching';
const root='.runtime/learning/saddle-fluctuation-teaching';mkdirSync(root,{recursive:true});
const {url}=JSON.parse(readFileSync(root+'/review-server.json','utf8')) as {url:string};
type Harness=Window&{__saddleWorkers:{created:number;terminated:number;active:Set<Worker>}};
async function main(){
 const browser=await chromium.launch({headless:true}),rows:unknown[]=[],errors:string[]=[];let accessibilityCases=0,conceptChecks=0,quizChecks=0,guideChecks=0,exampleApplications=0;
 try{
  for(const language of ['ko','en'] as const)for(const theme of ['light','dark'] as const)for(const width of [320,1440]){
   const context=await browser.newContext({viewport:{width,height:1000},hasTouch:width===320,isMobile:width===320,reducedMotion:'reduce'}),page=await context.newPage(),label=`${language}-${theme}-${width}`;
   page.on('pageerror',e=>errors.push(label+': '+e.message));
   await context.addInitScript({content:'globalThis.__name=fn=>fn;'});
   await page.addInitScript(({language,theme})=>{
    localStorage.setItem('suanlab-language',language);localStorage.setItem('theme',theme);
    const Original=window.Worker,stats={created:0,terminated:0,active:new Set<Worker>()};(window as unknown as Harness).__saddleWorkers=stats;
    window.Worker=class extends Original {constructor(url:string|URL,options?:WorkerOptions){super(url,options);stats.created++;stats.active.add(this);}terminate(){if(stats.active.delete(this)){stats.terminated++;super.terminate();}}};
   },{language,theme});
   const ready=async(kind:'chain'|'well'='chain')=>{await page.locator('[data-saddle-phase="ready"]').waitFor({timeout:120000});await page.locator(`[data-saddle-result="${kind}"]`).waitFor();};
   const choose=async(id:string)=>{const e=saddleFluctuationExamples.find(e=>e.id===id)!;await page.getByLabel(language==='ko'?'전체 예제':'All examples',{exact:true}).selectOption(id);await ready(e.input.mode==='chain'?'chain':'well');exampleApplications++;};
   const stats=()=>page.evaluate(()=>{const s=(window as unknown as Harness).__saddleWorkers;return {created:s.created,terminated:s.terminated,active:s.active.size};});
   const audit=async(name:string)=>{const result=await new AxeBuilder({page}).include('main').withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();writeFileSync(`${root}/axe-${label}-${name}.json`,JSON.stringify(result,null,2)+'\n');assert.deepEqual(result.violations.map(v=>({id:v.id,nodes:v.nodes.map(n=>n.target)})),[],label+' '+name);accessibilityCases++;};
   await page.goto(url);await ready();
   assert.equal(await page.locator('main').evaluate(node=>node.getAttribute('lang')||document.documentElement.lang),language);
   assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),label+' page overflow');
   const beforeControls=await stats();
   for(let i=0;i<10;i++){
    await page.locator('.ai-concept-steps button').nth(i).click();await page.locator(`[data-saddle-reference="${i}"]`).waitFor();
    assert((await page.locator('.ai-concept-explanation').innerText()).includes(saddleFluctuationSteps[i].body[language]));conceptChecks++;
   }
   for(let i=0;i<8;i++){await page.locator('.information-question').nth(i).getByRole('radio').nth(saddleFluctuationQuestions[i].correct).check();assert((await page.locator('.information-question').nth(i).innerText()).includes(saddleFluctuationQuestions[i].explanation[language]));quizChecks++;}
   for(let i=0;i<7;i++){
    await page.getByLabel(language==='ko'?'안내 실험':'Guided experiment',{exact:true}).selectOption(String(i));
    await page.getByRole('button',{name:language==='ko'?'해설 보기':'Show explanation',exact:true}).click();
    assert.equal(await page.locator('[data-saddle-guide-explanation]').innerText(),saddleFluctuationExperiments[i].explanation[language]);guideChecks++;
   }
   assert.deepEqual(await stats(),beforeControls,'concepts/questions/guides must not calculate');
   if(language==='en')assert(!/[가-힣]/u.test(await page.locator('main').innerText()),label+' untranslated Korean');
   await page.getByLabel(language==='ko'?'행렬':'Matrix',{exact:true}).selectOption('hessian');
   await page.getByLabel(language==='ko'?'높이·색 척도':'Height and color scale',{exact:true}).selectOption('signed-log');
   await page.getByRole('button',{name:language==='ko'?'3D 행렬 보기':'Show 3D matrix',exact:true}).click();
   await page.locator('[data-saddle-matrix-plot="3d"]').waitFor({timeout:30000});
   const canvas=page.locator('.scalar-three-host canvas');assert((await canvas.boundingBox())!.height>300);
   assert.equal(await page.getByRole('button',{name:language==='ko'?'표시 시각 재생':'Play display times',exact:true}).isDisabled(),true);
   await page.getByLabel(language==='ko'?'표시 시각 첨자':'Display time index',{exact:true}).fill('0');
   assert.equal((await stats()).created,beforeControls.created,'display controls must not create Workers');
   assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),label+' 3D overflow');
   await audit('chain');
   await page.locator('[data-saddle-matrix-plot]').screenshot({path:`${root}/matrix-${label}.png`});
   await choose('double-well');await audit('well');
   assert.equal(await page.locator('[data-saddle-result="well"] table').first().locator('tbody tr').count(),3);
   await page.getByLabel(language==='ko'?'국소 확대의 최소점':'Minimum for the local zoom',{exact:true}).selectOption('left');
   await choose('degenerate-well');assert((await page.locator('[data-saddle-result="well"]').innerText()).includes(language==='ko'?'Gaussian 근사는 사용할 수 없지만':'Gaussian approximation unavailable'));
   if(language==='en')assert(!/[가-힣]/u.test(await page.locator('main').innerText()),label+' untranslated well');
   if(language==='en'&&theme==='light'&&width===1440){
    for(const e of saddleFluctuationExamples)await choose(e.id);
   }
   await choose(width===320?'maximum-thirty-two':'one-variable-nonzero-hand');
   const countBefore=await stats();
   await page.getByRole('button',{name:language==='ko'?'3D 행렬 보기':'Show 3D matrix',exact:true}).click();await page.locator('[data-saddle-matrix-plot="3d"]').waitFor({timeout:30000});
   await canvas.focus();await page.keyboard.press('ArrowRight');await page.keyboard.press('Home');
   if(width===320){
    await canvas.scrollIntoViewIfNeeded();const before=await canvas.screenshot(),box=(await canvas.boundingBox())!,client=await context.newCDPSession(page),x=box.x+box.width*.35,y=box.y+box.height*.5;
    await client.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{id:1,x,y}]});await client.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{id:1,x:x+box.width*.25,y}]});await client.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});await client.detach();
    assert.notDeepEqual(await canvas.screenshot(),before,'touch gesture did not rotate the maximum matrix');
   }
   await page.getByLabel(language==='ko'?'행렬':'Matrix',{exact:true}).selectOption('hessian');
   const selected=page.locator('button[tabindex="0"][aria-pressed="true"]');await selected.focus();await page.keyboard.press('ArrowRight');
   assert.equal((await stats()).created,countBefore.created);
   if(language==='en'&&theme==='light'&&width===1440){
    const lost=await canvas.evaluate(node=>{const gl=(node as HTMLCanvasElement).getContext('webgl2');const extension=gl?.getExtension('WEBGL_lose_context');if(!extension)return false;extension.loseContext();return true;});assert(lost,'real WebGL loss extension unavailable');
    await page.getByRole('button',{name:'Retry 3D',exact:true}).waitFor();await page.getByRole('button',{name:'Retry 3D',exact:true}).click();await page.locator('[data-saddle-matrix-plot="3d"]').waitFor({timeout:30000});
   }
   const saved=await page.evaluate(()=>JSON.parse(localStorage.getItem('suanlab-saddle-fluctuation-v1')!));
   assert.equal(saved.learning.step,9);assert.equal(saved.learning.experiment,6);assert.deepEqual(saved.learning.answers,saddleFluctuationQuestions.map(q=>q.correct));assert.equal(saved.display.kind,'hessian');
   const beforeReload=await stats();assert.equal(beforeReload.active,0);assert.equal(beforeReload.created,beforeReload.terminated);
   await page.reload();await ready();
   const restored=await page.evaluate(()=>JSON.parse(localStorage.getItem('suanlab-saddle-fluctuation-v1')!));assert.deepEqual(restored,saved);
   assert.equal(await page.locator('[data-saddle-reference="9"]').count(),1);
   assert.equal(await page.getByLabel(language==='ko'?'행렬':'Matrix',{exact:true}).inputValue(),'hessian');
   assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),label+' restored overflow');
   const afterReload=await stats();assert.equal(afterReload.active,0);assert.equal(afterReload.created,afterReload.terminated);
   rows.push({language,theme,width,beforeReload,afterReload,actualModuleWorkers:true,maximumInput:width===320,threeD:true,reducedMotion:true,restored:true});writeFileSync(root+'/native-progress.json',JSON.stringify({rows,accessibilityCases,conceptChecks,quizChecks,guideChecks,exampleApplications,errors},null,2)+'\n');console.log(JSON.stringify({case:label,finished:rows.length,exampleApplications}));
   await context.close();
  }
  assert.deepEqual(errors,[]);const report={rows,accessibilityCases,conceptChecks,quizChecks,guideChecks,exampleApplications,errors,readerRegistered:false,productionNextAssetsVerified:false,all32ExamplesApplied:true,webglContextLossRecovery:true,chromiumDevToolsTouch:true,physicalTouchDevice:false};writeFileSync(root+'/native-teaching.json',JSON.stringify(report,null,2)+'\n');
 }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
