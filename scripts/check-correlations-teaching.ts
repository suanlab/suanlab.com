/** Standalone component harness QA; does not prove registered-reader behavior. */
import assert from 'node:assert/strict';
import {writeFileSync} from 'node:fs';
import {chromium,type Page} from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import {correlationKinds as kinds,correlationPresets as presets,correlationGuides as guides,correlationPresetInput as preset} from '../src/lib/learning-correlations';
import {correlationConceptSteps as steps} from '../src/lib/learning-correlations-concepts';
import {correlationInputKey as key} from '../src/lib/learning-correlations-protocol';
const folder='.runtime/learning/correlations-teaching',base='http://localhost:4293';
const results:unknown[]=[],errors:string[]=[];
async function main(){
const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
async function waitLab(page:Page,expected:string){await page.locator('[data-correlation-lab][data-correlation-state="ready"]').waitFor({timeout:60000});await page.waitForFunction(k=>document.querySelector('[data-correlation-lab]')?.getAttribute('data-correlation-current')===k,expected,{timeout:60000});}
try{
 for(let index=0;index<3;index++)for(const language of ['ko','en'] as const)for(const dark of [false,true])for(const width of [320,1440]){
  const kind=kinds[index],context=await browser.newContext({viewport:{width,height:900},reducedMotion:'reduce'}),page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
  await page.addInitScript(()=>{const Native=window.Worker,live=new Set<Worker>();const stats={created:0,terminated:0,started:0};Object.assign(window,{correlationWorkerStats:stats,correlationLiveWorkers:live});window.Worker=class extends Native{constructor(...args:ConstructorParameters<typeof Worker>){super(...args);stats.created++;live.add(this);this.addEventListener('message',e=>{if(e.data?.phase==='started')stats.started++;});}terminate(){if(live.delete(this))stats.terminated++;super.terminate();}};});
  await page.goto(`${base}/?kind=${index}`);await page.getByRole('button',{name:language==='ko'?'KO':'EN',exact:true}).click();if(dark)await page.getByRole('button',{name:'Theme',exact:true}).click();
  await waitLab(page,key(preset(kind,0)));await page.locator('[data-correlation-concept-state="ready"]').waitFor({timeout:60000});
  let fixedChoices=0;
  for(let step=0;step<6;step++){
   await page.locator('.ai-concept-steps button').nth(step).click();await page.locator(`[data-correlation-concept][data-correlation-step="${step}"][data-correlation-concept-state="ready"]`).waitFor({timeout:60000});assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'concept overflow');
   const concept=page.locator('[data-correlation-concept]');for(let choice=0;choice<3;choice++){await concept.locator('input[type=radio]').nth(choice).check();await concept.getByRole('button',{name:language==='ko'?'예측 확인':'Check prediction',exact:true}).click();const answer=await concept.locator('.ai-answer').innerText();assert.ok(answer.includes(choice===steps[kind][step].check.answer?(language==='ko'?'맞았습니다.':'Correct.'):(language==='ko'?'다시 생각해보세요.':'Try again.')));fixedChoices++;}
  }
  let prepared=0;
  const lab=page.locator('[data-correlation-lab]');
  for(let guide=0;guide<3;guide++){
   await lab.locator('.ai-guided-choices button').nth(guide).click();const explain=lab.locator('.ai-guided-lab details');await explain.locator('summary').click();assert.ok((await explain.innerText()).includes(guides[kind][guide].explain[language]));
   for(const i of guides[kind][guide].presetIndices){await lab.locator('.ai-guided-lab .ai-lab-actions button').filter({hasText:presets[kind][i].label[language]}).click();await waitLab(page,key(preset(kind,i)));assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'preset overflow');prepared++;}
  }
  const expected=key(preset(kind,presets[kind].length-1));await page.waitForFunction(()=>((window as unknown as {correlationLiveWorkers:Set<Worker>}).correlationLiveWorkers.size===0));const priorWorkers=await page.evaluate(()=>(window as unknown as {correlationWorkerStats:{created:number;terminated:number;started:number}}).correlationWorkerStats);await page.reload();await waitLab(page,expected);await page.locator('[data-correlation-concept-state="ready"]').waitFor({timeout:60000});assert.equal(await page.locator('[data-correlation-lab] .ai-presets button[aria-pressed="true"]').innerText(),presets[kind].at(-1)!.label[language]);
  const quiz=page.locator('#check');for(let q=0;q<3;q++)for(let choice=0;choice<3;choice++){const field=quiz.locator('fieldset').nth(q);await field.locator('input[type=radio]').nth(choice).check();await field.getByRole('button').click();assert.ok(await field.locator('.ai-answer').isVisible());}
  if(language==='en')assert.doesNotMatch(await page.locator('body').innerText(),/[가-힣]/);
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'document overflow');
  const axe=await new AxeBuilder({page}).withTags(['wcag2a','wcag2aa','wcag21a','wcag21aa']).analyze();assert.deepEqual(axe.violations.map(v=>({id:v.id,nodes:v.nodes.map(n=>n.target)})),[]);
  await page.waitForFunction(()=>((window as unknown as {correlationLiveWorkers:Set<Worker>}).correlationLiveWorkers.size===0));
  assert.equal(await page.evaluate(()=>document.documentElement.classList.contains('dark')),dark);const afterWorkers=await page.evaluate(()=>(window as unknown as {correlationWorkerStats:{created:number;terminated:number;started:number}}).correlationWorkerStats);const workers={created:priorWorkers.created+afterWorkers.created,terminated:priorWorkers.terminated+afterWorkers.terminated,started:priorWorkers.started+afterWorkers.started};assert.equal(workers.created,workers.terminated);
  results.push({kind,language,dark,width,steps:6,fixedChoices,guides:3,presets:prepared,quizChoices:9,reload:true,axeViolations:0,workers});console.log(JSON.stringify(results.at(-1)));
  if(language==='en'&&dark&&width===1440)await lab.screenshot({path:`${folder}/final-${index}.png`});await context.close();
 }
 assert.deepEqual(errors,[]);writeFileSync(`${folder}/browser-verification.json`,JSON.stringify({scope:'standalone components with native workers; reader registration/export still pending',cases:results,errors},null,2));
}finally{await browser.close();}

}
main().catch(error=>{console.error(error);process.exitCode=1;});
