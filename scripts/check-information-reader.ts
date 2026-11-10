import assert from 'node:assert/strict';
import {writeFileSync} from 'node:fs';
import {chromium,type Page} from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import {informationTeaching} from '../src/data/information-teaching';
const root='.runtime/learning/quantum-information-reader-integration',base=process.env.INFORMATION_PREVIEW_BASE??'http://127.0.0.1:4176';
const instrumentation=`globalThis.__name=fn=>fn;window.__informationWorkers={created:0,terminated:0};const Native=Worker;window.Worker=class extends Native{constructor(...args){super(...args);window.__informationWorkers.created++;this.closed=false;}terminate(){if(!this.closed){this.closed=true;window.__informationWorkers.terminated++;}super.terminate();}};`;
async function main(){
 const browser=await chromium.launch({headless:true}),states:unknown[]=[],errors:string[]=[];let concepts=0,questions=0,accessibility=0;
 const ready=async(page:Page)=>page.locator('[data-information-phase="ready"]').waitFor({timeout:60000});
 const released=async(page:Page)=>{const counts=await page.evaluate(()=>(window as unknown as {__informationWorkers:{created:number;terminated:number}}).__informationWorkers);assert.equal(counts.created,counts.terminated);return counts;};
 try{
  for(const language of ['ko','en']as const)for(const dark of [false,true])for(const width of [320,1440]){
   const context=await browser.newContext({viewport:{width,height:1000},reducedMotion:'reduce'});
   await context.addInitScript({content:instrumentation});await context.addInitScript(({language,dark})=>{localStorage.setItem('suanlab-language',language);localStorage.setItem('theme',dark?'dark':'light');},{language,dark});
   const page=await context.newPage();page.on('pageerror',error=>errors.push(error.message));
   for(const item of informationTeaching){
    const kind=item.unit.id;
    await page.goto(base+'/book/online/physics-math-for-ai/'+kind+'/',{waitUntil:'domcontentloaded'});await ready(page);
    await page.waitForFunction(lang=>document.querySelector('.ai-reader')?.getAttribute('lang')===lang,language);
    assert.equal(await page.locator('h1').first().innerText(),item.unit.title[language]);
    assert.equal(await page.locator('[data-information-reader-lab]').getAttribute('data-information-reader-lab'),kind);
    assert.equal(await page.locator('[data-information-concept]').count(),1);
    assert.equal(await page.locator('.information-step,.information-question').count(),0,'Embedded lab duplicates concepts or quiz');
    assert.equal(await page.locator('.ai-question').count(),item.unit.questions.length);
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),kind+' overflow');
    if(language==='en')assert(!/[가-힣]/.test(await page.locator('.ai-reader').innerText()),kind+' untranslated');
    for(const[index,step]of item.guide.steps.entries()){
     await page.locator('[data-information-concept] .ai-concept-steps button').nth(index).click();
     assert.equal(await page.locator('[data-information-concept]').getAttribute('data-information-concept-step'),String(index));
     assert.equal(await page.locator('.ai-concept-explanation h3').innerText(),step.title[language]);
     assert.equal(await page.locator('.ai-concept-explanation > p').first().innerText(),step.explanation[language]);concepts++;
    }
    for(const[index,question]of item.unit.questions.entries()){
     const field=page.locator('.ai-question').nth(index),radios=field.locator('input[type=radio]');
     await radios.nth((question.answer+1)%question.options.length).check();await field.getByRole('button').click();
     assert((await field.locator('.ai-answer strong').innerText()).startsWith(language==='ko'?'다시 생각해보세요':'Try again.'));
     await radios.nth(question.answer).check();await field.getByRole('button').click();
     assert.equal(await field.locator('.ai-answer p').innerText(),question.explanation[language]);questions++;
    }
    await page.locator('.ai-completion input').check();
    const examples=page.getByLabel(language==='ko'?'전체 예제':'All examples',{exact:true});
    const last=await examples.locator('option').last().getAttribute('value');assert(last);await examples.selectOption(last);await ready(page);
    await released(page);
    const axe=await new AxeBuilder({page}).include('.ai-reader').withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();
    assert.deepEqual(axe.violations.map(v=>({id:v.id,nodes:v.nodes.map(n=>n.target)})),[],kind);accessibility++;
    if(language==='en'&&width===320)await page.screenshot({path:root+'/'+kind+'-'+(dark?'dark':'light')+'-320.png'});
    await page.reload();await ready(page);assert.equal(await examples.inputValue(),last);assert(await page.locator('.ai-completion input').isChecked());
    for(const[index,question]of item.unit.questions.entries())assert(await page.locator('.ai-question').nth(index).locator('input[type=radio]').nth(question.answer).isChecked());
    await page.locator('.ai-reader-top a').first().click();await page.locator('.ai-book-overview').waitFor();assert.equal(await page.locator('[data-information-lab]').count(),0);const workers=await released(page);
    states.push({kind,language,dark,width,concepts:item.guide.steps.length,questions:item.unit.questions.length,workers,inputQuizCompletionReload:true,accessibilityPassed:true});
    writeFileSync(root+'/reader-browser-progress.json',JSON.stringify({states,concepts,questions,accessibility,errors},null,2)+'\n');
    console.log(JSON.stringify({cases:states.length,kind,language,dark,width}));
   }await context.close();
  }
  assert.equal(states.length,72);assert.equal(concepts,504);assert.equal(questions,360);assert.equal(accessibility,72);assert.deepEqual(errors,[]);
  writeFileSync(root+'/reader-browser.json',JSON.stringify({states,concepts,questions,accessibility,errors,actualProductionReader:true,workerFactoryOverride:false},null,2)+'\n');
 }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
