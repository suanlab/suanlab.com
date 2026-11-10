import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {chromium} from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import {realTimePhaseGuides as guides,realTimePhaseGuidePreparations as preparations,realTimePhaseQuestions as questions} from '../src/data/real-time-phase-teaching';
const root='.runtime/learning/real-time-phase-lessons',styleOnly=process.env.REAL_TIME_STYLE_ONLY==='1';
async function main(){
  const {url}=JSON.parse(readFileSync(`${root}/review-server.json`,'utf8')),browser=await chromium.launch({headless:true}),rows=[];
  try{for(const language of ['ko','en'] as const)for(const theme of ['light','dark'])for(const width of [320,1440]){
    const context=await browser.newContext({viewport:{width,height:1000},reducedMotion:'reduce'});
    await context.addInitScript({content:`globalThis.__name=fn=>fn;localStorage.setItem('suanlab-language',${JSON.stringify(language)});localStorage.setItem('theme',${JSON.stringify(theme)});const Native=Worker,active=new Set();globalThis.__lessonCounts={created:0,closed:0,active:0};globalThis.Worker=class extends Native{constructor(url,options){super(url,options);active.add(this);__lessonCounts.created++;__lessonCounts.active=active.size;}terminate(){if(active.delete(this)){__lessonCounts.closed++;__lessonCounts.active=active.size;}super.terminate();}};`});
    const page=await context.newPage(),errors:string[]=[],t=(ko:string,en:string)=>language==='ko'?ko:en;
    page.on('pageerror',e=>errors.push(e.message));
    const counts=()=>page.evaluate(()=>({...((window as unknown as {__lessonCounts:{created:number;closed:number;active:number}}).__lessonCounts)}));
    const ready=async()=>{await page.waitForSelector('[data-real-time-phase="ready"]',{timeout:120000});await page.waitForFunction(()=>((window as unknown as {__lessonCounts:{active:number}}).__lessonCounts.active===0));};
    await page.goto(url);await ready();const before=await counts();assert.equal(before.active,0);
    for(const [i,guide]of guides.entries()){
      await page.getByLabel(t('안내 실험','Guided experiment'),{exact:true}).selectOption(String(i));assert.equal(await page.locator('[data-real-time-guide-explanation]').count(),0);
      await page.getByLabel(t('내 예측','My prediction'),{exact:true}).fill(`prediction ${i}`);
      await page.getByRole('button',{name:t('해설 보기','Show explanation'),exact:true}).click();assert.equal(await page.locator('[data-real-time-guide-explanation]').innerText(),guide.explanation[language]);
      assert.equal(await page.locator('[data-real-time-preparation]').count(),guide.preparations.length);
      assert(await page.evaluate(()=>document.documentElement.scrollWidth<=document.documentElement.clientWidth+1),'overflow: '+JSON.stringify({language,theme,width,i}));
    }
    for(const [i,q]of questions.entries())for(let answer=0;answer<q.options.length;answer++){
      const field=page.locator('[data-real-time-question]').nth(i);await field.locator('input').nth(answer).check();const feedback=field.locator('[data-real-time-answer]');assert.equal(await feedback.getAttribute('data-real-time-answer'),answer===q.correct?'correct':'incorrect');assert((await feedback.innerText()).includes(q.explanation[language]));
    }
    await page.getByRole('button',{name:t('문제 다시 풀기','Reset answers'),exact:true}).click();assert.equal(await page.locator('[data-real-time-answer]').count(),0);
    await page.getByLabel(t('안내 실험','Guided experiment'),{exact:true}).selectOption('0');assert.equal(await page.getByLabel(t('내 예측','My prediction'),{exact:true}).inputValue(),'prediction 0');
    const selector=page.getByLabel(t('안내 실험','Guided experiment'),{exact:true});await selector.focus();await selector.press('ArrowDown');assert.equal(await selector.inputValue(),'1');
    const radio=page.locator('[data-real-time-question]').first().locator('input').first();await radio.focus();await radio.press('Space');assert.equal(await radio.isChecked(),true);
    const other=language==='ko'?'en':'ko';await page.getByRole('button',{name:other==='en'?'English':'한국어',exact:true}).click();assert.equal(await page.getByLabel(other==='en'?'My prediction':'내 예측',{exact:true}).inputValue(),'prediction 1');await page.getByRole('button',{name:language==='en'?'English':'한국어',exact:true}).click();
    assert.deepEqual(await counts(),before,'Learning controls must not launch Workers');
    const axe=await new AxeBuilder({page}).include('[data-real-time-lessons]').withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();assert.deepEqual(axe.violations.map(v=>({id:v.id,targets:v.nodes.map(n=>n.target)})),[]);
    if(language==='en')assert(!/[가-힣]/.test(await page.locator('main').innerText()));
    let applications=0;const all=language==='en'&&theme==='light'&&width===1440;
    for(const [i,guide]of guides.entries()){
      if(styleOnly||!all&&![0,2,5].includes(i))continue;
      await selector.selectOption(String(i));
      for(const id of all?guide.preparations:[guide.preparations[0]]){
        const start=await counts();await page.locator(`[data-real-time-preparation="${id}"]`).click();
        await page.waitForFunction(id=>document.querySelector('[data-loaded-preparation]')?.getAttribute('data-loaded-preparation')===id,id);await ready();
        assert.deepEqual(JSON.parse((await page.locator('[data-current-input]').getAttribute('data-current-input'))!),preparations.find(p=>p.id===id)!.input);
        const end=await counts();assert.equal(end.created-start.created,2,id);assert.equal(end.active,0);applications++;
      }
    }
    await selector.selectOption('0');await page.locator('[data-real-time-lessons]').screenshot({path:`${root}/lessons-${language}-${theme}-${width}.png`});
    const final=await counts();assert.equal(final.created,final.closed);assert.equal(final.active,0);assert.deepEqual(errors,[]);
    rows.push({language,theme,width,guides:6,questionChoices:24,applications,keyboard:true,predictionsPreserved:true,localePreservesState:true,pureLearningControls:true,accessibility:true,counts:final});writeFileSync(`${root}/native-${styleOnly?'style-':''}progress.json`,JSON.stringify(rows,null,2)+'\n');await context.close();
  }writeFileSync(`${root}/native-${styleOnly?'style':'lessons'}.json`,JSON.stringify({rows},null,2)+'\n');}finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
