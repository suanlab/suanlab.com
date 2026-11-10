import assert from 'node:assert/strict';
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {chromium} from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import {realTimePhaseSteps} from '../src/data/real-time-phase-teaching';
const root='.runtime/learning/real-time-phase-concepts';
const {url}=JSON.parse(readFileSync(root+'/review-server.json','utf8')) as {url:string};
const key='suanlab-learning:real-time-phase-and-classical-limit:concept';
async function main(){
  mkdirSync(root,{recursive:true});const browser=await chromium.launch({headless:true}),rows:unknown[]=[],errors:string[]=[];
  let accessibilityCases=0,conceptChecks=0,keyboardChecks=0,storageChecks=0;
  try{
    for(const language of ['ko','en'] as const)for(const theme of ['light','dark'] as const)for(const width of [320,1440]){
      const context=await browser.newContext({viewport:{width,height:1000},reducedMotion:'reduce'});
      await context.addInitScript(({language,theme})=>{localStorage.setItem('suanlab-language',language);localStorage.setItem('theme',theme);},{language,theme});
      const page=await context.newPage(),label=`${language}-${theme}-${width}`;page.on('pageerror',e=>errors.push(e.message));
      await page.goto(url);await page.waitForFunction(language=>document.documentElement.lang===language,language);
      await page.locator('[data-real-time-concept-ready="true"]').waitFor();await page.evaluate(()=>document.fonts.ready);
      assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),label+' document overflow');
      const previous=page.getByRole('button',{name:language==='ko'?'이전 단계':'Previous step',exact:true});
      const next=page.getByRole('button',{name:language==='ko'?'다음 단계':'Next step',exact:true});assert(await previous.isDisabled());
      for(let step=0;step<realTimePhaseSteps.length;step++){
        const choice=page.locator('.ai-concept-steps button').nth(step);await choice.click();
        const graphic=page.locator('.ai-intuition svg');await page.locator(`[data-real-time-concept="${step}"]`).waitFor();
        assert.equal(await choice.getAttribute('aria-pressed'),'true');assert.equal(await page.locator('.ai-concept-steps [aria-pressed="true"]').count(),1);
        assert((await page.locator('.ai-concept-explanation').innerText()).includes(realTimePhaseSteps[step].body[language]));
        if(language==='en')assert(!/[가-힣]/u.test(await page.locator('main').innerText()));
        assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),`${label} step${step} document overflow`);
        const outside=await graphic.locator('text').evaluateAll(nodes=>nodes.flatMap(node=>{
          const b=(node as SVGGraphicsElement).getBBox();return b.x<0||b.y<0||b.x+b.width>721||b.y+b.height>441?[{text:node.textContent,x:b.x,y:b.y,width:b.width,height:b.height}]:[];
        }));assert.deepEqual(outside,[],`${label} step${step} text outside viewBox`);
        const accessibility=await new AxeBuilder({page}).include('main').withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();
        assert.deepEqual(accessibility.violations.map(x=>({id:x.id,nodes:x.nodes.map(n=>n.target)})),[],`${label} step${step}`);
        accessibilityCases++;conceptChecks++;
        if(width===1440&&theme==='light')await graphic.screenshot({path:`${root}/reference-${language}-${step}.png`});
      }
      assert(await next.isDisabled());await previous.focus();await page.keyboard.press('Enter');
      await page.locator('[data-real-time-concept="11"]').waitFor();keyboardChecks++;
      await page.reload();await page.locator('[data-real-time-concept-ready="true"]').waitFor();
      assert.equal(await page.locator('.ai-intuition').getAttribute('data-real-time-concept'),'11');storageChecks++;
      await page.locator('[data-real-time-cover]').screenshot({path:`${root}/cover-${label}.png`});
      await page.getByRole('button',{name:language==='ko'?'English':'한국어',exact:true}).click();
      await page.waitForFunction(language=>document.documentElement.lang!==language,language);
      assert.equal(await page.locator('.ai-intuition').getAttribute('data-real-time-concept'),'11');
      if(width===320){const region=page.locator('.ai-concept-explanation [role="region"]');await region.focus();await page.keyboard.press('ArrowRight');await page.waitForFunction(()=>document.querySelector('.ai-concept-explanation [role="region"]')!.scrollLeft>0);keyboardChecks++;}
      rows.push({language,theme,width,steps:13,keyboard:true,restoration:true,languageSwitchPreservesStep:true});
      writeFileSync(root+'/native-progress.json',JSON.stringify({rows,conceptChecks,accessibilityCases,keyboardChecks,storageChecks,errors},null,2)+'\n');
      console.log(JSON.stringify({completed:rows.length,label}));await context.close();
    }
    for(const [label,raw] of [['malformed','{'],['negative',JSON.stringify({version:1,value:-1})],['outside',JSON.stringify({version:1,value:13})],['fraction',JSON.stringify({version:1,value:1.5})],['version',JSON.stringify({version:2,value:5})],['string',JSON.stringify({version:1,value:'5'})]] as const){
      const context=await browser.newContext();await context.addInitScript(({key,raw})=>localStorage.setItem(key,raw),{key,raw});const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));await page.goto(url);await page.locator('[data-real-time-concept-ready="true"]').waitFor();assert.equal(await page.locator('.ai-intuition').getAttribute('data-real-time-concept'),'0',label);storageChecks++;await context.close();
    }
    const denied=await browser.newContext();await denied.addInitScript(()=>{Storage.prototype.getItem=()=>{throw Error('storage denied');};Storage.prototype.setItem=()=>{throw Error('storage denied');};});
    const deniedPage=await denied.newPage();deniedPage.on('pageerror',e=>errors.push(e.message));await deniedPage.goto(url);await deniedPage.locator('[data-real-time-concept-ready="true"]').waitFor();await deniedPage.locator('.ai-concept-steps button').nth(8).click();await deniedPage.locator('[data-real-time-concept="8"]').waitFor();storageChecks++;await denied.close();
    const controlled=await browser.newContext();await controlled.addInitScript(key=>localStorage.setItem(key,JSON.stringify({version:1,value:7})),key);const controlledPage=await controlled.newPage();controlledPage.on('pageerror',e=>errors.push(e.message));await controlledPage.goto(url+'?controlled=1');await controlledPage.locator('[data-real-time-concept="0"]').waitFor();await controlledPage.locator('.ai-concept-steps button').nth(12).click();await controlledPage.locator('[data-real-time-concept="12"]').waitFor();assert.equal(await controlledPage.evaluate(key=>JSON.parse(localStorage.getItem(key)!).value,key),7);await controlled.close();
    assert.equal(conceptChecks,104);assert.equal(accessibilityCases,104);assert.deepEqual(errors,[]);
    const report={rows,conceptChecks,accessibilityCases,keyboardChecks,storageChecks,controlledStateVerified:true,errors,previewOnly:true,readerRegistered:false,browserVersion:browser.version()};writeFileSync(root+'/native-concepts.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
  }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
