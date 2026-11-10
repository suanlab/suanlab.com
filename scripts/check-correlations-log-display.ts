/** Supplemental final-format QA after improving near-one log projections. */
import assert from 'node:assert/strict';
import {writeFileSync} from 'node:fs';
import {chromium} from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import {correlationKinds,correlationPresetInput} from '../src/lib/learning-correlations';
import {correlationInputKey} from '../src/lib/learning-correlations-protocol';
async function main(){const browser=await chromium.launch({headless:true,args:['--no-sandbox']}),cases:unknown[]=[],errors:string[]=[];
 try{for(const language of ['ko','en'] as const)for(const dark of [false,true])for(const width of [320,1440]){
  const context=await browser.newContext({viewport:{width,height:1000},reducedMotion:'reduce'}),page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));await page.goto('http://localhost:4293/?kind=2');await page.getByRole('button',{name:language==='ko'?'KO':'EN',exact:true}).click();if(dark)await page.getByRole('button',{name:'Theme',exact:true}).click();
  const lab=page.locator('[data-correlation-lab]');await lab.locator('[data-correlation-result]').waitFor();await lab.locator('.ai-presets button').nth(5).click();await page.waitForFunction(expected=>document.querySelector('[data-correlation-lab]')?.getAttribute('data-correlation-current')===expected,correlationInputKey(correlationPresetInput(correlationKinds[2],5)));
  const region=lab.getByRole('region',{name:language==='ko'?'모든 지연의 자기공분산':'Autocovariances at all lags',exact:true});const rows=region.locator('tbody tr');assert.equal(await rows.count(),32);
  assert.equal(await rows.nth(0).locator('td').last().innerText(),'0');for(let i=1;i<32;i++){const log=Number(await rows.nth(i).locator('td').last().innerText());assert.ok(Number.isFinite(log)&&log<=0,`lag ${i}: ${log}`);}
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));if(language==='en')assert.doesNotMatch(await page.locator('body').innerText(),/[가-힣]/);
  const axe=await new AxeBuilder({page}).withTags(['wcag2a','wcag2aa','wcag21a','wcag21aa']).analyze();assert.deepEqual(axe.violations.map(v=>v.id),[]);
  if(language==='en'&&dark&&width===1440)await lab.screenshot({path:'.runtime/learning/correlations-teaching/final-near-one-log.png'});cases.push({language,dark,width,exactOneLogZero:true,nearOneProjectedLogsNonpositive:true,axeViolations:0,overflow:false});await context.close();
 }assert.deepEqual(errors,[]);writeFileSync('.runtime/learning/correlations-teaching/log-display-verification.json',JSON.stringify({scope:'final near-one display projection only; prior full matrix and boundary controls retain separate evidence',cases,errors},null,2)+'\n');console.log(JSON.stringify({cases:cases.length,errors}));
 }finally{await browser.close();}}
main().catch(error=>{console.error(error);process.exitCode=1;});
