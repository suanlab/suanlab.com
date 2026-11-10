/** Focused final-format checks: unavailable output versus undefined derivative. */
import assert from 'node:assert/strict';
import {writeFileSync} from 'node:fs';
import {chromium} from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import {phaseKinds,phasePresetInput as preset} from '../src/lib/learning-phase-transitions';
import {phaseInputKey as key} from '../src/lib/learning-phase-transitions-protocol';
async function main(){
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']}),cases:unknown[]=[],errors:string[]=[];
 const conditions=[{index:1,preset:0,cap:false,fold:false},{index:1,preset:3,cap:true,fold:false},{index:2,preset:4,cap:false,fold:true},{index:2,preset:11,cap:true,fold:false},{index:1,preset:3,cap:false,fold:false,zoom:true},{index:0,preset:8,cap:false,fold:false,finite:true}];
 try{for(const condition of conditions)for(const language of ['ko','en'] as const)for(const dark of [false,true])for(const width of [320,1440]){
  const context=await browser.newContext({viewport:{width,height:900},reducedMotion:'reduce'}),page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));await page.goto(`http://localhost:4291/?kind=${condition.index}`);await page.getByRole('button',{name:language==='ko'?'KO':'EN',exact:true}).click();if(dark)await page.getByRole('button',{name:'Theme',exact:true}).click();const lab=page.locator('[data-phase-lab]');await lab.locator('fieldset.ai-presets button').nth(condition.preset).click();if(condition.cap)await lab.getByRole('combobox').selectOption('128');
  const input=preset(phaseKinds[condition.index],condition.preset),expected=key(condition.cap?{...input,options:{minimumPrecision:128,maximumPrecision:128}}:input);
  await page.waitForFunction(k=>document.querySelector('[data-phase-lab]')?.getAttribute('data-phase-current')===k,expected,{timeout:60000});await page.locator('[data-phase-concept-state="ready"]').waitFor({timeout:60000});const body=await lab.innerText();assert.doesNotMatch(body,/Undefined/);if(language==='en')assert.doesNotMatch(await page.locator('body').innerText(),/[가-힣\u3400-\u9fff]/);
  assert.ok(('zoom'in condition&&condition.zoom)||('finite'in condition&&condition.finite)||body.includes(condition.fold?(language==='ko'?'단일값 미분 없음':'No single-valued derivative'):(language==='ko'?'미제공':'Not available')));assert.equal(await page.evaluate(()=>document.documentElement.classList.contains('dark')),dark);assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  if(condition.index!==0){await lab.getByRole('button',{name:language==='ko'?'도함수':'Derivative',exact:true}).click();
  const roles=lab.getByRole('region',{name:language==='ko'?'모든 정지근의 역할':'Roles of all stationary roots',exact:true});
  const rows=roles.locator('tbody tr');if(await rows.count()){
   const expectedM=await rows.last().locator('td').first().innerText();await lab.getByRole('group',{name:language==='ko'?'상세 근 선택':'Choose a root for details',exact:true}).getByRole('button').last().click();
   const readout=lab.getByRole('region',{name:language==='ko'?'선택한 근과 비교 경계':'Selected root and comparison thresholds',exact:true});assert.equal(await readout.getByRole('rowheader',{name:'m',exact:true}).locator('..').locator('td').first().innerText(),expectedM);
   await lab.getByRole('button',{name:language==='ko'?'근 확대 1':'Zoom root 1',exact:true}).click();
  }
  if('zoom'in condition&&condition.zoom){assert.equal(await lab.locator('[data-phase-landscape]').getAttribute('data-phase-drawing'),'unresolved');assert.equal(await rows.count(),3);}
  }else{const table=lab.getByRole('region',{name:language==='ko'?'유한 거시상태의 확률과 다중도':'Finite macrostate probabilities and multiplicities',exact:true});const values=new Set<string>();for(;;){for(const value of await table.locator('tbody th').allTextContents())values.add(value);const next=lab.getByRole('button',{name:language==='ko'?'다음 상태 행':'Next state rows',exact:true});if(await next.isDisabled())break;await next.click();}assert.equal(values.size,33);await lab.getByRole('button',{name:language==='ko'?'이전 상태 행':'Previous state rows',exact:true}).click();}
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  const report=await new AxeBuilder({page}).withTags(['wcag2a','wcag2aa','wcag21a','wcag21aa']).analyze();assert.deepEqual(report.violations.map(v=>({id:v.id,nodes:v.nodes.map(n=>n.target)})),[]);
  if(language==='en'&&dark&&width===1440&&condition.fold)await lab.locator('[data-phase-landscape]').screenshot({path:'.runtime/learning/phase-transitions-teaching/final-landscape.png'});
  cases.push({...condition,language,dark,width,axeViolations:0,overflow:false});console.log(JSON.stringify(cases.at(-1)));await context.close();
 }assert.deepEqual(errors,[]);writeFileSync('.runtime/learning/phase-transitions-teaching/view-state-verification.json',JSON.stringify({scope:'final null-value and fold formatting on native component harness',cases,errors},null,2));}finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
