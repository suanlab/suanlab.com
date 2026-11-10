/** Final display controls/boundaries in the standalone native teaching harness. */
import assert from 'node:assert/strict';
import {writeFileSync} from 'node:fs';
import {chromium} from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import {correlationKinds as kinds,correlationPresetInput as preset} from '../src/lib/learning-correlations';
import {correlationInputKey as key} from '../src/lib/learning-correlations-protocol';
const folder='.runtime/learning/correlations-teaching',base='http://localhost:4293';
const conditions=[{index:0,preset:5,id:'source-vector'},{index:0,preset:9,id:'precision-cap'},{index:1,preset:8,id:'distant-underflow'},{index:2,preset:7,id:'endpoint-log-zero'},{index:2,preset:5,id:'near-constant-axis'},{index:2,preset:10,id:'maximum-lag-pages'}];
async function main(){const browser=await chromium.launch({headless:true,args:['--no-sandbox']}),cases:unknown[]=[],errors:string[]=[];
 try{for(const condition of conditions)for(const language of ['ko','en'] as const)for(const dark of [false,true])for(const width of [320,1440]){
  const context=await browser.newContext({viewport:{width,height:1000},reducedMotion:'reduce'}),page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
  await page.goto(`${base}/?kind=${condition.index}`);await page.getByRole('button',{name:language==='ko'?'KO':'EN',exact:true}).click();if(dark)await page.getByRole('button',{name:'Theme',exact:true}).click();
  const lab=page.locator('[data-correlation-lab]');await lab.locator('[data-correlation-result]').waitFor({timeout:60000});await lab.locator('.ai-presets button').nth(condition.preset).click();await page.waitForFunction(expected=>document.querySelector('[data-correlation-lab]')?.getAttribute('data-correlation-current')===expected,key(preset(kinds[condition.index],condition.preset)),{timeout:60000});
  if(condition.id==='source-vector'){
   const image=lab.getByRole('img',{name:language==='en'?/^Direction and scale/:/^관측과 대응/});assert.equal(await image.locator('path[stroke="#ea580c"]').getAttribute('d'),'M260 160L336 198');
  }else if(condition.id==='precision-cap'){
   assert.equal(await lab.locator('[data-correlation-result]').getAttribute('data-correlation-resolved'),'false');assert.match(await lab.getByRole('row',{name:/^C12 = C21/}).innerText(),language==='ko'?/미해결/:/unresolved/i);
  }else if(condition.id==='distant-underflow'){
   const slider=lab.locator('input[type=range]');await slider.focus();await slider.press('Home');for(let i=0;i<64;i++)await slider.press('ArrowRight');assert.equal(await slider.inputValue(),'64');
   assert.match(await lab.getByRole('row',{name:language==='en'?/^Selected-site correlation/:/^선택 위치의 상관/}).innerText(),language==='ko'?/표시 범위 밖/:/outside display range/);
   const next=lab.getByRole('button',{name:language==='ko'?'다음 관측 행':'Next observation rows',exact:true});for(let i=0;i<3;i++)await next.click();assert.equal(await lab.getByLabel(language==='ko'?'관측 표 페이지':'Observation-table page').innerText(),'4/4');
  }else if(condition.id==='endpoint-log-zero'){
   await lab.getByRole('button',{name:language==='ko'?'분산 로그 축':'Log variance axis',exact:true}).click();
   const plot=lab.getByRole('img',{name:language==='en'?/^Mean uncertainty by observation count/:/^관측 수에 따른 평균 오차/});assert.equal(await plot.locator('circle[cy="251"]').count(),2);
   assert.equal(await lab.getByRole('button',{name:language==='ko'?'점근 참조 표시':'Show asymptotic reference',exact:true}).isDisabled(),true);
   assert.match(await lab.getByRole('row',{name:language==='en'?/^Ordinary summed IAT/:/^보통 합 IAT/}).innerText(),language==='ko'?/미제공/:/Not provided/);assert.match(await lab.getByRole('row',{name:language==='en'?/^Variance-equivalent count/:/^분산 등가 수/}).innerText(),/\+∞/);
  }else if(condition.id==='near-constant-axis'){
   await lab.getByRole('button',{name:language==='ko'?'분산 로그 축':'Log variance axis',exact:true}).click();await lab.getByRole('button',{name:language==='ko'?'점근 참조 표시':'Show asymptotic reference',exact:true}).click();
   assert.equal(await lab.getByText(language==='ko'?'확대된 축의 범위':'Magnified axis range',{exact:false}).count(),1);
  }else{
   const image=lab.getByRole('img',{name:language==='en'?/^Stationary autocovariance window/:/^정상 자기공분산 창/});assert.equal(await image.locator('rect').count(),64);
   const next=lab.getByRole('button',{name:language==='ko'?'다음 관측 행':'Next observation rows',exact:true});for(let table=0;table<2;table++)for(let i=0;i<15;i++)await next.nth(table).click();
   assert.deepEqual(await lab.getByLabel(language==='ko'?'관측 표 페이지':'Observation-table page').allInnerTexts(),['16/16','16/16']);
   const lags=lab.getByRole('region',{name:language==='ko'?'모든 지연의 자기공분산':'Autocovariances at all lags',exact:true});await lags.getByRole('rowheader',{name:'511',exact:true}).waitFor();
  }
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));if(language==='en')assert.doesNotMatch(await page.locator('body').innerText(),/[가-힣]/);
  const axe=await new AxeBuilder({page}).withTags(['wcag2a','wcag2aa','wcag21a','wcag21aa']).analyze();assert.deepEqual(axe.violations.map(v=>({id:v.id,nodes:v.nodes.map(n=>n.target)})),[]);
  if(language==='en'&&dark&&width===1440)await lab.screenshot({path:`${folder}/final-${condition.id}.png`});cases.push({...condition,language,dark,width,axeViolations:0,overflow:false});console.log(JSON.stringify(cases.at(-1)));await context.close();
 }assert.deepEqual(errors,[]);writeFileSync(`${folder}/view-state-verification.json`,JSON.stringify({scope:'final teaching boundary display, plot controls and pagination; not registered readers',cases,errors},null,2)+'\n');
 }finally{await browser.close();}}
main().catch(error=>{console.error(error);process.exitCode=1;});
