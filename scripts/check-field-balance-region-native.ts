/** Check user edits and region selection against worked examples in the actual review UI. */
import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {chromium} from 'playwright';
const root='.runtime/learning/field-balance-teaching';
async function main(){
 const base=(JSON.parse(readFileSync(root+'/review-server.json','utf8'))as{url:string}).url,browser=await chromium.launch({headless:true}),states=[];
 try{for(const language of ['ko','en']as const){const context=await browser.newContext({viewport:{width:390,height:900},reducedMotion:'reduce'});await context.addInitScript(lang=>localStorage.setItem('suanlab-language',lang),language);const page=await context.newPage(),errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));await page.goto(base);
  const ready=()=>page.locator('[data-balance-phase="ready"]').waitFor(),t=(ko:string,en:string)=>language==='ko'?ko:en,region=page.getByLabel(t('구간의 마지막 점','Last site in region'),{exact:true}),apply=page.getByRole('button',{name:t('입력 적용','Apply input'),exact:true}),share=page.getByLabel(t('왼쪽 에너지 몫 α','Left energy share α'),{exact:true});
  await ready();assert((await page.locator('[data-balance-prefix]').innerText()).includes('dE/dt=-1'));await region.selectOption('0');assert((await page.locator('[data-balance-prefix]').innerText()).includes('dE/dt=-3/2'));await region.selectOption('2');assert((await page.locator('[data-balance-prefix]').innerText()).includes('dE/dt=0'));
  await share.fill('0');await apply.click();await ready();assert.equal(await page.locator('[data-balance-metric=H]').innerText(),'7/2');assert.equal(await page.locator('table').first().locator('tbody tr').first().locator('td').first().innerText(),'5/2');
  await share.fill('2');await apply.click();assert.equal(await page.getByRole('alert').count(),1);assert.equal(await page.locator('[data-balance-metric=H]').innerText(),'7/2');
  await share.fill('1/2');await apply.click();await ready();assert.equal(await page.getByRole('alert').count(),0);assert.equal(await page.locator('table').first().locator('tbody tr').first().locator('td').first().innerText(),'7/4');
  const examples=page.getByLabel(t('전체 예제','All examples'),{exact:true});await examples.selectOption('quartic-cyclic');await ready();assert.equal(await page.locator('[data-balance-metric=Pdot]').innerText(),'3');assert.equal(await page.locator('[data-balance-metric=H]').innerText(),'29/4');await examples.selectOption('fixed-instant-zero');await ready();assert.equal(await page.locator('[data-balance-metric=Qdot]').innerText(),'0');assert.equal(await page.locator('[data-balance-metric=Qddot]').innerText(),'-1');assert.equal(await page.locator('[data-balance-metric=Pdot]').count(),0);
  assert.deepEqual(errors,[]);states.push({language,regionEndpointsChecked:3,customAllocationChecked:true,invalidAllocationRetainsAppliedResult:true,recoveryChecked:true,quarticCounterexampleChecked:true,fixedBoundaryGeneratorAbsent:true,errors});await context.close();
 }writeFileSync(root+'/region-native.json',JSON.stringify({states,readerRegistered:false,productionNextAssetsVerified:false},null,2)+'\n');console.log(JSON.stringify({regionCases:states.length}));}finally{await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
