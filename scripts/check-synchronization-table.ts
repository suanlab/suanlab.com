/** Inspect the final exported mathematical table after the Markdown pipe fix. */
import {chromium} from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import assert from 'node:assert/strict';
import {writeFileSync} from 'node:fs';
async function main(){
 const browser=await chromium.launch({headless:true}),report={cases:0,axeCases:0,pageErrors:[]as string[]};
 try{for(const language of ['ko','en'])for(const theme of ['light','dark'])for(const width of [390,1600]){
  const c=await browser.newContext({viewport:{width,height:1000},colorScheme:theme as 'light'|'dark',reducedMotion:'reduce'});await c.addInitScript(({language,theme})=>{localStorage.setItem('suanlab-language',language);localStorage.setItem('theme',theme);},{language,theme});const p=await c.newPage();p.on('pageerror',e=>report.pageErrors.push(e.message));await p.goto('http://localhost:4175/book/online/physics-math-for-ai/hopf-amplitude-critical-relaxation/');await p.waitForFunction(l=>document.documentElement.lang===l,language);await p.locator('[data-synchronization-lab] [data-synchronization-result]').waitFor();
  for(const formula of ['1/|μ|','1/(2|μ|)']){const cell=p.getByRole('cell',{name:formula,exact:true});assert.equal(await cell.count(),1);assert.equal(await cell.locator('xpath=..').locator('td, th').count(),3);}
  assert(!(await p.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1)));const axe=await new AxeBuilder({page:p}).include('main').withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();assert.deepEqual(axe.violations.map(v=>v.id),[]);report.axeCases++;report.cases++;await c.close();
 }assert.deepEqual(report.pageErrors,[]);writeFileSync('.runtime/learning/synchronization-interface/table-browser-verification.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));}finally{await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
