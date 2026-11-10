import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {chromium} from 'playwright';
import AxeBuilder from '@axe-core/playwright';
const root='.runtime/learning/real-time-phase-waves',url=JSON.parse(readFileSync(root+'/review-server.json','utf8')).url as string;
async function main(){
  const browser=await chromium.launch({headless:true});const errors:string[]=[];
  try{
    const context=await browser.newContext();const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
    await page.addInitScript({content:`globalThis.__name=fn=>fn;localStorage.setItem('suanlab-language','en');window.__ports={created:0,terminated:0,active:new Set()};const Native=Worker;window.Worker=class extends Native{constructor(url,options){super(url,options);window.__ports.created++;window.__ports.active.add(this);}terminate(){if(window.__ports.active.delete(this)){window.__ports.terminated++;super.terminate();}}};`});
    await page.goto(url);await page.waitForFunction(()=>document.documentElement.lang==='en');
    const ready=async()=>{await page.locator('[data-real-time-phase="ready"]').waitFor({timeout:120000});};await ready();
    if(process.argv.includes('--zero-guide-only')){
      const figure=page.getByText('Lattice positions of a stationary path',{exact:true}).locator('..');assert.equal(await figure.locator('[data-real-time-zero-guide]').count(),1);await figure.screenshot({path:root+'/signed-stationary-path.png'});
      const audit=await new AxeBuilder({page}).include('main').withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();assert.deepEqual(audit.violations,[]);assert.deepEqual(errors,[]);writeFileSync(root+'/native-zero-guide.json',JSON.stringify({zeroGuide:true,accessibility:true,errors},null,2)+'\n');return;
    }
    const choose=async(id:string)=>{await page.getByLabel('All examples',{exact:true}).selectOption(id);await ready();};
    await choose('maximum-negative-physical-control');const original=page.getByText('Original current Airy integral and scaled/log intervals',{exact:true});await original.click();
    const values=JSON.parse(await original.locator('..').locator('pre').innerText());assert.equal(values.rawValueBelowObservationQuantum,true);assert.equal(values.scaledSign,'positive');assert(values.logMagnitude);assert.notEqual(values.scaledIntegral.upper,'0');
    await page.locator('[data-real-time-complex-saddles]').screenshot({path:root+'/complex-saddles-extreme.png'});
    await choose('minimum-width');const packet=page.locator('[data-real-time-wave="packet"]');assert.equal(await packet.locator('figure').first().locator('circle').count(),65);await packet.locator('figure').first().screenshot({path:root+'/minimum-width-density.png'});
    await page.getByRole('button',{name:'Calculate both families',exact:true}).click();await page.waitForFunction('window.__ports.active.size>0');await choose('merging-stationary-points');assert.equal(await page.locator('[data-real-time-comparison]').count(),0);await page.waitForFunction('window.__ports.active.size===0');
    await page.getByRole('button',{name:'Calculate 33 Airy samples',exact:true}).click();await page.locator('[data-real-time-sweep="ready"]').waitFor({timeout:600000});const figure=page.locator('[data-real-time-wave="fold"] figure').last();
    assert.equal(await figure.locator('circle').count(),49);await figure.screenshot({path:root+'/airy-sweep-scaled.png'});
    await page.getByLabel('Airy display',{exact:true}).selectOption('log');assert.equal(await figure.locator('circle').count(),33);await figure.screenshot({path:root+'/airy-sweep-log.png'});
    await page.getByLabel('Airy display',{exact:true}).selectOption('raw');assert.equal(await figure.locator('circle').count(),49);await figure.screenshot({path:root+'/airy-sweep-raw.png'});
    await page.waitForFunction('window.__ports.active.size===0');const ports=await page.evaluate('({created:window.__ports.created,terminated:window.__ports.terminated})') as {created:number;terminated:number};assert.equal(ports.created,ports.terminated);assert.deepEqual(errors,[]);
    writeFileSync(root+'/native-extremes.json',JSON.stringify({extremeShadowKeepsScaledAndLog:true,complexSaddles:true,minimumWidthSamples:65,inputReplacementCancelsComparison:true,airySamples:33,scaledAndRawPoints:49,logPoints:33,ports,errors},null,2)+'\n');console.log(JSON.stringify({ports,errors}));
  }finally{await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
