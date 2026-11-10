import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {chromium} from 'playwright';
import AxeBuilder from '@axe-core/playwright';
const root='.runtime/learning/real-time-phase-waves',url=JSON.parse(readFileSync(root+'/review-server.json','utf8')).url as string;
async function main(){
  const browser=await chromium.launch({headless:true}),rows:unknown[]=[],errors:string[]=[];let accessibilityCases=0;
  try{
    for(const language of ['ko','en'] as const)for(const theme of ['light','dark'] as const)for(const width of [320,1440]){
      const context=await browser.newContext({viewport:{width,height:1000},reducedMotion:'reduce'});
      await context.addInitScript({content:`globalThis.__name=fn=>fn;window.__waveStats={created:0,terminated:0,active:new Set()};const Native=Worker;window.Worker=class extends Native{constructor(url,options){super(url,options);window.__waveStats.created++;window.__waveStats.active.add(this);}terminate(){if(window.__waveStats.active.delete(this)){window.__waveStats.terminated++;super.terminate();}}};localStorage.setItem('suanlab-language',${JSON.stringify(language)});localStorage.setItem('theme',${JSON.stringify(theme)});`});
      const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));await page.goto(url);await page.waitForFunction(l=>document.documentElement.lang===l,language);
      const audit=async(mode:string)=>{const a=await new AxeBuilder({page}).include('main').withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();assert.deepEqual(a.violations.map(v=>({id:v.id,nodes:v.nodes.map(n=>n.target)})),[],mode);accessibilityCases++;assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),mode+' document overflow');};
      const ready=async()=>{await page.locator('[data-real-time-phase="ready"]').waitFor({timeout:120000});};await ready();
      const examples=page.getByLabel(language==='ko'?'전체 예제':'All examples',{exact:true});
      const choose=async(id:string)=>{await examples.selectOption(id);await ready();};
      await choose('caustic-compatible');assert((await page.locator('[data-real-time-stationary]').innerText()).includes('affine-family'));assert(await page.locator('[data-real-time-delta]').count());
      const nullInput=page.getByLabel(language==='ko'?'영공간 계수':'Null-direction coefficient',{exact:true});await nullInput.fill('1');assert.equal(await nullInput.inputValue(),'1');
      await choose('caustic-incompatible');assert((await page.locator('[data-real-time-stationary]').innerText()).includes('no-stationary-path'));
      await choose('zero-leading-pivot-invertible');await page.getByLabel(language==='ko'?'모드 번호':'Mode number',{exact:true}).fill('2');assert((await page.locator('[data-real-time-wave="chain"]').innerText()).includes('positive'));
      await audit('chain');await page.locator('[data-real-time-wave="chain"]').screenshot({path:`${root}/chain-${language}-${theme}-${width}.png`});
      await choose('fixed-width-free');const packet=page.locator('[data-real-time-wave="packet"]');assert.equal(await packet.locator('figure').nth(0).locator('circle').count(),65);assert.equal(await packet.locator('figure').nth(1).locator('circle').count(),130);
      await page.getByLabel(language==='ko'?'파동묶음 표본':'Packet sample',{exact:true}).fill('64');
      await page.getByRole('button',{name:language==='ko'?'두 족 계산':'Calculate both families',exact:true}).click();await page.locator('[data-real-time-comparison="ready"]').waitFor({timeout:240000});
      const comparison=page.locator('[data-real-time-comparison]');assert.equal(await comparison.locator('figure').count(),2);
      await page.getByLabel(language==='ko'?'비교할 ℏ':'Comparison hbar',{exact:true}).selectOption('1/8');await page.locator('[data-real-time-comparison="ready"]').waitFor({timeout:240000});
      await audit('packet');await page.locator('[data-real-time-wave="packet"]').screenshot({path:`${root}/packet-${language}-${theme}-${width}.png`});
      await choose('merging-stationary-points');assert((await page.locator('[data-real-time-fold-regime]').innerText()).includes('merged-stationary-point'));
      if(language==='en'&&theme==='light'&&width===1440){
        await page.getByRole('button',{name:'Calculate 33 Airy samples',exact:true}).click();await page.waitForFunction('window.__waveStats.active.size>0');await page.getByRole('button',{name:'Cancel sampling',exact:true}).click();await page.waitForFunction('window.__waveStats.active.size===0');
        await page.getByRole('button',{name:'Calculate 33 Airy samples',exact:true}).click();await page.locator('[data-real-time-sweep="ready"]').waitFor({timeout:600000});
        assert((await page.locator('[data-real-time-sweep]').innerText()).includes('33/33'));
        await page.getByLabel('Airy display',{exact:true}).selectOption('log');await page.getByLabel('Airy display',{exact:true}).selectOption('raw');await page.getByLabel('Airy display',{exact:true}).selectOption('scaled');
        await page.screenshot({path:root+'/fold-sweep.png'});
      }
      await audit('fold');
      assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'document overflow');if(language==='en')assert(!/[가-힣]/u.test(await page.locator('main').innerText()));
      await choose('no-real-stationary-points');assert((await page.locator('[data-real-time-fold-regime]').innerText()).includes('no-real-stationary-points'));
      await page.screenshot({path:`${root}/waves-${language}-${theme}-${width}.png`});await page.waitForFunction('window.__waveStats.active.size===0');
      const ports=await page.evaluate('({created:window.__waveStats.created,terminated:window.__waveStats.terminated})') as {created:number;terminated:number};assert.equal(ports.created,ports.terminated);
      rows.push({language,theme,width,affineAndIncompatibleFocus:true,modeSigns:true,packetSamples:65,widthComparisons:2,accessibility:true,ports});writeFileSync(root+'/native-progress.json',JSON.stringify({rows,errors},null,2)+'\n');console.log(JSON.stringify({completed:rows.length,language,theme,width}));await context.close();
    }
    assert.equal(rows.length,8);assert.deepEqual(errors,[]);writeFileSync(root+'/native-waves.json',JSON.stringify({rows,errors,accessibilityCases,sequentialAirySamples:33,sweepCancellationAndResume:true,previewOnly:true},null,2)+'\n');
  }finally{await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
