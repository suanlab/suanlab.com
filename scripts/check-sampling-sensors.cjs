const assert = require('node:assert/strict');
const fs = require('node:fs');
const {execFileSync} = require('node:child_process');
const {chromium} = require('playwright');
const AxeBuilder = require('@axe-core/playwright').default;
const base = process.env.SUANLAB_TEST_BASE || 'http://127.0.0.1:4175';
const dir = '.runtime/learning/sampling-sensors-browser';
const data = JSON.parse(execFileSync(process.cwd()+'/node_modules/.bin/tsx',['-e',"import{wavesKinds}from './src/lib/learning-waves';import{learningUnits,wavesLabs}from './src/data/interactive-book';console.log(JSON.stringify({ids:wavesKinds.slice(16,18),units:learningUnits.filter(u=>wavesKinds.slice(16,18).includes(u.id as any)),labs:wavesLabs}))"],{encoding:'utf8'}));
const regression = ['coherent-fourier-pupil-spatial-filtering','incoherent-psf-otf-image-formation'];
(async()=>{
 fs.mkdirSync(dir,{recursive:true});
 const browser = await chromium.launch({executablePath:process.env.SUANLAB_BROWSER_PATH||'/usr/bin/google-chrome',args:['--no-sandbox']});
 const rows=[],interactions=[],guides=[],errors=[],screenshots=[];
 const save=()=>fs.writeFileSync(dir+'/report.json',JSON.stringify({rows,interactions,guides,errors,screenshots},null,2));
 async function go(page,id,lang,dark){
  assert.equal((await page.goto(base+'/book/online/physics-math-for-ai/'+id+'/')).status(),200);
  await page.waitForFunction(({lang,dark})=>document.documentElement.lang===lang&&document.documentElement.classList.contains('dark')===dark&&document.querySelector('.ai-presets')?.disabled===false,{lang,dark});
  await page.evaluate(()=>document.fonts.ready);
 }
 async function audit(page,id,lang,dark,preset='default'){
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),id+': root overflow');
  if(lang==='en')assert.doesNotMatch(await page.locator('.ai-book').innerText(),/[가-힣]/);
  const bad=await page.locator('.ai-lab svg text').evaluateAll(nodes=>nodes.flatMap(n=>{const b=n.getBBox(),v=n.ownerSVGElement.viewBox.baseVal;return b.x<-.1||b.x+b.width>v.width+.1||b.y<-.1||b.y+b.height>v.height+.1?[n.textContent]:[];}));
  assert.deepEqual(bad,[],id+': SVG labels fit');
  assert.ok(await page.locator('.ai-lab svg path').evaluateAll(nodes=>nodes.every(n=>!/(NaN|Infinity)/.test(n.getAttribute('d')||''))));
  const axe=await new AxeBuilder({page}).include('.ai-book').withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();
  assert.deepEqual(axe.violations.map(v=>({id:v.id,targets:v.nodes.map(n=>n.target)})),[],id+' '+lang+' '+preset);
  rows.push({id,lang,dark,width:dark?1440:320,preset,violations:0});save();
 }
 async function wrapping(page){return page.locator('.ai-guided-lab p').evaluateAll(nodes=>nodes.flatMap(p=>{const b=p.getBoundingClientRect(),walker=document.createTreeWalker(p,NodeFilter.SHOW_TEXT);let n;while(n=walker.nextNode()){const range=document.createRange();range.selectNodeContents(n);if([...range.getClientRects()].some(r=>r.width>0&&(r.left<b.left-1||r.right>b.right+1)))return[p.textContent];}return[];}));}
 async function capture(page,id,lang,dark){
  const svg=page.locator('.ai-lab svg[data-waves-plot]');
  for(let i=0;i<await svg.count();i++){
   const marker=await svg.nth(i).getAttribute('data-waves-plot'),path=`${dir}/${id}-${lang}-${dark?'dark':'light'}-${marker}.png`;
   await svg.nth(i).screenshot({path});screenshots.push(path);
  }
  save();
 }
 try{
  for(const dark of [false,true])for(const lang of ['en','ko']){
   const context=await browser.newContext({viewport:{width:dark?1440:320,height:1100},reducedMotion:'reduce'});
   await context.addInitScript(({lang,dark,origin})=>{if(location.origin!==origin)return;localStorage.setItem('suanlab-language',lang);localStorage.setItem('theme',dark?'dark':'light');},{lang,dark,origin:new URL(base).origin});
   await context.route('**/*',route=>new URL(route.request().url()).origin===new URL(base).origin?route.continue():route.abort());
   const page=await context.newPage();page.on('pageerror',e=>errors.push({url:page.url(),message:e.message}));
   for(const id of [...data.ids,...regression]){
    await go(page,id,lang,dark);await audit(page,id,lang,dark);
    await page.locator('.ai-unit-outline summary').click();
    assert.ok(await page.locator('.ai-unit-outline a').evaluateAll(nodes=>nodes.every(a=>document.getElementById(decodeURIComponent(a.hash.slice(1))))));
    await page.locator('.ai-unit-outline summary').click();
    if(!data.ids.includes(id))continue;
    await capture(page,id,lang,dark);
    for(let i=0;i<4;i++){
     const button=page.locator('.ai-concept-steps button').nth(i);await button.click();assert.equal(await button.getAttribute('aria-pressed'),'true');
    }
    for(let i=0;i<3;i++){
     await page.locator('.ai-guided-choices button').nth(i).click();
     await page.getByRole('button',{name:lang==='en'?'Load this example':'이 예제 열기',exact:true}).click();
     const summary=page.locator('.ai-guided-answer summary');
     if(!(await page.locator('.ai-guided-answer').evaluate(n=>n.open)))await summary.click();
     assert.deepEqual(await wrapping(page),[],id+': guide text wraps');
     guides.push({id,lang,dark,guide:i,textRectanglesFit:true});save();
    }
    if(!dark){
     const spec=data.labs[id];
     for(let i=0;i<spec.presets.length;i++){
      await page.locator('.ai-presets button').nth(i).click();
      assert.deepEqual(await page.locator('.ai-parameters input').evaluateAll(nodes=>nodes.map(n=>Number(n.value))),spec.presets[i].values);
      const readText=async key=>page.locator('.ai-readout dl > div').filter({has:page.locator('dt').filter({hasText:new RegExp('^'+key.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')+'$')})}).locator('dd').innerText();
      const read=async key=>Number((await readText(key)).replaceAll(',',''));
      const near=async(key,value,tol=.00051)=>assert.ok(Math.abs(await read(key)-value)<tol,key);
      if(id===data.ids[0]){
       if(i===0)await near('H spatial · 1',.8583936913);
       if(i===1)assert.equal(await read('Nyquist · cycles/mm'),50);
       if(i===2)assert.equal(await read('Nyquist · cycles/mm'),25);
       if(i===3)assert.equal(await read('alias · cycles/mm'),-20);
       if(i===4)assert.ok(await read('H spatial · 1')<0);
       if(i===5)await near('sampling margin · cycles/mm',0,1e-10);
       if(i===7){await near('H temporal · 1',0,1e-14);assert.ok(await read('energy query · aJ')>0);}
       if([8,9].includes(i)){assert.equal(await readText('box mean query · mW/m²'),'—');assert.equal(await read('energy query · aJ'),0);assert.equal(await read('photons query · 1'),0);}
       if(i===10)assert.equal(await read('limit query · mW/m²'),0);
       if(i===13)assert.equal(await read('sample count · 1'),5);
      }else{
       if(i===0)await near('photons · 1',692.191028037);
       if(i===1)await near('photons · 1',1384.382056074);
       if(i===2)await near('photons · 1',692.191028037);
       if(i===4)assert.equal(await read('preADC SNR · 1'),5);
       if(i===5)assert.equal(await read('preADC SNR · 1'),10);
       if(i===6){await near('stored mean · e',12);await near('ADC mean · DN',12);assert.equal(await read('ADC clip high · 1'),0);}
       if(i===7)await near('ADC mean · DN',15);
       if(i===8){assert.equal(await read('ADC mean · DN'),8);assert.equal(await read('ADC variance · DN²'),0);assert.equal(await read('P query code · 1'),1);}
       if(i===9){assert.equal(await read('ADC clip high · 1'),0);assert.equal(await read('P query code · 1'),1);}
       if(i===10)assert.equal(await read('photons · 1'),0);
       if(i===11){assert.equal(await read('photo mean · e'),0);assert.equal(await read('dark mean · e'),.2);}
       if(i===12)assert.equal(await read('ADC mean · DN'),0);
      }
      await audit(page,id,lang,dark,i);
     }
     const continuous=page.locator('.ai-parameters input').nth(id===data.ids[0]?4:2);await continuous.fill('0.73123456789');await continuous.press('Tab');
     const saved=await page.locator('.ai-parameters input').evaluateAll(nodes=>nodes.map(n=>n.value));
     await page.reload();await page.waitForFunction(()=>document.querySelector('.ai-presets')?.disabled===false);
     assert.deepEqual(await page.locator('.ai-parameters input').evaluateAll(nodes=>nodes.map(n=>n.value)),saved);
     const unit=data.units.find(u=>u.id===id);
     for(let i=0;i<2;i++){
      const question=page.locator('.ai-question').nth(i);
      await question.getByRole('radio').nth(1-unit.questions[i].answer).check();await question.getByRole('button').click();
      assert.ok((await question.innerText()).includes(lang==='en'?'Try again':'다시'));
      await question.getByRole('radio').nth(unit.questions[i].answer).check();await question.getByRole('button').click();
      assert.ok((await question.innerText()).includes(lang==='en'?'Correct.':'맞았습니다.'));
     }
     await page.locator('.ai-completion input').check();await page.reload();await page.waitForFunction(()=>document.querySelector('.ai-presets')?.disabled===false);assert.ok(await page.locator('.ai-completion input').isChecked());
     const regions=page.locator('.ai-lab .ai-discrete-table');let scrolling=0;
     for(let i=0;i<await regions.count();i++){
      const region=regions.nth(i),max=await region.evaluate(n=>n.scrollWidth-n.clientWidth);
      if(max>1){await region.focus();await page.keyboard.press('ArrowRight');await page.waitForTimeout(150);assert.ok(await region.evaluate(n=>n.scrollLeft>0));await region.evaluate(n=>{n.scrollLeft=n.scrollWidth;});assert.ok(await region.evaluate(n=>n.scrollLeft>=n.scrollWidth-n.clientWidth-1));scrolling++;}
     }
     await page.getByRole('button',{name:lang==='en'?'Reset all experiment settings':'모든 실험 설정 초기화',exact:true}).click();
     assert.deepEqual(await page.locator('.ai-parameters input').evaluateAll(nodes=>nodes.map(n=>Number(n.value))),spec.controls.map(c=>c.initial));
     interactions.push({id,lang,concepts:4,guides:3,presets:spec.presets.length,wrongAndRightQuizzes:2,persistence:true,completion:true,keyboardScrollableTables:scrolling});save();
    }
    console.log(id,lang,dark?'dark':'light','passed');
   }
   await context.close();
  }
  assert.equal(rows.length,72);assert.equal(guides.length,24);assert.equal(interactions.length,4);assert.deepEqual(errors,[]);
  console.log('Sampling and sensors browser passed:72 states,24 expanded guide checks,4 complete interaction groups.');
 }finally{await browser.close();save();}
})().catch(error=>{console.error(error);process.exitCode=1;});
