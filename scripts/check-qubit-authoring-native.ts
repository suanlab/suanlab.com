import assert from 'node:assert/strict';
import {mkdirSync,readFileSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {build} from 'esbuild';
import {chromium} from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import {qubitTeaching} from '../src/data/qubit-teaching';
const runtime='.runtime/learning/quantum-qubit-authoring',sha=(v:string|Buffer)=>createHash('sha256').update(v).digest('hex');
async function main(){
 mkdirSync(runtime,{recursive:true});
 const bundle=await build({entryPoints:['scripts/learning/qubit-authoring-preview.tsx'],bundle:true,write:false,platform:'browser',format:'iife',jsx:'automatic',define:{'process.env.NODE_ENV':'"production"'},metafile:true,logLevel:'silent'}),code=bundle.outputFiles[0].text,inputs=Object.keys(bundle.metafile!.inputs);
 for(const name of ['learning-qubit-controller.ts','learning-qubit-worker-factory.ts','learning-qubit-scalar.ts','learning-qubit-channel.ts','learning-qubit-circuit-exact.ts','learning-qubit-circuit-report.ts','learning-qubit-channel-algebra.ts','learning-qubit-decode.ts','learning-quantum-hilbert-core.ts'])assert(!inputs.includes('src/lib/'+name),name);
 assert(!inputs.includes('src/data/interactive-book.ts'));
 const cssUrls=Array.from(readFileSync('out/book/online/physics-math-for-ai/finite-spin-chain-local-hamiltonians/index.html','utf8').matchAll(/href="([^\"]+\.css)"/g),m=>m[1]);assert(cssUrls.length>0);
 writeFileSync(`${runtime}/concept-main-thread-graph.json`,JSON.stringify({mathematicalEnginesAndWorkersExcluded:true,registeredUnitRegistryExcluded:true,inputs,dependencyHashes:Object.fromEntries(inputs.map(p=>[p,sha(readFileSync(p))])),metafile:bundle.metafile,css:cssUrls.map(url=>({url,file:'out'+url,sha256:sha(readFileSync('out'+url))})),bundleSha256:sha(code)},null,2)+'\n');
 writeFileSync(`${runtime}/authoring-preview.js`,code);
 const browser=await chromium.launch({headless:true}),errors:string[]=[],states:unknown[]=[],flows:unknown[]=[];let workers=0,axes=0;
 try{
  for(const language of ['ko','en'] as const)for(const dark of [false,true])for(const width of [320,1440]){
   const context=await browser.newContext({viewport:{width,height:1000},reducedMotion:'reduce'});await context.addInitScript({content:'globalThis.__name=(fn)=>fn;'});
   const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));page.on('worker',()=>workers++);
   await page.route('**/__qubit_authoring__/**',route=>route.fulfill({contentType:'text/html',body:`<!doctype html><html lang="${language}"><head><meta name="viewport" content="width=device-width, initial-scale=1"><title>Qubit concept review</title>${cssUrls.map(url=>`<link rel="stylesheet" href="${url}">`).join('')}</head><body><div id="root"></div></body></html>`}));
   await page.route('**/_next/static/css/*.css',route=>{const url=new URL(route.request().url()).pathname;return route.fulfill({contentType:'text/css',body:readFileSync('out'+url)});});
   for(const item of qubitTeaching){
    const url=`http://localhost:4175/__qubit_authoring__/?unit=${item.unit.id}&lang=${language}&theme=${dark?'dark':'light'}`;
    await page.goto(url);await page.addScriptTag({content:code});
    await page.waitForFunction(({id,language})=>document.querySelector('main')?.getAttribute('data-qubit-authoring')===id&&document.querySelector('main')?.getAttribute('data-language')===language,{id:item.unit.id,language});
    await page.waitForFunction(()=>!document.querySelector('fieldset')?.hasAttribute('disabled'));
    const steps=page.locator('.ai-concept-steps button');assert.equal(await steps.count(),item.guide.steps.length);
    for(let step=0;step<item.guide.steps.length;step++){
     await steps.nth(step).click();await page.waitForFunction(n=>document.querySelector('[data-qubit-concept]')?.getAttribute('data-qubit-concept-step')===String(n),step);
     assert.equal(await page.locator('[data-qubit-concept-explanation]').count(),0);
     const reveal=page.locator('[data-qubit-concept] button[aria-controls]');await reveal.press('Enter');assert.equal(await reveal.getAttribute('aria-expanded'),'true');assert.equal(await page.locator('[data-qubit-concept-explanation]').innerText(),item.guide.experiments[step].explain[language]);
     const geometry=await page.locator('[data-qubit-diagram] svg').evaluate(svg=>{
      const element=svg as SVGSVGElement,scale=element.getScreenCTM()!.a;
      const texts=Array.from(element.querySelectorAll('text')).map(t=>{const b=t.getBBox();return {text:t.textContent??'',x:b.x,y:b.y,width:b.width,height:b.height,pixels:Number.parseFloat(getComputedStyle(t).fontSize)*scale};});
      const clipped=texts.filter(t=>t.x<-.5||t.y<-.5||t.x+t.width>640.5||t.y+t.height>320.5),overlaps:string[]=[];
      for(let i=0;i<texts.length;i++)for(let j=i+1;j<texts.length;j++){const a=texts[i],b=texts[j],dx=Math.min(a.x+a.width,b.x+b.width)-Math.max(a.x,b.x),dy=Math.min(a.y+a.height,b.y+b.height)-Math.max(a.y,b.y);if(dx>1&&dy>1)overlaps.push(a.text+' / '+b.text);}
      return {minimumTextPixels:Math.min(...texts.map(t=>t.pixels)),clipped,overlaps};
     });
     assert(geometry.minimumTextPixels>=16,item.unit.id+' text too small');assert.deepEqual(geometry.clipped,[],`${item.unit.id}/${step} clipped SVG text`);assert.deepEqual(geometry.overlaps,[],`${item.unit.id}/${step} overlapping SVG text`);
     if(width===320){const region=page.locator('[data-qubit-diagram] [role="region"]');await region.focus();await region.press('ArrowRight');await page.waitForFunction(()=>document.querySelector('[data-qubit-diagram] [role="region"]')!.scrollLeft>0);await region.evaluate(el=>{el.scrollLeft=0;});}
     const style=await page.locator('.ai-concept-steps button[aria-pressed="true"]').evaluate(el=>{const primary=document.createElement('span');primary.style.color='hsl('+getComputedStyle(document.documentElement).getPropertyValue('--primary')+')';document.body.append(primary);const expected=getComputedStyle(primary).color,result={expected,actual:getComputedStyle(el).borderColor};primary.remove();return result;});assert.equal(style.actual,style.expected);
     const a11y=await new AxeBuilder({page}).include('main').withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();assert.deepEqual(a11y.violations,[],`${item.unit.id}/${step}/${language}/${width} accessibility`);axes++;
     const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1);assert(!overflow,`${item.unit.id} body overflow`);
     if(language==='en')assert(!/[가-힣]/.test(await page.locator('main').innerText()));
     states.push({id:item.unit.id,language,dark,width,step,geometry,accessibilityPassed:true,bodyOverflow:false});
    }
    for(const q of item.unit.questions){const field=page.locator(`fieldset:has(input[name="${q.id}"])`),wrong=(q.answer+1)%3;
     await field.locator(`input[value="${wrong}"]`).check();await field.getByRole('button').click();assert((await field.locator('[role="status"]').innerText()).includes(q.explanation[language]));
     await field.locator(`input[value="${q.answer}"]`).check();await field.getByRole('button').click();assert.equal(await field.locator('.ai-answer-correct').count(),1);
    }
    const complete=page.locator('.ai-completion input');await complete.check();await page.waitForFunction(id=>JSON.parse(localStorage.getItem(`suanlab-learning:${id}:study`)??'null')?.value.complete===true,item.unit.id);
    if(language==='en')assert(!/[가-힣]/.test(await page.locator('main').innerText()));
    await page.locator('[data-qubit-concept]').scrollIntoViewIfNeeded();if(language==='en')await page.screenshot({path:`${runtime}/${item.unit.id}-${dark?'dark':'light'}-${width}.png`,fullPage:false});
    await page.reload();await page.addScriptTag({content:code});await page.waitForFunction(()=>document.querySelector('.ai-completion input')?.matches(':checked'));assert.equal(await page.locator('input[type="radio"]:checked').count(),item.unit.questions.length);
    flows.push({id:item.unit.id,language,dark,width,explainedQuestions:item.unit.questions.length,answersAndCompletionReloadVerified:true});
   }
   // Switching a four-step lesson to a three-step lesson must reset local state.
   await page.locator('.ai-concept-steps button').last().click();await page.locator('#unit-select').selectOption(qubitTeaching[0].unit.id);await page.waitForFunction(()=>document.querySelector('[data-qubit-concept]')?.getAttribute('data-qubit-concept-step')==='0');
   await context.close();
  }
  assert.equal(states.length,104);assert.equal(axes,104);assert.equal(flows.length,32);assert.equal(workers,0);assert.deepEqual(errors,[]);
  writeFileSync(`${runtime}/native-concepts.json`,JSON.stringify({conceptStates:states.length,accessibilityStates:axes,quizAndReloadFlows:flows.length,workersCreated:workers,states,flows,crossLessonStepResetVerified:true,mathematicalLabUIImplemented:false,actualReaderVerified:false,errors},null,2)+'\n');console.log(JSON.stringify({conceptStates:states.length,accessibilityStates:axes,quizAndReloadFlows:flows.length,workers}));
 }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
