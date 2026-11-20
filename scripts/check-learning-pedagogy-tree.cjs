const assert=require('node:assert/strict');
const fs=require('fs');
const {execFileSync}=require('child_process');
const {chromium}=require('playwright');
const AxeBuilder=require('@axe-core/playwright').default;
const base=process.env.SUANLAB_TEST_BASE||'http://127.0.0.1:43309';
const root='/book/online/physics-math-for-ai/';
const directory='.runtime/learning/pedagogy-tree';
const report=process.env.SUANLAB_PEDAGOGY_REPORT||`${directory}/browser.json`;
const data=JSON.parse(execFileSync('node_modules/.bin/tsx',['-e',`
import {firstStepIds,firstStepEquation,firstStepDisplayEquation,projectionPreparation,descentPreparation,bayesPreparation,interferencePreparation} from './src/lib/learning-first-steps';
import {firstStepLessons} from './src/data/learning-first-steps';
console.log(JSON.stringify(firstStepIds.map(id=>({id,variants:firstStepLessons[id].examples.map((_,example)=>({example,stages:Array.from({length:4},(_,stage)=>({stage,tex:firstStepDisplayEquation(id,example,stage),exactTex:firstStepEquation(id,example,stage),geometry:id==='inner-products'?projectionPreparation(example):id==='gradients'?descentPreparation(example,stage):id==='bayesian-inference'?bayesPreparation(example,stage):interferencePreparation(example,stage)}))}))}))));
`],{encoding:'utf8',maxBuffer:4e6}));
(async()=>{
 fs.mkdirSync(directory,{recursive:true});
 const browser=await chromium.launch({executablePath:'/usr/bin/google-chrome',args:['--no-sandbox','--enable-unsafe-swiftshader']});
 const rows=[],flows=[],stages=[],motions=[],errors=[];
 const save=()=>fs.writeFileSync(report,JSON.stringify({rows,flows,stages,motions,errors,svgCoordinateTolerance:1e-4},null,2)+'\n');
 const context=async(lang,width,dark,reduced='reduce',seed={},blocked=false)=>{
  const c=await browser.newContext({viewport:{width,height:1000},reducedMotion:reduced,colorScheme:dark?'dark':'light'});
  await c.addInitScript(({lang,dark,seed,origin,blocked})=>{if(location.origin!==origin)return;localStorage.setItem('suanlab-language',lang);localStorage.setItem('theme',dark?'dark':'light');for(const[k,v]of Object.entries(seed))localStorage.setItem(k,v);if(blocked){Object.defineProperty(Storage.prototype,'getItem',{configurable:true,value(){throw Error('Storage unavailable');}});Object.defineProperty(Storage.prototype,'setItem',{configurable:true,value(){throw Error('Storage unavailable');}});}},{lang,dark,seed,blocked,origin:new URL(base).origin});
  await c.route('**/*',r=>new URL(r.request().url()).origin===new URL(base).origin?r.continue():r.abort());
  c.on('page',p=>{p.on('pageerror',e=>errors.push(e.message));p.on('console',m=>{if(m.type()==='error'&&/Hydration|did not match|Encountered two children/.test(m.text()))errors.push(m.text());});});return c;
 };
 const go=async(p,id,lang)=>{assert.equal((await p.goto(base+root+(id?id+'/':''))).status(),200);await p.waitForFunction(lang=>document.documentElement.lang===lang,lang);await p.evaluate(()=>document.fonts.ready);};
 const audit=async(p,label)=>{assert.equal(await p.locator('.katex-error').count(),0);assert(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),label+' page overflow');const a=await new AxeBuilder({page:p}).include('.ai-book').withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();assert.deepEqual(a.violations.map(v=>({id:v.id,targets:v.nodes.map(n=>n.target)})),[],label);};
 try{
  const combinations=['ko','en'].flatMap(lang=>[320,768,1440].map(width=>({lang,width,dark:false})).concat([{lang,width:1440,dark:true}]));
  for(const {lang,width,dark}of combinations){
   const c=await context(lang,width,dark),p=await c.newPage();await go(p,null,lang);
   const tree=p.locator('[data-course-tree="overview"]');await tree.locator('button.ai-tree-path').first().waitFor({state:'visible'});await p.waitForFunction(()=>document.querySelector('.ai-tree-path')?.disabled===false);
   assert.equal(await tree.locator('[data-course-path]').count(),7);assert.equal(await tree.locator('[data-tree-unit]').count(),650);assert.equal(await tree.locator('[data-tree-unit]:visible').count(),0);assert.equal(await p.locator('.ai-unit-card').count(),0);
   if(lang==='en')assert(!/[가-힣]/.test(await p.locator('.ai-book').innerText()));
   if(lang==='en'&&width===320&&!dark){
    const path=tree.locator('[data-course-path="mathematics"]>button');await path.focus();await p.keyboard.press('Enter');assert.equal(await path.getAttribute('aria-expanded'),'true');
    const domain=tree.locator('[data-course-domain="linear-algebra"]>button');await domain.click();assert(await tree.locator('[data-tree-unit="inner-products"]').isVisible());
    await p.reload();await p.waitForFunction(()=>document.querySelector('[data-course-domain="linear-algebra"]>button')?.getAttribute('aria-expanded')==='true');
    const search=tree.getByRole('searchbox');await search.fill('ＰＣＡ');assert(await tree.locator('[data-tree-unit="eigenvalues"]').isVisible());
    await search.fill('no-such-topic-823591');assert.equal(await tree.locator('[data-tree-unit]').count(),0);assert((await tree.getByRole('status').innerText()).includes('0 matching'));
    await tree.getByRole('button',{name:'Reset exploration',exact:true}).click();assert.equal(await tree.locator('[data-tree-unit]:visible').count(),0);
    await p.screenshot({path:`${directory}/tree-en-320.png`});flows.push({type:'tree',keyboard:true,savedExpansion:true,normalizedSearch:true,noResults:true,reset:true});
   }
   await audit(p,`overview/${lang}/${width}/${dark}`);rows.push({id:'overview',lang,width,dark});save();
   for(const lesson of data){
    await go(p,lesson.id,lang);const studio=p.locator('[data-learning-studio]');await studio.waitFor();assert.equal(await studio.getAttribute('data-learning-studio'),lesson.id);
    if(lang==='en')assert(!/[가-힣]/.test(await p.locator('.ai-reader').innerText()));
    const current=studio.locator('.ai-learning-math-block').first();assert((await current.innerHTML()).includes('<math'));
    if(width===320&&!dark){
     for(const variant of lesson.variants){
      await studio.locator('select').selectOption(String(variant.example));
      for(const state of variant.stages){
       await studio.locator('.ai-studio-stages button').nth(state.stage).click();await p.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));assert.equal(await studio.getAttribute('data-studio-stage'),String(state.stage));
       assert.equal(await current.locator('annotation[encoding="application/x-tex"]').textContent(),state.tex);assert(await current.evaluate(n=>n.scrollWidth<=n.clientWidth+1),'Primary result fits the mobile viewport');
       if(lesson.id==='gradients')assert.equal(await studio.locator('[data-studio-exact] annotation[encoding=\"application/x-tex\"]').textContent(),state.exactTex);
       if(state.stage>=2&&lesson.id==='inner-products'){
        const marker=studio.locator('circle[fill="var(--learning-orange)"]');assert(Math.abs(Number(await marker.getAttribute('cx'))-(220+45*state.geometry.projected[0]))<1e-6);assert(Math.abs(Number(await marker.getAttribute('cy'))-(235-45*state.geometry.projected[1]))<1e-6);
       }
       if(lesson.id==='gradients'){const point=studio.locator('circle').last();assert(Math.abs(Number(await point.getAttribute('cx'))-(280+18*state.geometry.current.x))<1e-6);assert(Math.abs(Number(await point.getAttribute('cy'))-(170-18*state.geometry.current.y))<1e-6);}
       if(lesson.id==='bayesian-inference'||lesson.id==='finite-qubit-circuit-order-interference'){
        const values=lesson.id==='bayesian-inference'?[state.geometry.fair,state.geometry.biased]:[state.geometry.p0,state.geometry.p1],scale=lesson.id==='bayesian-inference'?390:210;
        for(let i=0;i<2;i++)assert(Math.abs((await studio.locator(`[data-studio-bar="${i}"]`).evaluate(n=>n.getBBox().width))-values[i]*scale)<1e-4);
       }
       stages.push({id:lesson.id,lang,example:variant.example,stage:state.stage,exactMath:true,geometry:true,primaryEquationFitsViewport:true});
      }
     }
     await studio.getByRole('button',{name:lang==='ko'?'예측과 설명 비교하기':'Compare your prediction with the explanation',exact:true}).click();assert(await studio.locator('div[id$="-explanation"]').isVisible());
     assert(await studio.getByRole('button',{name:lang==='ko'?'단계 재생':'Play stages',exact:true}).isDisabled());
     if(lesson.id==='gradients'){await studio.locator('.ai-studio-exact>summary').click();assert(await studio.locator('[data-studio-exact]').isVisible());}
     let overflowRegionsChecked=0;
     for(const region of await studio.locator('.ai-learning-math-block').all())if(await region.evaluate(n=>n.scrollWidth>n.clientWidth+2)){await region.focus();await p.keyboard.press('ArrowRight');await p.waitForFunction(n=>n.scrollLeft>0,await region.elementHandle());overflowRegionsChecked++;}
     await studio.getByRole('button',{name:lang==='ko'?'처음으로':'Restart',exact:true}).click();
     const outer=p.locator('.ai-unit-selector');await outer.locator(':scope>summary').click();const readerTree=outer.locator('[data-course-tree="reader"]');assert.equal(await readerTree.locator('[aria-current="page"]').count(),1);assert(await readerTree.locator(`[data-tree-unit="${lesson.id}"]`).isVisible());await outer.locator(':scope>summary').click();
     const completion=p.locator('.ai-completion input');await completion.check();await outer.locator(':scope>summary').click();await p.waitForFunction(()=>document.querySelector('[data-course-tree="reader"] .ai-tree-status')?.textContent.includes('1 / 650')||document.querySelector('[data-course-tree="reader"] .ai-tree-status')?.textContent.includes('1 / 650단원'));
     await outer.locator(':scope>summary').click();await completion.uncheck();
     // New normal-reading capture, after math overflow scrolling and tree exploration.
     await p.reload();await studio.waitFor();await p.evaluate(()=>document.fonts.ready);await p.waitForTimeout(800);await p.evaluate(()=>scrollTo(0,document.querySelector('[data-learning-studio]').getBoundingClientRect().top+scrollY-90));await p.waitForTimeout(800);await p.screenshot({path:`${directory}/${lesson.id}.${lang}.png`});await studio.locator('.ai-studio-scene').evaluate(n=>scrollTo(0,n.getBoundingClientRect().top+scrollY-90));await p.waitForTimeout(120);await p.screenshot({path:`${directory}/${lesson.id}.scene.${lang}.png`});
     flows.push({type:'lesson',id:lesson.id,lang,variants:lesson.variants.length,stages:4,prediction:true,workedExamples:2,reducedMotion:true,mathKeyboardScroll:overflowRegionsChecked>0,overflowRegionsChecked,exactCalculationDisclosure:lesson.id==='gradients',treeCurrentLocation:true,liveCompletion:true});
    }
    await audit(p,`${lesson.id}/${lang}/${width}/${dark}`);rows.push({id:lesson.id,lang,width,dark});save();
   }
   if(lang==='en'&&width===320&&!dark){await go(p,null,lang);assert((await p.locator('.ai-tree-resume').getAttribute('href')).endsWith('finite-qubit-circuit-order-interference/'));flows.push({type:'resume',lastVisited:true});}
   await c.close();
  }
  const c=await context('en',1440,false,'no-preference'),p=await c.newPage();
  for(const lesson of data){
   await go(p,lesson.id,'en');const studio=p.locator('[data-learning-studio]');await studio.waitFor();await studio.getByRole('button',{name:'Play stages',exact:true}).scrollIntoViewIfNeeded();await studio.getByRole('button',{name:'Play stages',exact:true}).waitFor();await p.waitForFunction(()=>!document.querySelector('[data-learning-studio] .ai-studio-playback button:nth-child(2)').disabled);
   if(lesson.id==='gradients'){
    await studio.locator('.ai-studio-stages button').nth(1).click();const samples=await studio.locator('circle').last().evaluate(node=>new Promise(resolve=>{const points=[];let k=0;const read=()=>{points.push(Number(node.getAttribute('cx')));if(++k<24)requestAnimationFrame(read);else resolve(points);};requestAnimationFrame(read);}));assert(samples.some(x=>x>298.01&&x<315.99),'Actual intermediate motion frame');
    await studio.getByRole('button',{name:'Restart',exact:true}).click();
   }
   await studio.getByRole('button',{name:'Play stages',exact:true}).click();await p.waitForFunction(()=>document.querySelector('[data-learning-studio]')?.getAttribute('data-studio-stage')==='1');await studio.getByRole('button',{name:'Pause stages',exact:true}).click();const stopped=await studio.getAttribute('data-studio-stage');await p.waitForTimeout(1750);assert.equal(await studio.getAttribute('data-studio-stage'),stopped);
   await studio.getByRole('button',{name:'Play stages',exact:true}).click();await p.locator('footer').scrollIntoViewIfNeeded();await p.waitForFunction(()=>document.querySelector('[data-learning-studio]')?.getAttribute('data-studio-playing')==='false');motions.push({id:lesson.id,play:true,pause:true,offscreenPause:true});save();
  }
  await c.close();
  for(const storage of ['corrupt','blocked']){
   const c=await context('ko',320,false,'reduce',storage==='corrupt'?{'suanlab-learning:curriculum-tree':JSON.stringify({version:1,value:['branch:missing']})}:{},storage==='blocked');

   const p=await c.newPage();await go(p,null,'ko');await p.waitForFunction(()=>document.querySelector('.ai-tree-path')?.disabled===false);assert.equal(await p.locator('[data-tree-unit]:visible').count(),0);await p.locator('[data-course-path=\"mathematics\"]>button').click();assert.equal(await p.locator('[data-course-path=\"mathematics\"]>button').getAttribute('aria-expanded'),'true');await audit(p,'storage/'+storage);flows.push({type:'storage',storage,navigable:true});await c.close();
  }
  assert.equal(rows.length,40);assert.equal(stages.length,80);assert.equal(motions.length,4);assert.deepEqual(errors,[]);save();console.log('Pedagogy tree checks passed: 40 reader/overview states, 80 exact-math and geometry states, 4 motion flows; no accessibility violations or page errors.');
 }finally{save();await browser.close();console.log('Verification browser closed.');}
})().catch(e=>{console.error(e);process.exitCode=1});
