import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {build} from 'esbuild';
import {chromium,type Page} from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import {getLearningContent} from '../src/lib/learning-content';
import {bosonTeachingPresets as presets,bosonLearningId} from '../src/lib/learning-many-body-boson-teaching';
const runtime='.runtime/learning/many-body-boson-teaching';
async function main(){
 const content=await getLearningContent(bosonLearningId);
 const app=`import React,{StrictMode} from 'react';import {createRoot} from 'react-dom/client';import {LanguageProvider,useLanguage} from './src/components/language-provider';import {SiteMotionProvider} from './src/components/motion/motion-provider';import Lesson from './src/components/learning/many-body-boson-lesson';function Toolbar(){const{setLanguage}=useLanguage();return <header><button id="locale-ko" onClick={()=>setLanguage('ko')}>KO</button><button id="locale-en" onClick={()=>setLanguage('en')}>EN</button></header>;}const root=createRoot(document.getElementById('root'));root.render(<StrictMode><LanguageProvider><Toolbar/><SiteMotionProvider><main id="main-content" style={{maxWidth:1200,margin:'auto',padding:16}}><Lesson content={${JSON.stringify(content)}}/></main></SiteMotionProvider></LanguageProvider></StrictMode>);globalThis.bosonTeachingHarness.unmount=()=>root.unmount();`;
 const bundles=await Promise.all([
  build({entryPoints:['src/lib/learning-many-body.worker.ts'],bundle:true,write:false,platform:'browser',format:'iife',target:'es2020'}),
  build({entryPoints:['src/lib/learning-many-body-validation.worker.ts'],bundle:true,write:false,platform:'browser',format:'iife',target:'es2020'}),
  build({stdin:{contents:app,resolveDir:process.cwd(),loader:'tsx'},jsx:'automatic',bundle:true,write:false,metafile:true,platform:'browser',format:'iife',target:'es2020',plugins:[{name:'preview-worker-urls',setup(b){b.onLoad({filter:/learning-many-body-worker-factory\.ts$/},()=>({contents:"export const createManyBodyWorker=()=>globalThis.bosonTeachingHarness.makeWorker('calculate');",loader:'ts'}));b.onLoad({filter:/learning-many-body-validation-worker-factory\.ts$/},()=>({contents:"export const createManyBodyValidationWorker=()=>globalThis.bosonTeachingHarness.makeWorker('validate');",loader:'ts'}));}}]})
 ]);
 for(const[i,name]of['calculation','validation','lesson'].entries())writeFileSync(`${runtime}/native-${name}-bundle.js`,bundles[i].outputFiles[0].text);
 const staticHtml=readFileSync('out/book/online/physics-math-for-ai/finite-state-rayleigh-energy-variance/index.html','utf8'),styles=(staticHtml.match(/<link[^>]*rel="stylesheet"[^>]*>/g)||[]).join('');assert(styles.length>0);
 const browser=await chromium.launch({headless:true,args:['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader']});
 const cases:any[]=[],errors:string[]=[],lifetimes:any[]=[];let accessibilityStates=0;
 async function setup(locale:'ko'|'en',dark=false,width=1440,reduced=false,options:{storage?:'blocked'|'corrupt';webgl?:boolean;fail?:boolean}={}){
  const context=await browser.newContext({viewport:{width,height:950},reducedMotion:reduced?'reduce':'no-preference'}),p=await context.newPage();p.on('pageerror',e=>{errors.push(e.message);console.log('PAGE ERROR '+e.message);});
  await p.addInitScript(({locale,options})=>{if(options.storage==='blocked'){Object.defineProperty(Storage.prototype,'getItem',{value(){throw Error('storage unavailable');}});Object.defineProperty(Storage.prototype,'setItem',{value(){throw Error('storage unavailable');}});}else{localStorage.setItem('suanlab-language',locale);if(options.storage==='corrupt')localStorage.setItem('suanlab-many-body-boson-fock','{"version":2,"value":{}}');}if(options.webgl===false){const prior=HTMLCanvasElement.prototype.getContext;HTMLCanvasElement.prototype.getContext=function(this:HTMLCanvasElement,type:any,...args:any[]){if(type==='webgl'||type==='webgl2'||type==='experimental-webgl')return null;return prior.call(this,type,...args);} as typeof prior;}},{locale,options});
  await p.route('**/__many_body_boson_teaching__/',r=>r.fulfill({contentType:'text/html',body:`<!doctype html><html lang="${locale}" class="${dark?'dark':''}"><head><title>Boson Fock lesson verification</title>${styles}</head><body><div id="root"></div></body></html>`}));
  await p.goto('http://localhost:4175/__many_body_boson_teaching__/');
  await p.evaluate(({calc,validate,fail})=>{const urls={calculate:URL.createObjectURL(new Blob([calc],{type:'text/javascript'})),validate:URL.createObjectURL(new Blob([validate],{type:'text/javascript'}))};const h:any={created:0,terminated:0,urls,fail:fail?'calculate':null};h.makeWorker=(role:'calculate'|'validate')=>{if(h.fail===role)throw Error('worker unavailable');const w=new Worker(urls[role],{type:'module'});h.created++;const terminate=w.terminate.bind(w);let closed=false;w.terminate=()=>{if(!closed){closed=true;h.terminated++;}terminate();};return w;};(globalThis as any).bosonTeachingHarness=h;},{calc:bundles[0].outputFiles[0].text,validate:bundles[1].outputFiles[0].text,fail:options.fail});
  await p.addScriptTag({content:bundles[2].outputFiles[0].text});if(options.storage==='blocked'&&locale==='en')await p.locator('#locale-en').click();await p.waitForFunction(locale=>document.documentElement.lang===locale,locale);return p;
 }
 async function ready(p:Page,index?:number){try{await p.waitForFunction(({key})=>{const lab=document.querySelector('[data-boson-lab]');return lab?.getAttribute('data-boson-state')==='ready'&&(!key||lab.getAttribute('data-boson-input')===key);},{key:index===undefined?null:JSON.stringify(presets[index].input)},{timeout:15000});}catch(e){writeFileSync(`${runtime}/native-failure.json`,JSON.stringify({errors,diagnostic:await p.evaluate(()=>({lab:document.querySelector('[data-boson-lab]')?.outerHTML.slice(-8000),created:(globalThis as any).bosonTeachingHarness?.created,terminated:(globalThis as any).bosonTeachingHarness?.terminated})),expected:index===undefined?null:JSON.stringify(presets[index].input)},null,2));throw e;}}
 async function close(p:Page){const v=await p.evaluate(()=>{const h=(globalThis as any).bosonTeachingHarness;h.unmount();Object.values(h.urls).forEach(u=>URL.revokeObjectURL(u as string));return{created:h.created,terminated:h.terminated};});assert.equal(v.created,v.terminated);lifetimes.push(v);await p.context().close();}
 try{
  for(const locale of (process.argv.includes('--interaction-only')?[]:['ko','en']) as ('ko'|'en')[])for(const dark of[false,true])for(const width of[320,1440]){
   const p=await setup(locale,dark,width,true);await ready(p,0);
   assert.equal(await p.locator('[data-boson-result]').getAttribute('data-boson-energy'),'0');
   if(locale==='en')assert(!/[가-힣]/.test(await p.locator('[data-boson-lesson]').innerText()));
   const overflow=await p.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth,elements:Array.from(document.querySelectorAll('*')).map(e=>({tag:e.tagName,cls:e.className,text:e.textContent?.slice(0,100),rect:e.getBoundingClientRect().toJSON()})).filter(v=>v.rect.right>innerWidth+1&&v.rect.width>0).filter(v=>!['semantics','mrow','msqrt','mi','mtext','mo','mn','msup','mspace','annotation','math'].includes(v.tag)).slice(-35)}));if(overflow.scroll>overflow.width+1){writeFileSync(`${runtime}/overflow-failure.json`,JSON.stringify(overflow,null,2));await p.screenshot({path:`${runtime}/overflow-failure.png`,fullPage:true});}assert(overflow.scroll<=overflow.width+1,'horizontal overflow');
   const axe=await new AxeBuilder({page:p}).include('[data-boson-lesson]').withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();
   if(axe.violations.length)writeFileSync(`${runtime}/axe-failures.json`,JSON.stringify(axe.violations,null,2));
   assert.deepEqual(axe.violations.map(v=>v.id),[]);accessibilityStates++;
   await p.locator('[data-boson-kind="boson-coherent"]').click();await ready(p,6);
   if(locale==='en')assert(!/[가-힣]/.test(await p.locator('[data-boson-lesson]').innerText()));
   assert(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'coherent horizontal overflow');
   const coherentAxe=await new AxeBuilder({page:p}).include('[data-boson-lesson]').withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();if(coherentAxe.violations.length)writeFileSync(`${runtime}/coherent-axe-failures.json`,JSON.stringify(coherentAxe.violations,null,2));assert.deepEqual(coherentAxe.violations.map(v=>v.id),[]);accessibilityStates++;
   cases.push({locale,dark,width,models:2});await close(p);
  }
  const p=await setup('en',true,320);await ready(p,0);
  for(let i=0;i<presets.length;i++){
   await p.locator(`[data-boson-kind="${presets[i].input.kind}"]`).click();
   await p.locator(`[data-boson-preset="${presets[i].id}"]`).click();await ready(p,i);
   if(i>=6)await p.locator('[data-boson-coherent-scopes]').waitFor();
  }
  await p.locator('[data-boson-kind="boson-fock"]').click();await ready(p,5);
  await p.locator('[data-boson-kind="boson-coherent"]').click();await ready(p,11);
  await p.locator('[data-boson-preset="coherent-tiny-tail"]').click();await ready(p,10);
  await p.locator('[data-boson-coherent-scopes]').scrollIntoViewIfNeeded();await p.screenshot({path:`${runtime}/boson-en-coherent-mobile.png`});
  await p.locator('[data-boson-kind="boson-fock"]').click();await ready(p,5);
  await p.locator('[data-boson-preset="raw-top"]').click();await ready(p,1);
  const ladder=p.locator('[data-boson-ladder]');await ladder.locator('select').selectOption('3');
  await ladder.getByRole('button',{name:'Step 1',exact:true}).click();assert((await p.locator('[data-boson-ladder-status]').innerText()).includes('zero vector'));
  await ladder.locator('select').selectOption('1');await ladder.getByRole('button',{name:'Step 1',exact:true}).focus();await p.keyboard.press('Enter');assert((await p.locator('[data-boson-ladder-status]').innerText()).includes('√2'));
  await p.locator('summary').filter({hasText:'Edit all active integer'}).click();
  const norm=p.locator('[data-boson-field="hDenominator"]');await norm.fill('0');await norm.press('Enter');assert.equal(await norm.getAttribute('aria-invalid'),'true');await ready(p,1);
  const axe=await new AxeBuilder({page:p}).include('[data-boson-lesson]').withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();if(axe.violations.length)writeFileSync(`${runtime}/expanded-axe-failures.json`,JSON.stringify(axe.violations,null,2));assert.deepEqual(axe.violations.map(v=>v.id),[]);accessibilityStates++;
  await close(p);
  const failed=await setup('en',false,1440,false,{fail:true});await failed.waitForFunction(()=>document.querySelector('[data-boson-lab]')?.getAttribute('data-boson-state')==='error');await failed.evaluate(()=>{(globalThis as any).bosonTeachingHarness.fail=null;});await failed.getByRole('button',{name:'Retry',exact:true}).click();await ready(failed,0);await close(failed);
  const denied=await setup('en',false,320,true,{storage:'blocked'});await ready(denied,0);assert((await denied.locator('[data-boson-lab]').innerText()).includes('storage is unavailable'));await close(denied);
  assert.deepEqual(errors,[]);writeFileSync(`${runtime}/${process.argv.includes('--interaction-only')?'native-interaction-result':'native-result'}.json`,JSON.stringify({scope:process.argv.includes('--interaction-only')?'interaction-only':'full',cases,accessibilityStates,presetChecks:12,independentModelStorageRestoration:true,zeroVectorAndSqrtFactors:true,keyboardStep:true,invalidInputRetainsPriorResult:true,workerRetry:true,deniedStorage:true,errors,workers:lifetimes},null,2)+'\n');
 }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
