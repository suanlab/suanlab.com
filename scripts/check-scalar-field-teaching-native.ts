import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {build} from 'esbuild';
import {chromium} from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import {scalarFieldSteps,scalarFieldQuestions} from '../src/data/scalar-field-teaching';
const root='.runtime/learning/scalar-field-teaching';
async function main(){
 const html=readFileSync('out/book/interactive/index.html','utf8'),links=[...html.matchAll(/<link[^>]*>/g)].map(x=>x[0]).filter(x=>x.includes('rel="stylesheet"')).map(x=>x.match(/href="([^"]+)"/)?.[1]).filter((v):v is string=>!!v),css=['src/styles/qubit-lab.css','src/styles/information-lab.css','src/styles/scalar-field-lab.css'].map(p=>readFileSync(p,'utf8')).join('\n');
 const bundle=await build({stdin:{contents:`import{createElement,useState}from'react';import{createRoot}from'react-dom/client';import Lab from'./src/components/learning/scalar-field-lab';function App(){const[language,setLanguage]=useState(window.__language);window.__setLanguage=setLanguage;return createElement('main',{className:'ai-book'},createElement('h1',null,'Interactive Book'),createElement(Lab,{language}));}const root=createRoot(document.getElementById('root'));window.__unmount=()=>root.unmount();root.render(createElement(App));`,resolveDir:process.cwd(),loader:'tsx'},bundle:true,write:false,platform:'browser',format:'esm',jsx:'automatic',define:{'process.env.NODE_ENV':'"production"'},metafile:true});
 assert(!Object.keys(bundle.metafile!.inputs).some(p=>/learning-scalar-field(?:\.ts|-validation\.ts|-evaluator\.ts|-validation-evaluator\.ts)/.test(p)));writeFileSync(root+'/reader-dependency-graph.json',JSON.stringify(bundle.metafile,null,2)+'\n');
 const codes={calculate:readFileSync('.runtime/learning/scalar-field-worker/calculate.js','utf8'),validate:readFileSync('.runtime/learning/scalar-field-worker/validate.js','utf8')};
 const browser=await chromium.launch({headless:true}),states=[];let accessibility=0;
 try{
  for(const language of ['ko','en']as const)for(const dark of [false,true])for(const width of [320,1440]){
   const context=await browser.newContext({viewport:{width,height:1000},deviceScaleFactor:width===320?2:1,reducedMotion:'reduce'});
   await context.addInitScript({content:`globalThis.__name=fn=>fn;window.__language=${JSON.stringify(language)};window.__counts={created:0,terminated:0,active:0};const Native=Worker;window.Worker=class extends Native{constructor(...args){super(...args);this.closed=false;window.__counts.created++;window.__counts.active++;}terminate(){if(!this.closed){this.closed=true;window.__counts.terminated++;window.__counts.active--;}super.terminate();}};`});
   const page=await context.newPage(),errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
   await page.route('**/__scalar_field_teaching__/',r=>r.fulfill({contentType:'text/html',body:`<!doctype html><html lang="${language}" class="${dark?'dark':''}"><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>Scalar field lesson</title>${links.map(h=>`<link rel="stylesheet" href="${h}">`).join('')}<style>${css}</style></head><body><div id="root"></div></body></html>`}));
   await page.route('**/learning-scalar-field.worker.ts',r=>r.fulfill({contentType:'text/javascript',body:codes.calculate}));await page.route('**/learning-scalar-field-validation.worker.ts',r=>r.fulfill({contentType:'text/javascript',body:codes.validate}));
   const mount=async()=>{await page.goto('http://127.0.0.1:4175/__scalar_field_teaching__/');await page.addScriptTag({type:'module',content:bundle.outputFiles[0].text});};
   const ready=()=>page.locator('[data-scalar-phase="ready"]').waitFor({timeout:30000});const released=async()=>{await page.waitForFunction(()=>(window as unknown as {__counts:{active:number}}).__counts.active===0);};
   const axe=async()=>{const v=(await new AxeBuilder({page}).analyze()).violations;assert.deepEqual(v.map(x=>({id:x.id,nodes:x.nodes.map(n=>n.target)})),[]);accessibility++;};
   await mount();await ready();const examples=page.getByLabel(language==='ko'?'전체 예제':'All examples',{exact:true});
   const energies=[['three-site-power','7/2'],['same-shape-rest','3'],['uniform-shift','7/2'],['half-spacing','25/4'],['periodic-uniform','0'],['neumann-uniform','0'],['fixed-uniform','1'],['onsite-restoring','3/2'],['double-well-minimum','-3/4'],['boundary-stabilized','5/2'],['tiny-quartic','3/4'+'0'.repeat(68)],['thirty-two-sites','15531643/259200']];
   for(const[id,energy]of energies){await examples.selectOption(id);await ready();assert.equal(await page.locator('[data-scalar-metric=H]').innerText(),energy,id);}await released();
   const before=await page.evaluate(()=>(window as unknown as {__counts:{created:number}}).__counts.created);
   for(const[index,step]of scalarFieldSteps.entries()){await page.getByRole('button',{name:`${index+1}. ${step.title[language]}`,exact:true}).click();assert.equal(await page.locator('.information-step > p').first().innerText(),step.body[language]);}
   assert.equal(await page.locator('.information-step').evaluate(el=>getComputedStyle(el).animationName),'none');
   for(let i=0;i<4;i++){await page.getByLabel(language==='ko'?'안내 실험':'Guided experiment',{exact:true}).selectOption(String(i));await page.getByRole('button',{name:language==='ko'?'해설 보기':'Show explanation',exact:true}).click();await page.locator('[data-scalar-guide-explanation]').waitFor();}
   for(const[index,q]of scalarFieldQuestions.entries()){const field=page.locator('.information-question').nth(index);await field.locator('input').nth(q.correct).check();assert((await field.locator('[role=status]').innerText()).includes(q.explanation[language]));}
   await page.getByText(language==='ko'?'헤시안과 안정성의 범위':'Hessian and the scope of stability claims',{exact:true}).click();await page.getByLabel(language==='ko'?'헤시안 행 블록':'Hessian row block',{exact:true}).selectOption('28');await page.getByLabel(language==='ko'?'헤시안 열 블록':'Hessian column block',{exact:true}).selectOption('28');assert.equal(await page.locator('[data-scalar-hessian] tbody tr').count(),4);assert.equal(await page.locator('[data-scalar-hessian] tbody tr').last().locator('th').innerText(),'31');
   await page.getByLabel(language==='ko'?'그림에서 볼 양':'Quantity to inspect',{exact:true}).selectOption('acceleration');assert.equal(await page.evaluate(()=>(window as unknown as {__counts:{created:number}}).__counts.created),before);await axe();
   await page.getByRole('button',{name:language==='ko'?'첫 예제로 초기화':'Reset to first example',exact:true}).click();await ready();
   const q=page.getByLabel(language==='ko'?'변위 q 목록':'Displacement values q',{exact:true}),apply=page.getByRole('button',{name:language==='ko'?'입력 적용':'Apply input',exact:true});await q.fill('2,0,-2');await apply.click();await ready();assert.equal(await page.locator('[data-scalar-metric=H]').innerText(),'25/2');
   await q.fill('1,2');await apply.click();assert(await page.getByRole('alert').count());assert.equal(await page.locator('[data-scalar-metric=H]').innerText(),'25/2');
   await released();const previousCounts=await page.evaluate(()=>(window as unknown as {__counts:{created:number;terminated:number}}).__counts);assert.equal(previousCounts.created,previousCounts.terminated);await mount();await ready();assert.equal(await examples.inputValue(),'custom');assert.equal(await page.locator('[data-scalar-metric=H]').innerText(),'25/2');
   await page.getByRole('button',{name:language==='ko'?'첫 예제로 초기화':'Reset to first example',exact:true}).click();await ready();
   const plot=page.locator('[data-scalar-plot]');await plot.scrollIntoViewIfNeeded();await page.locator('[data-scalar-plot="3d"] canvas').waitFor();const canvas=plot.locator('canvas');await plot.getByRole('button',{name:language==='ko'?'정면':'Front',exact:true}).click();const first=await canvas.screenshot();await canvas.press('ArrowRight');await canvas.press('ArrowUp');assert(!first.equals(await canvas.screenshot()),'Keyboard camera must change the view');
   await canvas.evaluate(el=>{const ext=(el as HTMLCanvasElement).getContext('webgl2')?.getExtension('WEBGL_lose_context');if(!ext)throw Error('No context loss extension');ext.loseContext();});await page.locator('[data-scalar-plot="2d"]').waitFor();await axe();
   await plot.getByRole('button',{name:language==='ko'?'3D 다시 시도':'Retry 3D',exact:true}).click();await page.locator('[data-scalar-plot="3d"] canvas').waitFor();
   assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));if(language==='en')assert(!/[가-힣]/.test(await page.locator('main').innerText()));
   await page.screenshot({path:`${root}/${language}-${dark?'dark':'light'}-${width}.png`});await page.evaluate(()=>(window as unknown as {__unmount:()=>void}).__unmount());await released();
   const counts=await page.evaluate(()=>(window as unknown as {__counts:{created:number;terminated:number;active:number}}).__counts);assert.equal(counts.created,counts.terminated);assert.deepEqual(errors,[]);counts.created+=previousCounts.created;counts.terminated+=previousCounts.terminated;states.push({language,dark,width,dpr:width===320?2:1,examples:12,conceptSteps:7,explainedQuestions:5,customInputReload:true,keyboardCamera:true,webglLossRetry:true,overflow:false,...counts});console.log(JSON.stringify({cases:states.length,language,dark,width}));await context.close();
  }
  writeFileSync(root+'/native-verification.json',JSON.stringify({states,accessibility,errors:[],readerRegistered:false,productionNextAssetsVerified:false},null,2)+'\n');console.log(JSON.stringify({cases:8,accessibility}));
 }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
