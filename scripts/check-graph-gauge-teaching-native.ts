import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {build} from 'esbuild';
import {chromium} from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import {graphGaugeSteps,graphGaugeExperiments,graphGaugeQuestions} from '../src/data/graph-gauge-teaching';
import {graphGaugeExamples} from '../src/lib/learning-graph-gauge-examples';
import {calculateGraphGauge} from '../src/lib/learning-graph-gauge';
const energies=graphGaugeExamples.map(c=>[c.id,calculateGraphGauge(c.input).energy.total]);
const root='.runtime/learning/graph-gauge-teaching';
type Harness={__language:string;__counts:{created:number;terminated:number;active:number};__unmount:()=>void};
async function main(){
 const html=readFileSync('out/book/interactive/index.html','utf8'),links=[...html.matchAll(/<link[^>]*>/g)].map(x=>x[0]).filter(x=>x.includes('rel="stylesheet"')).map(x=>x.match(/href="([^"]+)"/)?.[1]).filter((v):v is string=>!!v),css=['src/styles/qubit-lab.css','src/styles/information-lab.css','src/styles/scalar-field-lab.css'].map(p=>readFileSync(p,'utf8')).join('\n');
 const bundle=await build({stdin:{contents:`import{createElement,useState}from'react';import{createRoot}from'react-dom/client';import Lab from'./src/components/learning/graph-gauge-lab';function App(){const[language,setLanguage]=useState(window.__language);window.__setLanguage=setLanguage;return createElement('main',{className:'ai-book'},createElement('h1',null,'Interactive Book'),createElement(Lab,{language}));}const root=createRoot(document.getElementById('root'));window.__unmount=()=>root.unmount();root.render(createElement(App));`,resolveDir:process.cwd(),loader:'tsx'},bundle:true,write:false,platform:'browser',format:'esm',jsx:'automatic',define:{'process.env.NODE_ENV':'"production"'},metafile:true});
 assert(!Object.keys(bundle.metafile!.inputs).some(p=>/learning-(?:graph-gauge|quantum-hilbert-core)(?:\.ts|-validation\.ts|-evaluator\.ts|-validation-evaluator\.ts)/.test(p)));writeFileSync(root+'/reader-dependency-graph.json',JSON.stringify(bundle.metafile,null,2)+'\n');
 const codes={calculate:readFileSync('.runtime/learning/graph-gauge-worker/calculate.js','utf8'),validate:readFileSync('.runtime/learning/graph-gauge-worker/validate.js','utf8')};
 const browser=await chromium.launch({headless:true}),states=[];let accessibility=0;
 try{for(const language of ['ko','en']as const)for(const dark of [false,true])for(const width of [320,1440]){
  const context=await browser.newContext({viewport:{width,height:1000},deviceScaleFactor:width===320?2:1,reducedMotion:'reduce'});
  await context.addInitScript({content:`globalThis.__name=fn=>fn;window.__language=${JSON.stringify(language)};window.__counts={created:0,terminated:0,active:0};const Native=Worker;window.Worker=class extends Native{constructor(...args){super(...args);this.closed=false;window.__counts.created++;window.__counts.active++;}terminate(){if(!this.closed){this.closed=true;window.__counts.terminated++;window.__counts.active--;}super.terminate();}};`});
  const page=await context.newPage(),errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/__graph_gauge_teaching__/',r=>r.fulfill({contentType:'text/html',body:`<!doctype html><html lang="${language}" class="${dark?'dark':''}"><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>Graph gauge lesson</title>${links.map(h=>`<link rel="stylesheet" href="${h}">`).join('')}<style>${css}</style></head><body><div id="root"></div></body></html>`}));
  await page.route('**/learning-graph-gauge.worker.ts',r=>r.fulfill({contentType:'text/javascript',body:codes.calculate}));await page.route('**/learning-graph-gauge-validation.worker.ts',r=>r.fulfill({contentType:'text/javascript',body:codes.validate}));
  const mount=async()=>{await page.goto('http://127.0.0.1:4175/__graph_gauge_teaching__/');await page.addScriptTag({type:'module',content:bundle.outputFiles[0].text});};
  const ready=()=>page.locator('[data-graph-gauge-phase="ready"]').waitFor({timeout:30000}),released=()=>page.waitForFunction(()=>(window as unknown as Harness).__counts.active===0);
  const axe=async()=>{const v=(await new AxeBuilder({page}).analyze()).violations;assert.deepEqual(v.map(x=>({id:x.id,nodes:x.nodes.map(n=>n.target)})),[]);accessibility++;};
  await mount();await ready();const examples=page.getByLabel(language==='ko'?'전체 예제':'All examples',{exact:true});
  for(const[id,energy]of energies){await examples.selectOption(id);await ready();assert.equal(await page.locator('[data-graph-gauge-metric=total]').innerText(),energy,id);}await released();
  const before=await page.evaluate(()=>(window as unknown as Harness).__counts.created);
  for(const[index,s]of graphGaugeSteps.entries()){await page.getByRole('button',{name:`${index+1}. ${s.title[language]}`,exact:true}).click();assert.equal(await page.locator('.information-step > p').first().innerText(),s.body[language]);}
  assert.equal(await page.locator('.information-step').evaluate(el=>getComputedStyle(el).animationName),'none');
  for(let i=0;i<graphGaugeExperiments.length;i++){await page.getByLabel(language==='ko'?'안내 실험':'Guided experiment',{exact:true}).selectOption(String(i));await page.getByRole('button',{name:language==='ko'?'해설 보기':'Show explanation',exact:true}).click();assert.equal(await page.locator('[data-graph-gauge-guide-explanation]').innerText(),graphGaugeExperiments[i].explanation[language]);}
  for(const[index,q]of graphGaugeQuestions.entries()){const field=page.locator('.information-question').nth(index);await field.locator('input').nth(q.correct).check();assert((await field.locator('[role=status]').innerText()).includes(q.explanation[language]));}
  const operation=page.getByLabel(language==='ko'?'그래프에 표시할 조작':'Operation shown on the graph',{exact:true});await operation.selectOption('matter-only');assert.equal(await page.evaluate(()=>(window as unknown as Harness).__counts.created),before);
  await page.getByLabel(language==='ko'?'표에 표시할 연산자':'Operator shown in the table',{exact:true}).selectOption('transformed');assert.equal(await page.evaluate(()=>(window as unknown as Harness).__counts.created),before);await axe();
  await examples.selectOption('reverse-walk');await ready();await page.getByLabel(language==='ko'?'선택한 닫힌 경로':'Selected closed walk',{exact:true}).selectOption('1');assert.equal(await page.locator('[data-graph-gauge-metric=loop]').innerText(),'(0) + (-1)i');
  await page.getByRole('button',{name:language==='ko'?'첫 예제로 초기화':'Reset to first example',exact:true}).click();await ready();
  const matter=page.getByLabel(language==='ko'?'물질 목록: Re Im':'Matter rows: Re Im',{exact:true}),apply=page.getByRole('button',{name:language==='ko'?'입력 적용':'Apply input',exact:true});await matter.fill('2 0\n1 0\n1 0');await apply.click();await ready();assert.equal(await page.locator('[data-graph-gauge-metric=total]').innerText(),'2');
  await page.getByLabel(language==='ko'?'닫힌 경로: start: +edge −edge':'Closed walks: start: +edge -edge',{exact:true}).fill('0: +0');await apply.click();assert(await page.getByRole('alert').count());assert.equal(await page.locator('[data-graph-gauge-metric=total]').innerText(),'2');
  await released();const previousCounts=await page.evaluate(()=>(window as unknown as Harness).__counts);assert.equal(previousCounts.created,previousCounts.terminated);await mount();await ready();assert.equal(await examples.inputValue(),'custom');assert.equal(await page.locator('[data-graph-gauge-metric=total]').innerText(),'2');
  await page.getByRole('button',{name:language==='ko'?'첫 예제로 초기화':'Reset to first example',exact:true}).click();await ready();
  const plot=page.locator('[data-graph-gauge-plot]');await plot.scrollIntoViewIfNeeded();await page.locator('[data-graph-gauge-plot="3d"] canvas').waitFor();const canvas=plot.locator('canvas');await plot.getByRole('button',{name:language==='ko'?'정면':'Front',exact:true}).click();const first=await canvas.screenshot();await canvas.press('ArrowRight');await canvas.press('ArrowUp');assert(!first.equals(await canvas.screenshot()),'Keyboard camera changes the view');
  await canvas.evaluate(el=>{const ext=(el as HTMLCanvasElement).getContext('webgl2')?.getExtension('WEBGL_lose_context');if(!ext)throw Error('No context loss extension');ext.loseContext();});await page.locator('[data-graph-gauge-plot="2d"]').waitFor();await axe();
  await plot.getByRole('button',{name:language==='ko'?'3D 다시 시도':'Retry 3D',exact:true}).click();await page.locator('[data-graph-gauge-plot="3d"] canvas').waitFor();
  await plot.getByRole('button',{name:language==='ko'?'2D 보기':'Show 2D',exact:true}).click();await page.locator('[data-graph-gauge-plot="2d"] svg').waitFor();await plot.getByRole('button',{name:language==='ko'?'3D 보기':'Show 3D',exact:true}).click();await page.locator('[data-graph-gauge-plot="3d"] canvas').waitFor();
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));if(language==='en')assert(!/[가-힣]/.test(await page.locator('main').innerText()));
  await plot.screenshot({path:`${root}/${language}-${dark?'dark':'light'}-${width}.png`});await page.evaluate(()=>(window as unknown as Harness).__unmount());await released();
  const counts=await page.evaluate(()=>(window as unknown as Harness).__counts);assert.equal(counts.created,counts.terminated);assert.deepEqual(errors,[]);counts.created+=previousCounts.created;counts.terminated+=previousCounts.terminated;states.push({language,dark,width,dpr:width===320?2:1,examples:13,conceptSteps:8,explainedQuestions:7,customInputReload:true,keyboardCamera:true,webglLossRetry:true,overflow:false,...counts});console.log(JSON.stringify({cases:states.length,language,dark,width}));await context.close();
 }
 writeFileSync(root+'/native-verification.json',JSON.stringify({states,accessibility,errors:[],readerRegistered:false,productionNextAssetsVerified:false},null,2)+'\n');console.log(JSON.stringify({cases:8,accessibility}));
 }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
