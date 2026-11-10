import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {build} from 'esbuild';
import {chromium} from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import {gaugeOrbitSteps,gaugeOrbitExperiments,gaugeOrbitQuestions} from '../src/data/gauge-orbits-teaching';
import {gaugeOrbitExamples} from '../src/lib/learning-gauge-orbits-examples';
import {calculateGaugeOrbits} from '../src/lib/learning-gauge-orbits';
const results=gaugeOrbitExamples.map(c=>({id:c.id,result:calculateGaugeOrbits(c.input)}));
const root='.runtime/learning/gauge-orbits-teaching';
type Harness={__language:string;__counts:{created:number;terminated:number;active:number};__unmount:()=>void};
async function main(){
 const html=readFileSync('out/book/interactive/index.html','utf8'),links=[...html.matchAll(/<link[^>]*>/g)].map(x=>x[0]).filter(x=>x.includes('rel="stylesheet"')).map(x=>x.match(/href="([^"]+)"/)?.[1]).filter((v):v is string=>!!v),css=['src/styles/qubit-lab.css','src/styles/information-lab.css','src/styles/scalar-field-lab.css'].map(p=>readFileSync(p,'utf8')).join('\n');
 const bundle=await build({stdin:{contents:`import{createElement,useState}from'react';import{createRoot}from'react-dom/client';import Lab from'./src/components/learning/gauge-orbits-lab';function App(){const[language,setLanguage]=useState(window.__language);window.__setLanguage=setLanguage;return createElement('main',{className:'ai-book'},createElement('h1',null,'Interactive Book'),createElement(Lab,{language}));}const root=createRoot(document.getElementById('root'));window.__unmount=()=>root.unmount();root.render(createElement(App));`,resolveDir:process.cwd(),loader:'tsx'},bundle:true,write:false,platform:'browser',format:'esm',jsx:'automatic',define:{'process.env.NODE_ENV':'"production"'},metafile:true});
 assert(!Object.keys(bundle.metafile!.inputs).some(p=>/learning-(?:gauge-orbits|quantum-hilbert-core)(?:\.ts|-validation\.ts|-evaluator\.ts|-validation-evaluator\.ts)/.test(p)));writeFileSync(root+'/reader-dependency-graph.json',JSON.stringify(bundle.metafile,null,2)+'\n');
 const codes={calculate:readFileSync('.runtime/learning/gauge-orbits-worker/calculate.js','utf8'),validate:readFileSync('.runtime/learning/gauge-orbits-worker/validate.js','utf8')};
 const browser=await chromium.launch({headless:true}),states=[];let accessibility=0;
 try{for(const language of ['ko','en']as const)for(const dark of [false,true])for(const width of [320,1440]){
  const context=await browser.newContext({viewport:{width,height:1000},deviceScaleFactor:width===320?2:1,reducedMotion:'reduce'});
  await context.addInitScript({content:`globalThis.__name=fn=>fn;window.__language=${JSON.stringify(language)};window.__counts={created:0,terminated:0,active:0};const Native=Worker;window.Worker=class extends Native{constructor(...args){super(...args);this.closed=false;window.__counts.created++;window.__counts.active++;}terminate(){if(!this.closed){this.closed=true;window.__counts.terminated++;window.__counts.active--;}super.terminate();}};`});
  const page=await context.newPage(),errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/__gauge_orbits_teaching__/',r=>r.fulfill({contentType:'text/html',body:`<!doctype html><html lang="${language}" class="${dark?'dark':''}"><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>Graph gauge lesson</title>${links.map(h=>`<link rel="stylesheet" href="${h}">`).join('')}<style>${css}</style></head><body><div id="root"></div></body></html>`}));
  await page.route('**/learning-gauge-orbits.worker.ts',r=>r.fulfill({contentType:'text/javascript',body:codes.calculate}));await page.route('**/learning-gauge-orbits-validation.worker.ts',r=>r.fulfill({contentType:'text/javascript',body:codes.validate}));
  const mount=async()=>{await page.goto('http://127.0.0.1:4175/__gauge_orbits_teaching__/');await page.addScriptTag({type:'module',content:bundle.outputFiles[0].text});};
  const ready=()=>page.locator('[data-gauge-orbits-phase="ready"]').waitFor({timeout:30000}),released=()=>page.waitForFunction(()=>(window as unknown as Harness).__counts.active===0);
  const axe=async()=>{const v=(await new AxeBuilder({page}).analyze()).violations;assert.deepEqual(v.map(x=>({id:x.id,nodes:x.nodes.map(n=>n.target)})),[]);accessibility++;};
  await mount();await ready();const examples=page.getByLabel(language==='ko'?'전체 예제':'All examples',{exact:true});
  for(const {id,result}of results){await examples.selectOption(id);await ready();for(const group of ['full','anchored']as const)for(const kind of ['connectionsEquivalent','jointEquivalent']as const)assert.equal(await page.locator(`[data-gauge-orbits-decision="${group}-${kind}"]`).innerText(),result.comparison[group][kind]?(language==='ko'?'동치':'Equivalent'):(language==='ko'?'동치 아님':'Not equivalent'),id);}await released();
  const before=await page.evaluate(()=>(window as unknown as Harness).__counts.created);
  for(const[index,s]of gaugeOrbitSteps.entries()){await page.getByRole('button',{name:`${index+1}. ${s.title[language]}`,exact:true}).click();assert.equal(await page.locator('.information-step > p').first().innerText(),s.body[language]);}
  assert.equal(await page.locator('.information-step').evaluate(el=>getComputedStyle(el).animationName),'none');
  for(let i=0;i<gaugeOrbitExperiments.length;i++){await page.getByLabel(language==='ko'?'안내 실험':'Guided experiment',{exact:true}).selectOption(String(i));await page.getByRole('button',{name:language==='ko'?'해설 보기':'Show explanation',exact:true}).click();assert.equal(await page.locator('[data-gauge-orbits-guide-explanation]').innerText(),gaugeOrbitExperiments[i].explanation[language]);}
  for(const[index,q]of gaugeOrbitQuestions.entries()){const field=page.locator('.information-question').nth(index);await field.locator('input').nth(q.correct).check();assert((await field.locator('[role=status]').innerText()).includes(q.explanation[language]));}
  const scheme=page.getByLabel(language==='ko'?'허용 군과 고정 방식':'Allowed group and fixing scheme',{exact:true});
  await scheme.selectOption('anchored');await page.getByLabel(language==='ko'?'표시할 상태':'State to display',{exact:true}).selectOption('right');await page.getByLabel(language==='ko'?'그래프 표현':'Graph representation',{exact:true}).selectOption('fixed');await page.getByLabel(language==='ko'?'확인할 변환':'Witness to inspect',{exact:true}).selectOption('connectionWitness');assert.equal(await page.evaluate(()=>(window as unknown as Harness).__counts.created),before);await axe();
  await examples.selectOption('two-incompatible-anchors');await ready();await scheme.selectOption('anchored');assert((await page.locator('[data-gauge-orbits-coordinate]').innerText()).includes(language==='ko'?'열린 앵커 곱':'Open anchor product'));
  await page.getByRole('button',{name:language==='ko'?'첫 예제로 초기화':'Reset to first example',exact:true}).click();await ready();
  const anchors=page.getByLabel(language==='ko'?'앵커 노드 번호':'Anchor site indices',{exact:true}),apply=page.getByRole('button',{name:language==='ko'?'입력 적용':'Apply input',exact:true});await anchors.fill('0, 1');await apply.click();await ready();assert.equal(await page.locator('[data-gauge-orbits-decision="anchored-jointEquivalent"]').innerText(),language==='ko'?'동치 아님':'Not equivalent');
  await page.getByLabel(language==='ko'?'신장 숲 간선 번호':'Spanning forest edge indices',{exact:true}).fill('0');await apply.click();assert(await page.getByRole('alert').count());assert.equal(await page.locator('[data-gauge-orbits-decision="anchored-jointEquivalent"]').innerText(),language==='ko'?'동치 아님':'Not equivalent');
  await released();const previousCounts=await page.evaluate(()=>(window as unknown as Harness).__counts);assert.equal(previousCounts.created,previousCounts.terminated);await mount();await ready();assert.equal(await anchors.inputValue(),'0, 1');assert.equal(await page.locator('[data-gauge-orbits-decision="anchored-jointEquivalent"]').innerText(),language==='ko'?'동치 아님':'Not equivalent');
  await page.getByRole('button',{name:language==='ko'?'첫 예제로 초기화':'Reset to first example',exact:true}).click();await ready();
  const plot=page.locator('[data-gauge-orbits-plot]');await plot.scrollIntoViewIfNeeded();await page.locator('[data-gauge-orbits-plot="3d"] canvas').waitFor();const canvas=plot.locator('canvas');await plot.getByRole('button',{name:language==='ko'?'정면':'Front',exact:true}).click();const first=await canvas.screenshot();await canvas.press('ArrowRight');await canvas.press('ArrowUp');assert(!first.equals(await canvas.screenshot()),'Keyboard camera changes the view');
  await canvas.evaluate(el=>{const ext=(el as HTMLCanvasElement).getContext('webgl2')?.getExtension('WEBGL_lose_context');if(!ext)throw Error('No context loss extension');ext.loseContext();});await page.locator('[data-gauge-orbits-plot="2d"]').waitFor();await axe();
  await plot.getByRole('button',{name:language==='ko'?'3D 다시 시도':'Retry 3D',exact:true}).click();await page.locator('[data-gauge-orbits-plot="3d"] canvas').waitFor();
  await plot.getByRole('button',{name:language==='ko'?'2D 보기':'Show 2D',exact:true}).click();await page.locator('[data-gauge-orbits-plot="2d"] svg').waitFor();await plot.getByRole('button',{name:language==='ko'?'3D 보기':'Show 3D',exact:true}).click();await page.locator('[data-gauge-orbits-plot="3d"] canvas').waitFor();
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));if(language==='en')assert(!/[가-힣]/.test(await page.locator('main').innerText()));
  await plot.screenshot({path:`${root}/${language}-${dark?'dark':'light'}-${width}.png`});await page.evaluate(()=>(window as unknown as Harness).__unmount());await released();
  const counts=await page.evaluate(()=>(window as unknown as Harness).__counts);assert.equal(counts.created,counts.terminated);assert.deepEqual(errors,[]);counts.created+=previousCounts.created;counts.terminated+=previousCounts.terminated;states.push({language,dark,width,dpr:width===320?2:1,examples:17,conceptSteps:9,explainedQuestions:8,customInputReload:true,keyboardCamera:true,webglLossRetry:true,overflow:false,...counts});console.log(JSON.stringify({cases:states.length,language,dark,width}));await context.close();
 }
 writeFileSync(root+'/native-verification.json',JSON.stringify({states,accessibility,errors:[],readerRegistered:false,productionNextAssetsVerified:false},null,2)+'\n');console.log(JSON.stringify({cases:8,accessibility}));
 }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
