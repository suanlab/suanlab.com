import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {build} from 'esbuild';
import {chromium} from 'playwright';
import AxeBuilder from '@axe-core/playwright';
const runtime='.runtime/learning/quantum-information-ensemble-teaching';
async function main(){
 const base=readFileSync('out/book/interactive/index.html','utf8');
 const links=[...base.matchAll(/<link[^>]*>/g)].map(x=>x[0]).filter(x=>x.includes('rel="stylesheet"')).map(x=>x.match(/href="([^"]+)"/)?.[1]).filter((x):x is string=>!!x);
 const extraCss=['src/styles/qubit-lab.css','src/styles/information-lab.css'].map(p=>readFileSync(p,'utf8')).join('\n');
 const codes={calculate:readFileSync('.runtime/learning/quantum-information-workers/calculate.js','utf8'),validate:readFileSync('.runtime/learning/quantum-information-workers/validate.js','utf8')};
 const result=await build({stdin:{contents:`import{createElement,useState}from'react';import{createRoot}from'react-dom/client';import Lab from'./src/components/learning/information-ensemble-lab';function App(){const[language,setLanguage]=useState(window.__language);window.__languageChange=setLanguage;return createElement('main',{className:'ai-book'},createElement('h1',null,'Interactive Book'),createElement(Lab,{language}));}const root=createRoot(document.getElementById('root'));window.__unmount=()=>root.unmount();root.render(createElement(App));`,resolveDir:process.cwd(),loader:'tsx'},bundle:true,write:false,platform:'browser',format:'esm',jsx:'automatic',define:{'process.env.NODE_ENV':'"production"'},metafile:true});
 assert(!Object.keys(result.metafile!.inputs).some(p=>/learning-quantum-hilbert-core|learning-quantum-information-(?:calculation|response|.*validator|evaluator)/.test(p)));
 writeFileSync(`${runtime}/reader-dependency-graph.json`,JSON.stringify(result.metafile,null,2)+'\n');
 const browser=await chromium.launch({headless:true}),reports=[];let created=0,terminated=0;
 try{
  for(const language of ['ko','en'])for(const dark of [false,true])for(const width of [320,1600]){
   const context=await browser.newContext({viewport:{width,height:950},reducedMotion:'reduce'});await context.addInitScript({content:`globalThis.__name=(fn)=>fn;window.__language=${JSON.stringify(language)};window.__workerCount={created:0,terminated:0,active:0};const Native=Worker;window.Worker=class extends Native{constructor(...args){super(...args);window.__workerCount.created++;window.__workerCount.active++;this.closed=false;}terminate(){if(!this.closed){this.closed=true;window.__workerCount.terminated++;window.__workerCount.active--;}super.terminate();}};`});
   const page=await context.newPage(),errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
   await page.route('**/__information_ensemble__/',r=>r.fulfill({contentType:'text/html',body:`<!doctype html><html lang="${language}" class="${dark?'dark':''}"><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>Ensemble lesson</title>${links.map(h=>`<link rel="stylesheet" href="${h}">`).join('')}<style>${extraCss}</style></head><body><div id="root"></div></body></html>`}));
   await page.route('**/learning-quantum-information.worker.ts',r=>r.fulfill({contentType:'text/javascript',body:codes.calculate}));await page.route('**/learning-quantum-information-validation.worker.ts',r=>r.fulfill({contentType:'text/javascript',body:codes.validate}));
   await page.goto('http://127.0.0.1:4175/__information_ensemble__/');await page.addScriptTag({type:'module',content:result.outputFiles[0].text});
   const ready=()=>page.locator('[data-information-phase="ready"]').waitFor({timeout:30000});await ready().catch(async()=>{throw Error(JSON.stringify({errors,body:(await page.locator('body').innerText()).slice(0,2000)}));});
   const examples=page.getByLabel(language==='ko'?'전체 예제':'All examples',{exact:true});
   const expected=[['helstrom-table',.5310044064107188,.7219280948873623],['z-table',.4330398630967478,.7219280948873623],['constant-guess',0,.010503300578984905],['biased-z',.010503300578984905,.010503300578984905],['orthogonal-z',1,1],['same-mixed',0,0],['same-pure',0,0],['certain-label',0,0],['y-z',0,1],['y-helstrom',1,1],['mixed-z',.02904940554533136,.02904940554533136],['tiny-z',1.8033688011112042e-61,1.8033688011112042e-61]]as const;
   for(const [name,information,chi]of expected){
    await examples.selectOption(name);await ready();
    for(const [metric,value]of [['information',information],['chi',chi]]as const){const actual=await page.locator(`[data-information-metric=${metric}]`).innerText();assert(Math.abs(Number(actual)-value)<=Math.max(1e-65,Math.abs(value)*1e-4),`${name} ${metric}: ${actual}`);}
    if(name==='biased-z'||name==='mixed-z'||name==='tiny-z')assert.equal(await page.locator('[data-ensemble-gap]').getAttribute('data-ensemble-gap'),'includes-zero');
   }
   assert.match(await page.locator('[data-information-metric=information]').innerText(),/e-61/);
   const reset=()=>page.getByRole('button',{name:language==='ko'?'첫 예제로 초기화':'Reset to first example',exact:true}).click();
   await reset();await ready();
   const x=page.getByLabel('r0 · x',{exact:true}),y=page.getByLabel('r0 · y',{exact:true});await x.fill('1');await y.fill('1');await page.getByRole('button',{name:language==='ko'?'입력 적용':'Apply input',exact:true}).click();await page.locator('[data-information-phase=error]').waitFor();assert.equal(await page.locator('[data-information-result]').count(),0);
   await reset();await ready();await x.fill('0.2');await page.getByRole('button',{name:language==='ko'?'입력 적용':'Apply input',exact:true}).click();assert(await page.getByRole('alert').count());assert.match(await page.locator('[data-information-metric=information]').innerText(),/^0\.531/);
   await reset();await ready();
   await page.getByLabel(language==='ko'?'측정 기록 선택':'Measurement record',{exact:true}).selectOption('z');await page.getByRole('button',{name:language==='ko'?'입력 적용':'Apply input',exact:true}).click();await ready();assert.match(await page.locator('[data-information-metric=information]').innerText(),/^0\.433/);assert.equal(await page.locator('[data-ensemble-measurement]').getAttribute('data-ensemble-measurement'),'z');
   const before=await page.evaluate(()=>(window as unknown as {__workerCount:{created:number}}).__workerCount.created);
   await page.getByRole('button',{name:language==='ko'?'5. 맞힐 확률과 정보량은 다른 목표다':'5. Guessing accuracy and information are different objectives',exact:true}).click();
   assert.equal(await page.locator('.information-step').evaluate(el=>getComputedStyle(el).animationName),'none');
   await page.getByLabel(language==='ko'?'안내 실험':'Guided experiment',{exact:true}).selectOption('2');await page.getByRole('button',{name:language==='ko'?'해설 보기':'Show explanation',exact:true}).click();await page.locator('[data-information-guide-explanation]').waitFor();
   for(const q of await page.locator('.information-question').all()){await q.locator('input').first().check();assert(await q.locator('[role=status]').count());}
   await page.getByText(language==='ko'?'cq 상태와 원래 차이 구간 확인':'Inspect the cq state and raw difference intervals',{exact:true}).click();assert.equal(await page.evaluate(()=>(window as unknown as {__workerCount:{created:number}}).__workerCount.created),before);
   const violations=(await new AxeBuilder({page}).analyze()).violations;assert.equal(violations.length,0,JSON.stringify(violations.map(v=>({id:v.id,nodes:v.nodes.map(n=>n.target)}))));
   assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),JSON.stringify(await page.evaluate(()=>Array.from(document.querySelectorAll('main,section,figure,.qubit-scroll,.information-bloch-pair,.ai-discrete-table')).map(e=>({tag:e.tagName,cls:e.className,width:e.getBoundingClientRect().width,right:e.getBoundingClientRect().right})).filter(e=>e.right>innerWidth+1))));if(language==='en')assert(!/[가-힣]/.test(await page.locator('main').innerText()));
   const storage=await page.evaluate(()=>localStorage.getItem('suanlab-information-v1-ensemble'));assert(storage?.includes('ensemble'));
   await page.screenshot({path:`${runtime}/${language}-${dark?'dark':'light'}-${width}.png`,fullPage:true});
   await page.evaluate(()=>{(window as unknown as {__unmount:()=>void}).__unmount();});const count=await page.evaluate(()=>(window as unknown as {__workerCount:{created:number;terminated:number;active:number}}).__workerCount);assert.equal(count.active,0);created+=count.created;terminated+=count.terminated;assert.deepEqual(errors,[]);
   reports.push({language,dark,width,examples:12,invalidPhysicalHidden:true,invalidDraftRetainsApplied:true,axeViolations:0,overflow:false,viewControlsCreateWorkers:false,...count});await context.close();
  }
  writeFileSync(`${runtime}/native-verification.json`,JSON.stringify({reports,created,terminated,readerRegistered:false},null,2)+'\n');console.log(JSON.stringify({scenarios:reports.length,created,terminated}));
 }finally{await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
