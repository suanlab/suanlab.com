import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {chromium} from 'playwright';
const root='.runtime/learning/graph-gauge-teaching';
async function main(){const base=(JSON.parse(readFileSync(root+'/review-server.json','utf8'))as{url:string}).url,browser=await chromium.launch({headless:true}),states=[];
 try{for(const language of ['ko','en']as const){
  const context=await browser.newContext({viewport:{width:320,height:900},deviceScaleFactor:2,hasTouch:true,isMobile:true,reducedMotion:'no-preference'});
  await context.addInitScript(language=>{localStorage.setItem('suanlab-language',language);const counts={created:0,terminated:0};(window as unknown as{__inputWorkers:typeof counts}).__inputWorkers=counts;const Native=Worker;window.Worker=class extends Native{closed=false;constructor(url:string|URL,options?:WorkerOptions){super(url,options);counts.created++;}terminate(){if(!this.closed){this.closed=true;counts.terminated++;}super.terminate();}};},language);
  const page=await context.newPage(),errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));await page.goto(base);
  const ready=()=>page.locator('[data-graph-gauge-phase="ready"]').waitFor({timeout:30000}),metric=(key:string)=>page.locator(`[data-graph-gauge-metric=${key}]`).innerText(),apply=page.getByRole('button',{name:language==='ko'?'입력 적용':'Apply input',exact:true});
  await ready();await page.getByLabel(language==='ko'?'전체 예제':'All examples',{exact:true}).selectOption('positive-flux');await ready();
  const walks=page.getByLabel(language==='ko'?'닫힌 경로: start: +edge −edge':'Closed walks: start: +edge -edge',{exact:true});await walks.fill('0: -2 -1 -0');await apply.click();await ready();assert.equal(await metric('loop'),'(0) + (-1)i');
  const frames=page.getByLabel(language==='ko'?'기준 목록: t minusOne':'Frame rows: t minusOne',{exact:true});await frames.fill('0 false\n0 false\n0 false');await apply.click();await ready();assert.equal(await metric('fieldOnly'),'3');
  await page.getByLabel(language==='ko'?'간선 목록: from to t minusOne weight':'Edge rows: from to t minusOne weight',{exact:true}).fill('');await walks.fill('');await apply.click();await ready();assert.equal(await metric('total'),'0');assert.equal(await metric('components'),'3');assert.equal(await metric('cycles'),'0');assert(await page.getByLabel(language==='ko'?'선택한 닫힌 경로':'Selected closed walk',{exact:true}).isDisabled());
  await frames.fill('1 true\n0 false\n0 false');await apply.click();assert(await page.getByRole('alert').count());assert.equal(await metric('components'),'3');
  await page.getByRole('button',{name:language==='ko'?'첫 예제로 초기화':'Reset to first example',exact:true}).click();await ready();
  await page.getByLabel(language==='ko'?'개념 전환 효과 사용':'Animate concept transitions',{exact:true}).uncheck();await page.getByRole('button',{name:language==='ko'?'2. 공변과 불변':'2. Covariance and invariance',exact:true}).click();assert.equal(await page.locator('.information-step').evaluate(el=>getComputedStyle(el).animationName),'none');
  const plot=page.locator('[data-graph-gauge-plot]');await plot.scrollIntoViewIfNeeded();const canvas=page.locator('[data-graph-gauge-plot="3d"] canvas');await canvas.waitFor();const before=await canvas.screenshot(),box=await canvas.boundingBox();assert(box);const client=await context.newCDPSession(page),x=box.x+.35*box.width,y=box.y+.4*box.height;
  await client.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x,y,id:1}]});await client.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x:x+.25*box.width,y:y+8,id:1}]});await client.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});assert(!before.equals(await canvas.screenshot()),'Touch rotates the graph camera');
  await page.getByRole('button',{name:'Dark',exact:true}).click();await page.waitForFunction(()=>document.documentElement.classList.contains('dark'));assert.equal(await metric('total'),'0');
  const counts=await page.evaluate(()=>(window as unknown as{__inputWorkers:{created:number;terminated:number}}).__inputWorkers);assert.equal(counts.created,counts.terminated);assert.deepEqual(errors,[]);assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));states.push({language,width:320,touchCamera:true,reverseWalkEditor:true,identityFrameEditor:true,emptyEdgesAndWalks:true,invalidPoleRejected:true,motionToggle:true,themeSwitch:true,...counts});await context.close();
 }
 writeFileSync(root+'/input-native.json',JSON.stringify({states,errors:[],actualReviewServer:base},null,2)+'\n');console.log(JSON.stringify({inputCases:2,touchCameraCases:2}));
 }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
