import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {chromium} from 'playwright';
const root='.runtime/learning/gauge-orbits-teaching';
async function main(){const base=(JSON.parse(readFileSync(root+'/review-server.json','utf8'))as{url:string}).url,browser=await chromium.launch({headless:true}),states=[];
 try{for(const language of ['ko','en']as const){
  const context=await browser.newContext({viewport:{width:320,height:900},deviceScaleFactor:2,hasTouch:true,isMobile:true,reducedMotion:'no-preference'});
  await context.addInitScript(language=>{localStorage.setItem('suanlab-language',language);const counts={created:0,terminated:0};(window as unknown as{__inputWorkers:typeof counts}).__inputWorkers=counts;const Native=Worker;window.Worker=class extends Native{closed=false;constructor(url:string|URL,options?:WorkerOptions){super(url,options);counts.created++;}terminate(){if(!this.closed){this.closed=true;counts.terminated++;}super.terminate();}};},language);
  const page=await context.newPage(),errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));await page.goto(base);
  const ready=()=>page.locator('[data-gauge-orbits-phase="ready"]').waitFor({timeout:30000}),apply=page.getByRole('button',{name:language==='ko'?'입력 적용':'Apply input',exact:true});
  await ready();await page.getByLabel(language==='ko'?'전체 예제':'All examples',{exact:true}).selectOption('anchor-away-from-root');await ready();
  const anchors=page.getByLabel(language==='ko'?'앵커 노드 번호':'Anchor site indices',{exact:true});await anchors.fill('');await apply.click();await ready();assert.equal(await page.locator('[data-gauge-orbits-decision="anchored-jointEquivalent"]').innerText(),language==='ko'?'동치':'Equivalent');
  await page.getByLabel(language==='ko'?'간선 목록: from to':'Edge rows: from to',{exact:true}).fill('');for(const name of [language==='ko'?'왼쪽 링크: t minusOne':'Left links: t minusOne',language==='ko'?'오른쪽 링크: t minusOne':'Right links: t minusOne'])await page.getByLabel(name,{exact:true}).fill('');
  await page.getByLabel(language==='ko'?'신장 숲 간선 번호':'Spanning forest edge indices',{exact:true}).fill('');await page.getByLabel(language==='ko'?'성분별 루트 번호':'Component root indices',{exact:true}).fill('0 1 2');await apply.click();await ready();assert(await page.getByLabel(language==='ko'?'선택 좌표':'Selected coordinate',{exact:true}).isDisabled());
  await anchors.fill('1e0');await apply.click();assert(await page.getByRole('alert').count());assert(await page.getByLabel(language==='ko'?'선택 좌표':'Selected coordinate',{exact:true}).isDisabled());
  await page.getByRole('button',{name:language==='ko'?'첫 예제로 초기화':'Reset to first example',exact:true}).click();await ready();
  await page.getByLabel(language==='ko'?'개념 전환 효과 사용':'Animate concept transitions',{exact:true}).uncheck();await page.getByRole('button',{name:language==='ko'?'2. 트리 경로로 고정 위상을 만든다':'2. Build fixing phases along tree paths',exact:true}).click();assert.equal(await page.locator('.information-step').evaluate(el=>getComputedStyle(el).animationName),'none');
  const plot=page.locator('[data-gauge-orbits-plot]');await plot.scrollIntoViewIfNeeded();const canvas=page.locator('[data-gauge-orbits-plot="3d"] canvas');await canvas.waitFor();const before=await canvas.screenshot(),box=await canvas.boundingBox();assert(box);const client=await context.newCDPSession(page),x=box.x+.35*box.width,y=box.y+.4*box.height;
  await client.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x,y,id:1}]});await client.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x:x+.25*box.width,y:y+8,id:1}]});await client.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});assert(!before.equals(await canvas.screenshot()),'Touch rotates the graph camera');
  await page.getByRole('button',{name:'Dark',exact:true}).click();await page.waitForFunction(()=>document.documentElement.classList.contains('dark'));assert.equal(await page.locator('[data-gauge-orbits-decision="full-jointEquivalent"]').innerText(),language==='ko'?'동치':'Equivalent');
  const counts=await page.evaluate(()=>(window as unknown as{__inputWorkers:{created:number;terminated:number}}).__inputWorkers);assert.equal(counts.created,counts.terminated);assert.deepEqual(errors,[]);assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));states.push({language,width:320,touchCamera:true,anchorsEditor:true,emptyEdgesAndForest:true,invalidIndexRejected:true,motionToggle:true,themeSwitch:true,...counts});await context.close();
 }
 writeFileSync(root+'/input-native.json',JSON.stringify({states,errors:[],actualReviewServer:base},null,2)+'\n');console.log(JSON.stringify({inputCases:2,touchCameraCases:2}));
 }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
