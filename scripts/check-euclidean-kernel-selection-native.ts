import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {chromium} from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import {euclideanKernelDefault,euclideanKernelExamples,type EuclideanKernelInput} from '../src/lib/learning-euclidean-kernel-input';
import {euclideanKernelInputKey} from '../src/lib/learning-euclidean-kernel-protocol';
import type {EuclideanKernelResult} from '../src/lib/learning-euclidean-kernel';
const root='.runtime/learning/euclidean-kernel-teaching';
type Counts={created:number;closed:number;active:number};
async function main(){
 const {url}=JSON.parse(readFileSync(root+'/review-server.json','utf8')),browser=await chromium.launch({headless:true});
 try{
  const context=await browser.newContext({viewport:{width:320,height:1000},reducedMotion:'reduce'});
  await context.addInitScript({content:'globalThis.__name=fn=>fn;'});
  await context.addInitScript(()=>{localStorage.setItem('suanlab-language','en');localStorage.setItem('theme','dark');const Native=Worker,active=new Set<Worker>(),counts={created:0,closed:0,active:0};(window as unknown as{__selectionCounts:Counts}).__selectionCounts=counts;window.Worker=class extends Native{constructor(url:string|URL,options?:WorkerOptions){super(url,options);active.add(this);counts.created++;counts.active=active.size;}terminate(){if(active.delete(this)){counts.closed++;counts.active=active.size;}super.terminate();}};});
  const page=await context.newPage(),errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
  const ready=(input:EuclideanKernelInput)=>page.waitForFunction(key=>document.querySelector('[data-euclidean-result-key]')?.getAttribute('data-euclidean-result-key')===key&&document.querySelector('[data-euclidean-phase]')?.getAttribute('data-euclidean-phase')==='ready'&&(window as unknown as{__selectionCounts:Counts}).__selectionCounts.active===0,euclideanKernelInputKey(input),{timeout:120000});
  const counts=()=>page.evaluate(()=>(window as unknown as{__selectionCounts:Counts}).__selectionCounts);
  const selected=(pair:string)=>page.locator('[data-euclidean-selected-entry]').innerText().then(text=>assert(text.includes(pair),text.slice(0,120)));
  await page.goto(url);await ready(euclideanKernelDefault);assert.deepEqual(await counts(),{created:2,closed:2,active:0});await selected('C[1,1]');
  const four=euclideanKernelExamples.find(e=>e.id==='four-intervals')!.input;
  await page.getByLabel('All examples',{exact:true}).selectOption('four-intervals');await ready(four);await selected('C[2,2]');assert.equal(await page.getByLabel('Highlighted time index',{exact:true}).inputValue(),'2');
  const input=euclideanKernelExamples.find(e=>e.id==='mixed-coprime-denominators')!.input;
  await page.getByLabel('All examples',{exact:true}).selectOption('mixed-coprime-denominators');await ready(input);await selected('C[17,17]');assert.equal(await page.getByLabel('Highlighted time index',{exact:true}).inputValue(),'17');
  const beforeDisplay=await counts();assert.equal(await page.getByRole('group',{name:'Position covariance C',exact:true}).getByRole('button').count(),1089);
  const fixture=JSON.parse(readFileSync('scripts/fixtures/euclidean-kernel-references.json','utf8'))as{cases:{id:string;expected:EuclideanKernelResult}[]},expected=fixture.cases.find(c=>c.id==='mixed-coprime-denominators')!.expected;
  await page.getByLabel('Displayed matrix',{exact:true}).selectOption('precision');await selected('Q[17,17]');assert((await page.locator('[data-euclidean-selected-entry]').innerText()).includes(expected.precision[16][16]));
  await page.getByLabel('Displayed matrix',{exact:true}).selectOption('position');await selected('C[17,17]');
  await page.getByRole('group',{name:'Position covariance C',exact:true}).getByRole('button').nth(17*33+18).click();await selected('C[17,18]');
  await page.getByLabel('Displayed matrix',{exact:true}).selectOption('precision');await selected('Q[17,18]');
  await page.getByLabel('Displayed matrix',{exact:true}).selectOption('position');await selected('C[17,18]');
  const slider=page.getByLabel('Highlighted time index',{exact:true});await slider.focus();await slider.press('Home');for(let i=0;i<3;i++)await slider.press('ArrowRight');assert.equal(await slider.inputValue(),'3');await selected('C[17,18]');assert.deepEqual(await counts(),beforeDisplay);
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=document.documentElement.clientWidth+1));
  const axe=await new AxeBuilder({page}).include('main').withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();assert.deepEqual(axe.violations.map(v=>v.id),[]);
  await page.reload();await ready(input);await selected('C[17,18]');assert.equal(await page.getByLabel('Highlighted time index',{exact:true}).inputValue(),'3');const afterReload=await counts();assert.equal(afterReload.created,afterReload.closed);assert.deepEqual(errors,[]);
  writeFileSync(root+'/selection-native.json',JSON.stringify({inputChangesHighlightDeclaredProbe:true,positionPrecisionPreservePhysicalPairs:true,offDiagonalRoundTrip:true,displayDoesNotCalculate:true,selectedTimeIndependentFromMatrixPair:true,reloadPreservesBothSelections:true,maximumIntervals:32,beforeReload:beforeDisplay,afterReload,accessibility:true,errors},null,2)+'\n');
 }finally{await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
