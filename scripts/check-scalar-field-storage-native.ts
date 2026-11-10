import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {chromium} from 'playwright';
import AxeBuilder from '@axe-core/playwright';
const root='.runtime/learning/scalar-field-teaching';
async function main(){const base=(JSON.parse(readFileSync(root+'/review-server.json','utf8'))as{url:string}).url,browser=await chromium.launch({headless:true}),states=[];try{
 for(const scenario of ['invalid-json','wrong-version','bad-vector','read-failure','write-failure']){
  const context=await browser.newContext({viewport:{width:1024,height:900},reducedMotion:'reduce'});
  await context.addInitScript(scenario=>{
   const key='suanlab-scalar-field-v1';localStorage.setItem('suanlab-language','en');
   if(scenario==='invalid-json')localStorage.setItem(key,'{');if(scenario==='wrong-version')localStorage.setItem(key,JSON.stringify({version:2,value:{}}));if(scenario==='bad-vector')localStorage.setItem(key,JSON.stringify({version:1,value:{boundary:'periodic',spacing:'1',speed:'1',quadratic:'0',quartic:'0',position:['1'],velocity:['0','0','0']}}));
   const get=Storage.prototype.getItem,set=Storage.prototype.setItem;
   if(scenario==='read-failure')Storage.prototype.getItem=function(k){if(k===key)throw Error('QA storage unavailable');return get.call(this,k);};
   if(scenario==='write-failure')Storage.prototype.setItem=function(k,v){if(k===key)throw Error('QA storage unavailable');set.call(this,k,v);};
   const counts={created:0,terminated:0};(window as unknown as{__storageWorkers:typeof counts}).__storageWorkers=counts;const Native=Worker;window.Worker=class extends Native{closed=false;constructor(url:string|URL,options?:WorkerOptions){super(url,options);counts.created++;}terminate(){if(!this.closed){this.closed=true;counts.terminated++;}super.terminate();}};
  },scenario);
  const page=await context.newPage(),errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));await page.goto(base);await page.locator('[data-scalar-phase="ready"]').waitFor({timeout:30000});
  const storage=scenario.endsWith('failure')?'unavailable':'corrupt';await page.locator(`[data-scalar-storage="${storage}"]`).waitFor();assert.equal(await page.locator('[data-scalar-metric=H]').innerText(),'7/2');
  await page.getByLabel('All examples',{exact:true}).selectOption('fixed-uniform');await page.locator('[data-scalar-phase="ready"]').waitFor();assert.equal(await page.locator('[data-scalar-metric=H]').innerText(),'1');
  const violations=(await new AxeBuilder({page}).analyze()).violations;assert.deepEqual(violations.map(v=>({id:v.id,nodes:v.nodes.map(n=>n.target)})),[]);assert.deepEqual(errors,[]);
  const counts=await page.evaluate(()=>(window as unknown as{__storageWorkers:{created:number;terminated:number}}).__storageWorkers);assert.equal(counts.created,counts.terminated);
  states.push({scenario,status:storage,experimentUsable:true,initialFallback:true,accessibilityPassed:true,...counts});await context.close();
 }
 writeFileSync(root+'/storage-native.json',JSON.stringify({states,actualReviewServer:base,errors:[]},null,2)+'\n');console.log(JSON.stringify({storageCases:5,accessibilityCases:5}));
 }finally{await browser.close();}}
main().catch(error=>{console.error(error);process.exitCode=1;});
