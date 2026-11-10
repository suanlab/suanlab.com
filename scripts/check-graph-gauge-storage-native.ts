import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {chromium} from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import {graphGaugeExamples} from '../src/lib/learning-graph-gauge-examples';
const root='.runtime/learning/graph-gauge-teaching';
async function main(){const base=(JSON.parse(readFileSync(root+'/review-server.json','utf8'))as{url:string}).url,browser=await chromium.launch({headless:true}),states=[];
 try{for(const scenario of ['invalid-json','wrong-version','bad-vector','read-failure','write-failure']){
  const context=await browser.newContext({viewport:{width:1024,height:900},reducedMotion:'reduce'});
  await context.addInitScript(({scenario,input})=>{
   const key='suanlab-graph-gauge-v1';localStorage.setItem('suanlab-language','en');
   if(scenario==='invalid-json')localStorage.setItem(key,'{');if(scenario==='wrong-version')localStorage.setItem(key,JSON.stringify({version:2,value:input}));if(scenario==='bad-vector')localStorage.setItem(key,JSON.stringify({version:1,value:{...input,matter:input.matter.slice(0,1)}}));
   const get=Storage.prototype.getItem,set=Storage.prototype.setItem;
   if(scenario==='read-failure')Storage.prototype.getItem=function(k){if(k===key)throw Error('QA storage unavailable');return get.call(this,k);};
   if(scenario==='write-failure')Storage.prototype.setItem=function(k,v){if(k===key)throw Error('QA storage unavailable');set.call(this,k,v);};
   const counts={created:0,terminated:0};(window as unknown as{__storageWorkers:typeof counts}).__storageWorkers=counts;const Native=Worker;window.Worker=class extends Native{closed=false;constructor(url:string|URL,options?:WorkerOptions){super(url,options);counts.created++;}terminate(){if(!this.closed){this.closed=true;counts.terminated++;}super.terminate();}};
  },{scenario,input:graphGaugeExamples[0].input});
  const page=await context.newPage(),errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));await page.goto(base);await page.locator('[data-graph-gauge-phase="ready"]').waitFor({timeout:30000});
  const status=scenario.endsWith('failure')?'unavailable':'corrupt';await page.locator(`[data-graph-gauge-storage="${status}"]`).waitFor();assert.equal(await page.locator('[data-graph-gauge-metric=total]').innerText(),'0');
  await page.getByLabel('All examples',{exact:true}).selectOption('positive-flux');await page.locator('[data-graph-gauge-phase="ready"]').waitFor();assert.equal(await page.locator('[data-graph-gauge-metric=total]').innerText(),'3');
  const v=(await new AxeBuilder({page}).analyze()).violations;assert.deepEqual(v.map(x=>({id:x.id,nodes:x.nodes.map(n=>n.target)})),[]);assert.deepEqual(errors,[]);
  const counts=await page.evaluate(()=>(window as unknown as{__storageWorkers:{created:number;terminated:number}}).__storageWorkers);assert.equal(counts.created,counts.terminated);states.push({scenario,status,experimentUsable:true,initialFallback:true,accessibilityPassed:true,...counts});await context.close();
 }
 writeFileSync(root+'/storage-native.json',JSON.stringify({states,actualReviewServer:base,errors:[]},null,2)+'\n');console.log(JSON.stringify({storageCases:5,accessibilityCases:5}));
 }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
