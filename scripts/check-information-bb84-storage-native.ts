import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {build} from 'esbuild';
import {chromium} from 'playwright';
import {bb84Examples} from '../src/lib/learning-information-bb84-input';
const runtime='.runtime/learning/quantum-information-bb84-teaching',key='suanlab-information-v1-bb84';
type Harness=Window&{__unmount:()=>void;__storageMode:string;__counts:{created:number;terminated:number;active:number}};
async function main(){
 const result=await build({stdin:{contents:`import{createElement}from'react';import{createRoot}from'react-dom/client';import Lab from'./src/components/learning/information-bb84-lab';const root=createRoot(document.getElementById('root'));window.__unmount=()=>root.unmount();root.render(createElement(Lab,{language:'en'}));`,resolveDir:process.cwd(),loader:'tsx'},bundle:true,write:false,platform:'browser',format:'esm',jsx:'automatic',define:{'process.env.NODE_ENV':'"production"'}});
 const codes={calculate:readFileSync('.runtime/learning/quantum-information-workers/calculate.js','utf8'),validate:readFileSync('.runtime/learning/quantum-information-workers/validate.js','utf8')};
 const browser=await chromium.launch({headless:true}),reports=[];let created=0,terminated=0;
 try{
  for(const mode of ['restore','corrupt-json','wrong-family','get-blocked','set-blocked']){
   const context=await browser.newContext();await context.addInitScript({content:`globalThis.__name=(fn)=>fn;window.__storageMode='normal';window.__counts={created:0,terminated:0,active:0};const W=Worker;window.Worker=class extends W{constructor(...args){super(...args);window.__counts.created++;window.__counts.active++;this.closed=false;}terminate(){if(!this.closed){this.closed=true;window.__counts.terminated++;window.__counts.active--;}super.terminate();}};const get=Storage.prototype.getItem,set=Storage.prototype.setItem;Storage.prototype.getItem=function(k){if(window.__storageMode==='get-blocked')throw Error('read blocked');return get.call(this,k);};Storage.prototype.setItem=function(k,v){if(window.__storageMode==='set-blocked')throw Error('write blocked');return set.call(this,k,v);};`});
   const page=await context.newPage(),errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
   await page.route('**/__information_bb84_storage__/',r=>r.fulfill({contentType:'text/html',body:'<!doctype html><html lang="en"><head><title>Input storage check</title></head><body><div id="root"></div></body></html>'}));
   await page.route('**/learning-quantum-information.worker.ts',r=>r.fulfill({contentType:'text/javascript',body:codes.calculate}));await page.route('**/learning-quantum-information-validation.worker.ts',r=>r.fulfill({contentType:'text/javascript',body:codes.validate}));
   await page.goto('http://127.0.0.1:4175/__information_bb84_storage__/');
   const stored=mode==='restore'?JSON.stringify({version:1,value:bb84Examples.find(x=>x.id==='no-attack')!.input}):mode==='corrupt-json'?'invalid':mode==='wrong-family'?JSON.stringify({version:1,value:{kind:'capacity',probability:'1/2',input:['0','0','0']}}):null;
   await page.evaluate(({key,stored,mode})=>{if(stored!==null)localStorage.setItem(key,stored);(window as unknown as Harness).__storageMode=mode;},{key,stored,mode});
   const mount=async()=>{await page.addScriptTag({type:'module',content:result.outputFiles[0].text});await page.locator('[data-information-phase=ready]').waitFor({timeout:30000});};
   const unmount=async()=>{await page.evaluate(()=>(window as unknown as Harness).__unmount());const c=await page.evaluate(()=>(window as unknown as Harness).__counts);assert.equal(c.active,0);created+=c.created;terminated+=c.terminated;};
   await mount();assert.equal(await page.locator('[data-bb84-metric=qber]').innerText(),mode==='restore'?'0':'0.25');
   if(mode.startsWith('corrupt')||mode==='wrong-family')assert((await page.locator('body').innerText()).includes('saved input format was invalid'));
   if(mode.endsWith('blocked'))assert((await page.locator('body').innerText()).includes('Storage is unavailable'));
   await page.getByLabel('All examples',{exact:true}).selectOption('tiny-attack');await page.locator('[data-information-phase=ready]').waitFor();assert.equal(await page.locator('[data-bb84-metric=qber]').innerText(),'2.5000e-31');
   if(mode==='restore'){
    await unmount();await page.reload();await mount();assert.equal(await page.getByLabel('All examples',{exact:true}).inputValue(),'tiny-attack');assert.equal(await page.locator('[data-bb84-metric=qber]').innerText(),'2.5000e-31');
   }
   await unmount();assert.deepEqual(errors,[]);reports.push({mode,passed:true,reloadVerified:mode==='restore'});await context.close();
  }
  assert.equal(created,terminated);writeFileSync(`${runtime}/storage-native-verification.json`,JSON.stringify({reports,created,terminated},null,2)+'\n');console.log(JSON.stringify({scenarios:reports.length,created,terminated}));
 }finally{await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
