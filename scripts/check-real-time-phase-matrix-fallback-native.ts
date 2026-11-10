/** Actual native calculation/validation with unavailable WebGL. */
import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {chromium} from 'playwright';
const root='.runtime/learning/real-time-phase-matrix',url=JSON.parse(readFileSync(root+'/review-server.json','utf8')).url as string;
async function main(){
  const browser=await chromium.launch({headless:true});
  try{
    const page=await browser.newPage();const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
    await page.addInitScript({content:"globalThis.__name=fn=>fn;const original=HTMLCanvasElement.prototype.getContext;Object.defineProperty(HTMLCanvasElement.prototype,'getContext',{value:function(type,...args){return type.includes('webgl')?null:Reflect.apply(original,this,[type,...args]);}});localStorage.setItem('suanlab-language','en');"});
    await page.goto(url);await page.getByRole('button',{name:'English',exact:true}).click();await page.waitForFunction(()=>document.documentElement.lang==='en');await page.locator('[data-real-time-phase="ready"]').waitFor({timeout:120000});await page.getByLabel('Matrix',{exact:true}).selectOption('hessian');const before=await page.locator('[data-real-time-selected]').innerText();
    await page.getByRole('button',{name:'Show 3D matrix',exact:true}).click();await page.getByText('3D is unavailable. Continue comparing with the heat map and exact tables.',{exact:true}).waitFor();assert.equal(await page.locator('canvas').count(),0);assert.equal(await page.locator('[data-real-time-selected]').innerText(),before);
    await page.getByLabel('Coefficient display scale',{exact:true}).selectOption('signed-log');assert.equal(await page.locator('[data-real-time-selected]').innerText(),before);await page.getByRole('button',{name:'Retry 3D',exact:true}).click();await page.getByText('3D is unavailable. Continue comparing with the heat map and exact tables.',{exact:true}).waitFor();assert.deepEqual(errors,[]);
    writeFileSync(root+'/native-fallback.json',JSON.stringify({webGLUnavailable:true,retryHandled:true,exactIntervalsPreserved:true,scaleControlsWork:true,errors},null,2)+'\n');console.log('WebGL fallback and exact interval preservation passed.');
  }finally{await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
