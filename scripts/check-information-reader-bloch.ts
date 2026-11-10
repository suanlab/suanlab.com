import assert from 'node:assert/strict';
import {writeFileSync} from 'node:fs';
import {chromium} from 'playwright';
const root='.runtime/learning/quantum-information-reader-integration',base=process.env.INFORMATION_PREVIEW_BASE??'http://127.0.0.1:4176';
async function main(){const browser=await chromium.launch({headless:true}),states=[],errors:string[]=[];try{
 for(const language of ['ko','en']as const)for(const dark of [false,true])for(const width of [320,1440]){
  const context=await browser.newContext({viewport:{width,height:1000},reducedMotion:'reduce'});await context.addInitScript(({language,dark})=>{localStorage.setItem('suanlab-language',language);localStorage.setItem('theme',dark?'dark':'light');},{language,dark});
  const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));await page.goto(base+'/book/online/physics-math-for-ai/binary-qubit-state-discrimination/');await page.locator('[data-information-phase="ready"]').waitFor({timeout:60000});
  const figures=page.locator('[data-qubit-bloch]');assert.equal(await figures.count(),2);
  for(const figure of await figures.all()){
   await figure.scrollIntoViewIfNeeded();await figure.locator('canvas').waitFor();assert.equal(await figure.getAttribute('data-qubit-bloch'),'3d');
   await figure.getByRole('button',{name:language==='ko'?'정면':'Front',exact:true}).click();await figure.getByRole('button',{name:language==='ko'?'위에서':'Top',exact:true}).click();
   const canvas=figure.locator('canvas');await canvas.focus();await canvas.press('ArrowRight');await canvas.press('ArrowUp');
   await canvas.evaluate(el=>{const ext=(el as HTMLCanvasElement).getContext('webgl2')?.getExtension('WEBGL_lose_context');if(!ext)throw Error('WebGL context loss unavailable');ext.loseContext();});
   await page.waitForFunction(el=>el?.getAttribute('data-qubit-bloch')==='2d',await figure.elementHandle());await figure.getByRole('button',{name:language==='ko'?'3D 다시 시도':'Retry 3D',exact:true}).click();await figure.locator('canvas').waitFor();assert.equal(await figure.getAttribute('data-qubit-bloch'),'3d');
  }
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));if(language==='en')assert(!/[가-힣]/.test(await page.locator('.ai-reader').innerText()));
  states.push({language,dark,width,blochFigures:2,keyboardCamera:true,webglLossFallbackAndRetry:true});await context.close();
 }
 assert.deepEqual(errors,[]);writeFileSync(root+'/reader-bloch.json',JSON.stringify({states,errors,productionReader:true,webglLossRetries:16},null,2)+'\n');console.log(JSON.stringify({cases:states.length,webglLossRetries:16}));
 }finally{await browser.close();}}
main().catch(error=>{console.error(error);process.exitCode=1;});
