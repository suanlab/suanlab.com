import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {chromium} from 'playwright';
import AxeBuilder from '@axe-core/playwright';
const root='.runtime/learning/real-time-phase-matrix',url=JSON.parse(readFileSync(root+'/review-server.json','utf8')).url as string;
async function main(){
  const browser=await chromium.launch({headless:true,args:['--enable-unsafe-swiftshader']}),rows:unknown[]=[],errors:string[]=[];
  try{
    for(const language of ['ko','en'] as const)for(const theme of ['light','dark'] as const)for(const width of [320,1440]){
      const context=await browser.newContext({viewport:{width,height:1000},hasTouch:width===320,isMobile:width===320,reducedMotion:'reduce'});
      await context.addInitScript(({language,theme})=>{localStorage.setItem('suanlab-language',language);localStorage.setItem('theme',theme);},{language,theme});
      const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));await page.goto(url);await page.waitForFunction(language=>document.documentElement.lang===language,language);await page.locator('[data-real-time-phase="ready"]').waitFor({timeout:120000});
      const matrix=page.getByLabel(language==='ko'?'행렬':'Matrix',{exact:true});await matrix.selectOption('hessian');
      assert((await page.locator('[data-real-time-selected]').innerText()).includes('"-1"'));
      await matrix.selectOption('formal-connected');await page.getByLabel(language==='ko'?'복소 성분':'Complex component',{exact:true}).selectOption('imaginary');assert.equal(JSON.parse(await page.locator('[data-real-time-selected]').innerText()).multiplier,'1/2');
      const selectExample=page.getByLabel(language==='ko'?'전체 예제':'All examples',{exact:true});await selectExample.selectOption('zero-leading-pivot-invertible');await page.locator('[data-real-time-phase="ready"]').waitFor({timeout:120000});
      await matrix.selectOption('inverse');await page.getByLabel(language==='ko'?'선택 열':'Selected column',{exact:true}).selectOption('1');assert((await page.locator('[data-real-time-selected]').innerText()).includes('"coefficientBits": 512'));
      await page.getByRole('button',{name:language==='ko'?'3D 행렬 보기':'Show 3D matrix',exact:true}).click();const canvas=page.locator('[data-real-time-matrix-plot="3d"] canvas');await canvas.waitFor();await canvas.scrollIntoViewIfNeeded();
      const before=await canvas.screenshot();await canvas.focus();await page.keyboard.press('ArrowRight');assert(!before.equals(await canvas.screenshot()));
      await canvas.evaluate(node=>{const gl=(node as HTMLCanvasElement).getContext('webgl2');gl?.getExtension('WEBGL_lose_context')?.loseContext();});await page.locator('[data-real-time-matrix-plot="2d"]').waitFor();
      await page.getByRole('button',{name:language==='ko'?'3D 다시 시도':'Retry 3D',exact:true}).click();await page.locator('[data-real-time-matrix-plot="3d"] canvas').waitFor();
      if(width===320){const target=page.locator('[data-real-time-matrix-plot="3d"] canvas');await target.scrollIntoViewIfNeeded();const b=await target.boundingBox();assert(b);const cdp=await context.newCDPSession(page);const first=await target.screenshot();await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:b.x+b.width/2,y:b.y+b.height/2}]});await cdp.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x:b.x+b.width/2+40,y:b.y+b.height/2}]});await cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});assert(!first.equals(await target.screenshot()));await cdp.detach();}
      await page.screenshot({path:`${root}/matrix-${language}-${theme}-${width}.png`});
      if(language==='en')assert(!/[가-힣]/u.test(await page.locator('main').innerText()));
      assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'document overflow');
      const axe=await new AxeBuilder({page}).include('main').withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();assert.deepEqual(axe.violations.map(v=>({id:v.id,nodes:v.nodes.map(n=>n.target)})),[]);
      if(language==='en'&&theme==='light'&&width===1440){
        await selectExample.selectOption('free-thirty-two');await page.locator('[data-real-time-phase="ready"]').waitFor({timeout:120000});await matrix.selectOption('inverse');assert.equal(await page.locator('[data-real-time-matrix="inverse"] svg rect').count(),961);
        await page.getByLabel('Selected row',{exact:true}).selectOption('30');await page.getByLabel('Selected column',{exact:true}).selectOption('30');await page.getByLabel('Coefficient display scale',{exact:true}).selectOption('signed-log');
        await page.getByRole('button',{name:'Show 3D matrix',exact:true}).click();await page.locator('[data-real-time-matrix-plot="3d"] canvas').waitFor();await page.getByText('Original wire intervals for every entry',{exact:true}).click();assert.equal(await page.locator('tbody tr').count(),961);await page.getByText('Original wire intervals for every entry',{exact:true}).click();
        await selectExample.selectOption('regulated-free-branch');await page.locator('[data-real-time-phase="ready"]').waitFor({timeout:120000});await matrix.selectOption('regulated-connected');await page.getByLabel('Complex component',{exact:true}).selectOption('imaginary');
        const entry=JSON.parse(await page.locator('[data-real-time-selected]').innerText());assert(entry.imaginary.lower!=='0'||entry.imaginary.upper!=='0');
        await selectExample.selectOption('fixed-width-free');await page.locator('[data-real-time-phase="ready"]').waitFor({timeout:120000});await matrix.selectOption('final-covariance');await page.getByLabel('Complex component',{exact:true}).selectOption('real');assert.equal(await page.locator('[data-real-time-matrix="final-covariance"] svg rect').count(),4);
      }
      rows.push({language,theme,width,nativeWorkers:true,negativeMode:true,zeroPivotInverse:true,threeD:true,keyboard:true,contextRecovery:true,touch:width===320,accessibility:true});writeFileSync(root+'/native-progress.json',JSON.stringify({rows,errors},null,2)+'\n');console.log(JSON.stringify({completed:rows.length,language,theme,width}));await context.close();
    }
    assert.equal(rows.length,8);assert.deepEqual(errors,[]);writeFileSync(root+'/native-matrix.json',JSON.stringify({rows,errors,maximumDimension:31,maximumCells:961,previewOnly:true,allInteractiveViewsComplete:false},null,2)+'\n');
  }finally{await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
