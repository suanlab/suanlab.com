import assert from 'node:assert/strict';
import {writeFileSync} from 'node:fs';
import {chromium} from 'playwright';
const root='.runtime/learning/duffing-secular-reader-integration',base=process.env.DUFFING_SECULAR_PREVIEW_BASE??'http://127.0.0.1:4186';
async function main(){const browser=await chromium.launch({headless:true}),states=[];try{
 for(const language of ['ko','en']as const){const context=await browser.newContext({viewport:{width:320,height:900},reducedMotion:'reduce'});await context.addInitScript(language=>localStorage.setItem('suanlab-language',language),language);const page=await context.newPage();await page.goto(base+'/book/online/physics-math-for-ai/duffing-secular-frequency-long-time/');await page.locator('[data-duffing-phase="ready"]').waitFor({timeout:60000});await page.evaluate(()=>document.fonts.ready);
  for(let step=0;step<8;step++){await page.locator('[data-duffing-secular-concept] .ai-concept-steps button').nth(step).click();const svg=page.locator('[data-duffing-secular-concept] svg');await svg.scrollIntoViewIfNeeded();const boxes=await svg.evaluate(el=>{const root=el as SVGSVGElement,view=root.viewBox.baseVal,scale=root.getBoundingClientRect().width/view.width;return{scale,texts:Array.from(root.querySelectorAll('text')).map(node=>{const b=node.getBBox();return{text:node.textContent,x:b.x,y:b.y,width:b.width,height:b.height,fontPixels:parseFloat(getComputedStyle(node).fontSize)*scale};})};});assert(boxes.scale>=.99);for(const box of boxes.texts){assert(box.x>=-1&&box.y>=-1&&box.x+box.width<=641&&box.y+box.height<=321,JSON.stringify({language,step,...box}));assert(box.fontPixels>=17.5);}
   await page.setViewportSize({width:900,height:900});await svg.screenshot({path:`${root}/reference-${language}-${step}.png`});await page.setViewportSize({width:320,height:900});states.push({language,step,width:320,captureWidth:900,texts:boxes.texts.length,minFontPixels:Math.min(...boxes.texts.map(b=>b.fontPixels)),withinViewBox:true});
  }await context.close();
 }
 writeFileSync(root+'/reference-diagrams.json',JSON.stringify({states},null,2)+'\n');console.log(JSON.stringify({referenceCases:16,textBoundsPassed:true}));
 }finally{await browser.close();}}
main().catch(error=>{console.error(error);process.exitCode=1;});
