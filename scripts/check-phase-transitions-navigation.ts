/** Verify discovery and active classification through the exported homepage. */
import assert from 'node:assert/strict';
import {writeFileSync} from 'node:fs';
import {chromium} from 'playwright';
import {phaseKinds as kinds,phaseTitles} from '../src/lib/learning-phase-transitions';
const base=process.env.PHASE_READER_BASE_URL??'http://localhost:4175',book='/book/online/physics-math-for-ai/';
async function main(){const browser=await chromium.launch({headless:true,args:['--no-sandbox']}),report={menuCases:0,breadcrumbs:0,searches:0,cards:0,libraries:0,onlineExcluded:0,sitemap:false,errors:[] as string[]};
 try{for(const language of ['ko','en'] as const)for(const width of [320,1600]){
  const context=await browser.newContext({viewport:{width,height:1000},reducedMotion:'reduce'});await context.addInitScript(language=>localStorage.setItem('suanlab-language',language),language);const page=await context.newPage();page.on('pageerror',e=>report.errors.push(e.message));
  for(const kind of kinds){await page.goto(`${base}${book}${kind}/`);await page.waitForFunction(lang=>document.documentElement.lang===lang,language);await page.locator('main a[href="/book/interactive/"]').first().waitFor();report.breadcrumbs++;
   if(width===1600){const parent=page.locator('header nav a[href="/book/"]');await parent.focus();await parent.press('ArrowDown');assert.equal(await page.getByRole('menuitem',{name:'Interactive Book',exact:true}).getAttribute('aria-current'),'page');assert.equal(await page.getByRole('menuitem',{name:'Online Book',exact:true}).getAttribute('aria-current'),null);await page.keyboard.press('Escape');}
   else{await page.getByRole('button',{name:language==='ko'?'메뉴 열기':'Open menu',exact:true}).click();const dialog=page.getByRole('dialog');assert.equal(await dialog.getByRole('link',{name:'Interactive Book',exact:true}).getAttribute('aria-current'),'page');assert.equal(await dialog.getByRole('link',{name:'Online Book',exact:true}).getAttribute('aria-current'),null);await dialog.getByRole('button',{name:language==='ko'?'닫기':'Close',exact:true}).click();}report.menuCases++;
   await page.goto(`${base}/search/?q=${encodeURIComponent(phaseTitles[kind][language])}`);await page.locator(`main a[href="${book}${kind}/"]`).first().waitFor();report.searches++;
  }
  await page.goto(`${base}/book/interactive/`);await page.locator(`main a[href="${book}"]`).click();await page.waitForURL(`**${book}`);report.libraries++;for(const kind of kinds){await page.locator(`main a.ai-unit-card[href="${book}${kind}/"]`).waitFor();report.cards++;}
  await page.goto(`${base}/book/online/`);assert.equal(await page.locator('main a[href*="physics-math-for-ai"]').count(),0);assert.ok(await page.locator('main a[href^="/book/online/"]').count());report.onlineExcluded++;await context.close();
 }
 const xml=await (await fetch(`${base}/sitemap.xml`)).text();for(const kind of kinds)assert.ok(xml.includes(`${book}${kind}/`));report.sitemap=true;assert.deepEqual(report.errors,[]);writeFileSync('.runtime/learning/phase-transitions-integration/navigation-verification.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report));
 }finally{await browser.close();}}
main().catch(error=>{console.error(error);process.exitCode=1;});
