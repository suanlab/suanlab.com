import assert from 'node:assert/strict';
import {writeFileSync} from 'node:fs';
import {chromium} from 'playwright';
import {fieldModesReader} from '../src/data/field-modes-reader';
const informationTeaching=[fieldModesReader];
const root='.runtime/learning/field-modes-reader-integration',base=process.env.FIELD_MODES_PREVIEW_BASE??'http://127.0.0.1:4179';
async function main(){const browser=await chromium.launch({headless:true}),states=[];try{
 for(const language of ['ko','en']as const){
  const context=await browser.newContext({viewport:{width:1920,height:1080}});await context.addInitScript(lang=>localStorage.setItem('suanlab-language',lang),language);
  const page=await context.newPage();await page.goto(base+'/book/online/');await page.waitForFunction(lang=>document.documentElement.lang===lang,language);
  assert.equal(await page.locator('a.ai-book-entry').count(),0);
  assert.equal(await page.locator('a[href="/book/online/physics-math-for-ai/"]').count(),0,'Online Book must not contain the interactive book');
  const menu=page.locator('[data-nav-group]').filter({has:page.locator('a[href="/book"],a[href="/book/"]')});
  await menu.locator('[data-menu-trigger]').focus();await menu.locator('[data-menu-trigger]').press('ArrowDown');await page.getByRole('menuitem',{name:'Interactive Book',exact:true}).click();
  await page.locator('.ai-book-entry').waitFor();assert(new URL(page.url()).pathname==='/book/interactive/');await page.locator('.ai-book-entry').click();await page.locator('.ai-book-overview').waitFor();
  assert.equal(await page.locator('.ai-unit-card').count(),587);
  await page.locator('.ai-curriculum > summary').click();await page.getByLabel(language==='ko'?'영역 또는 주제 찾기':'Find a domain or topic',{exact:true}).fill(language==='ko'?'장이론':'fields');
  const domain=page.locator('#curriculum-field-multiscale');await domain.locator(':scope > summary').click();
  for(const item of informationTeaching){const href='/book/online/physics-math-for-ai/'+item.unit.id+'/';assert.equal(await domain.locator('a[href="'+href+'"]').count(),1);assert.equal(await page.locator('.ai-unit-card[href="'+href+'"]').count(),1);}
  await domain.getByRole('link',{name:informationTeaching[0].unit.title[language],exact:true}).click();await page.locator('[data-modes-phase="ready"]').waitFor({timeout:60000});
  await menu.locator('[data-menu-trigger]').focus();await menu.locator('[data-menu-trigger]').press('ArrowDown');assert.equal(await page.getByRole('menuitem',{name:'Interactive Book',exact:true}).getAttribute('aria-current'),'page');assert.equal(await page.getByRole('menuitem',{name:'Online Book',exact:true}).getAttribute('aria-current'),null);
  await page.goto(base+'/search/');await page.getByRole('searchbox').fill(language==='ko'?'격자 분산':'Fourier Modes');const href='/book/online/physics-math-for-ai/'+informationTeaching[0].unit.id+'/';const hit=page.locator('a[href="'+href+'"]').filter({has:page.locator('h2')});await hit.waitFor();assert.equal(await hit.locator('h2').innerText(),informationTeaching[0].unit.title[language]);await hit.click();await page.locator('[data-modes-phase="ready"]').waitFor({timeout:60000});
  await page.locator('.ai-prerequisite-links a').first().click();await page.locator('[data-scalar-phase="ready"]').waitFor({timeout:60000});assert.equal(await page.locator('[data-scalar-metric=H]').innerText(),'7/2');await page.locator('.ai-reader-bottom a').last().click();await page.locator('[data-balance-phase="ready"]').waitFor({timeout:60000});assert.equal(await page.locator('[data-balance-metric=H]').innerText(),'7/2');assert.equal(await page.locator('.ai-reader-bottom a').last().getAttribute('href'),href);await page.locator('.ai-reader-bottom a').last().click();await page.locator('[data-modes-phase="ready"]').waitFor({timeout:60000});
  states.push({language,bookSubmenu:true,separateFromOnlineBook:true,cards:587,newDomainLinks:1,localizedSearch:true,prerequisiteRoundTrip:true,priorUnitEnergySmokeChecks:2,interactiveMenuActiveOnReader:true});await context.close();
 }
 writeFileSync(root+'/discovery.json',JSON.stringify({states},null,2)+'\n');console.log(JSON.stringify({discoveryLocales:states.length}));
 }finally{await browser.close();}}
main().catch(error=>{console.error(error);process.exitCode=1;});
