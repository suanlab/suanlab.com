import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { chromium } from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import { euclideanKernelSteps, euclideanKernelExperiments } from '../src/data/euclidean-kernel-teaching';
import {euclideanKernelDefault} from '../src/lib/learning-euclidean-kernel-input';
import {euclideanKernelInputKey} from '../src/lib/learning-euclidean-kernel-protocol';
const root='.runtime/learning/euclidean-kernel-teaching';
async function main(){
  const {url}=JSON.parse(readFileSync(root+'/review-server.json','utf8')),browser=await chromium.launch({headless:true}),rows=[];
  try{for(const language of ['ko','en']as const){
    const context=await browser.newContext({viewport:{width:1440,height:1000},reducedMotion:'reduce'});
    await context.addInitScript(language=>{localStorage.setItem('suanlab-language',language);localStorage.setItem('theme','light');},language);
    const page=await context.newPage(),errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));await page.goto(url);await page.locator('[data-euclidean-result]').waitFor();
    for(const [step,value]of euclideanKernelSteps.entries()){
      await page.getByRole('button',{name:`${step+1}. ${value.title[language]}`,exact:true}).click();
      assert.equal(await page.locator('.ai-concept-explanation >p').first().innerText(),value.body[language]);
      const svg=page.locator('svg[data-euclidean-reference]');assert.equal(await svg.getAttribute('data-euclidean-reference'),String(step));
      const outside=await svg.locator('text').evaluateAll(nodes=>nodes.flatMap(node=>{const b=(node as SVGTextElement).getBBox();return b.x< -1||b.y< -1||b.x+b.width>721||b.y+b.height>401?[{text:node.textContent,x:b.x,y:b.y,width:b.width,height:b.height}]:[];}));
      assert.deepEqual(outside,[],`${language}:step${step}`);
      await svg.screenshot({path:`${root}/reference-final-${language}-${step}.png`});rows.push({language,step,labelsInsideViewBox:true});
    }
    for(const [i,guide]of euclideanKernelExperiments.entries()){
      await page.getByLabel(language==='ko'?'안내 실험':'Guided experiment',{exact:true}).selectOption(String(i));
      await page.getByRole('button',{name:language==='ko'?'해설 보기':'Show explanation',exact:true}).click();
      assert.equal(await page.locator('[data-euclidean-guide-explanation]').innerText(),guide.explanation[language]);
    }
    const t=(ko:string,en:string)=>language==='ko'?ko:en;
    await page.getByLabel(t('전체 예제','All examples'),{exact:true}).selectOption('free-particle');await page.locator('[data-euclidean-free-trace]').waitFor();
    assert.equal(await page.getByText(t('유한 열적 위치 밀도 · 근사 표시','Finite thermal position density; approximate display'),{exact:true}).count(),0);
    await page.getByLabel(t('전체 예제','All examples'),{exact:true}).selectOption('one-variable-hand-check');
    await page.waitForFunction(key=>document.querySelector('[data-euclidean-result-key]')?.getAttribute('data-euclidean-result-key')===key,euclideanKernelInputKey(euclideanKernelDefault));
    const custom={...euclideanKernelDefault,time:'1/999999999989',mass:'16',hbar:'1/999999999979',omegaSquared:'1/999999999961',end:'16'};
    for(const [label,value]of [[t('총시간 τ','Total time τ'),custom.time],[t('질량 m','Mass m'),custom.mass],[t('작용 상수 ℏ','Action constant hbar'),custom.hbar],[t('진동수 제곱 ω²','Squared frequency ω²'),custom.omegaSquared],[t('마지막 값 b','End value b'),custom.end]])await page.getByLabel(label,{exact:true}).fill(value);
    await page.getByRole('button',{name:t('입력 적용','Apply input'),exact:true}).click();await page.waitForFunction(key=>document.querySelector('[data-euclidean-result-key]')?.getAttribute('data-euclidean-result-key')===key,euclideanKernelInputKey(custom));
    const comparisons=page.getByRole('region',{name:t('Mehler 로그 비교 · 소수 근사','Mehler log comparison; decimal approximation'),exact:true}).locator('tbody tr');assert.equal(await comparisons.count(),4);
    for(let i=0;i<4;i++)for(const j of [1,2])assert((await comparisons.nth(i).locator('td').nth(j).innerText()).includes(t('미해결','Unresolved')));
    if(language==='en')assert(!/[가-힣]/.test(await page.locator('main').innerText()));
    const axe=await new AxeBuilder({page}).include('main').withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();assert.deepEqual(axe.violations.map(v=>v.id),[]);assert.deepEqual(errors,[]);await context.close();
  }
  writeFileSync(root+'/reference-copy-verification.json',JSON.stringify({rows,guidesChecked:12,accessibilityCases:2,actualScreenshots:16,freeTraceDivergenceExplicit:true,unresolvedDifferencesNotLabeledZero:true,errors:[]},null,2)+'\n');
  }finally{await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
