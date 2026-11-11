const assert = require('node:assert/strict');
const fs = require('node:fs');
const {execFileSync} = require('node:child_process');
const {chromium} = require('playwright');
const AxeBuilder = require('@axe-core/playwright').default;
const base = process.env.SUANLAB_TEST_BASE || 'http://127.0.0.1:4175';
const root = '/book/online/physics-math-for-ai/';
const report = process.env.SUANLAB_PRIVACY_RESEARCH_REPORT || '.runtime/learning/privacy-research-browser.json';
const partition = process.env.SUANLAB_PRIVACY_RESEARCH_PART || '0/1';
assert.match(partition, /^\d+\/[1-9]\d*$/);
const [part, total] = partition.split('/').map(Number);
assert.ok(part < total);
// Numerical correctness is checked independently by learning-*.test.ts. These
// source-side snapshots check the rendered adapter and controls against that model.
const source = `import {privacyLabs,learningUnits,learningGuides} from './src/data/interactive-book';
import {evaluatePrivacy} from './src/lib/learning-privacy';
import {formatNumber} from './src/lib/learning-simulations';
const stats=(id,values)=>Object.fromEntries(Object.entries(evaluatePrivacy(id,values).stats).map(([key,value])=>[key,value===null?'—':Array.isArray(value)?(value.length?'('+value.map(formatNumber).join(', ')+')':'∅'):formatNumber(value)]));
console.log(JSON.stringify({labs:privacyLabs,units:learningUnits.filter(u=>u.id in privacyLabs),cases:Object.fromEntries(Object.entries(privacyLabs).map(([id,s])=>[id,{guides:learningGuides[id].experiments.map(g=>({values:s.presets[g.preset].values,stats:stats(id,s.presets[g.preset].values)})),boundaries:['min','max'].map(key=>{const values=s.controls.map(c=>c[key]);return {values,stats:stats(id,values)};})}]))}));`;
const data = JSON.parse(execFileSync(process.cwd() + '/node_modules/.bin/tsx', ['-e', source], {encoding:'utf8'}));
const allIds = Object.keys(data.labs), ids = allIds.filter((_, i) => i % total === part);
assert.equal(allIds.length, 16);
(async () => {
  fs.mkdirSync('.runtime/learning', {recursive:true});
  const browser = await chromium.launch({executablePath:process.env.SUANLAB_BROWSER_PATH || '/usr/bin/google-chrome', args:['--no-sandbox']});
  const rows = [], errors = [], scenarios = [];
  const save = () => fs.writeFileSync(report, JSON.stringify({partition, ids, rows, errors, scenarios}, null, 2));
  async function context(lang='en', dark=false) {
    const c = await browser.newContext({viewport:{width:1440,height:1000}, reducedMotion:'reduce', colorScheme:dark?'dark':'light'});
    c.setDefaultTimeout(15000);
    await c.addInitScript(({lang,dark,origin}) => {
      if (location.origin !== origin) return;
      localStorage.setItem('suanlab-language', lang);
      localStorage.setItem('theme', dark?'dark':'light');
    }, {lang,dark,origin:new URL(base).origin});
    await c.route('**/*', r => new URL(r.request().url()).origin === new URL(base).origin ? r.continue() : r.abort());
    c.on('page', p => p.on('pageerror', e => errors.push({url:p.url(),message:e.message})));
    return c;
  }
  async function ready(p) { await p.waitForFunction(() => document.querySelector('.ai-presets')?.disabled === false); }
  async function go(p, id, lang='en') {
    assert.equal((await p.goto(base + root + (id?id+'/':''))).status(), 200);
    await p.waitForFunction(l => document.documentElement.lang === l, lang);
    if (id) await ready(p);
    await p.evaluate(() => document.fonts.ready);
  }
  async function inputs(p, expected) {
    await p.waitForFunction(values => JSON.stringify([...document.querySelectorAll('.ai-parameters input')].map(n=>n.value)) === JSON.stringify(values.map(String)), expected);
  }
  async function readouts(p, expected) {
    const actual = await p.locator('[data-testid=lab-readout]').evaluate(node => Object.fromEntries([...node.querySelectorAll('dl>div')].map(n=>[n.querySelector('dt').textContent,n.querySelector('dd').textContent.trim()])));
    assert.deepEqual(actual, expected);
  }
  async function figures(p) {
    assert.ok(await p.locator('.ai-lab svg').count() > 0);
    assert.ok(await p.locator('.ai-lab svg text').evaluateAll(nodes => nodes.every(n => {
      const b=n.getBBox(), v=n.ownerSVGElement.viewBox.baseVal;
      return b.x>=0 && b.x+b.width<=v.width && b.y>=0 && b.y+b.height<=v.height;
    })), 'All native privacy labels fit their SVG');
    assert.ok(await p.locator('.ai-lab svg path').evaluateAll(nodes=>nodes.every(n=>!/(NaN|Infinity)/.test(n.getAttribute('d')||''))));
  }
  async function audit(p, id, lang, width, theme='light') {
    if (await p.locator('.ai-lab').count()) await figures(p);
    await p.waitForFunction(()=>document.documentElement.scrollWidth<=innerWidth, null, {timeout:3000});
    if (lang === 'en') assert.ok(!/[가-힣]/.test(await p.locator('.ai-book').innerText()));
    const a = await new AxeBuilder({page:p}).include('.ai-book').withTags(['wcag2a','wcag2aa','wcag21aa']).analyze();
    assert.deepEqual(a.violations.map(v=>({id:v.id,nodes:v.nodes.map(n=>n.target)})), [], `${id} ${lang} ${width} ${theme}`);
    rows.push({id,lang,width,theme,violations:0}); save();
  }
  async function scroll(p, target) {
    if (await target.count() && await target.evaluate(n=>n.scrollWidth>n.clientWidth)) {
      await target.focus(); await p.keyboard.press('ArrowRight');
      await p.waitForFunction(n=>n.scrollLeft>0, await target.elementHandle());
    }
  }
  try {
    for (const lang of ['ko','en']) {
      const c=await context(lang), p=await c.newPage();
      for (const width of [320,768,1440]) {
        await p.setViewportSize({width,height:1000});
        for (const id of [...(part===0?[null]:[]),...ids]) {
          await go(p,id,lang); await audit(p,id,lang,width);
          if (!id || width!==320) continue;
          await scroll(p,p.locator('.ai-foundation-scroll').first());
          const prose=p.locator('.ai-prose');
          assert.equal(await prose.locator('h2').filter({hasText:lang==='en'?'Research extension:':'연구 확장:'}).count(),1);
          assert.ok(await prose.locator('table').count()>0, `Missing research table: ${id} ${lang}`);
          assert.ok(await prose.locator('pre').count()>0, `Missing Python example: ${id} ${lang}`);
          for (const target of [...await prose.locator('table').all(),...await prose.locator('pre').all()]) await scroll(p,target);
          if (['finite-events-differential-privacy','stratified-evaluation-shift-bounds'].includes(id)) {
            const img=prose.locator('img[src*="privacy-"]');
            assert.equal(await img.count(),1); assert.ok((await img.getAttribute('alt')).length>30);
            await img.scrollIntoViewIfNeeded();
            await p.waitForFunction(()=>document.querySelector('.ai-prose img[src*="privacy-"]')?.naturalWidth===440);
          }
        }
      }
      await c.close();
    }
    const c=await context(), p=await c.newPage();
    for (const id of ids) {
      await go(p,id);
      const spec=data.labs[id], unit=data.units.find(u=>u.id===id);
      assert.equal(await p.locator('.ai-concept-steps button').count(),4);
      for (let i=0;i<4;i++) {
        const step=p.locator('.ai-concept-steps button').nth(i); await step.click();
        assert.equal(await step.getAttribute('aria-pressed'),'true');
      }
      await p.locator('.ai-unit-outline summary').click();
      assert.ok(await p.locator('.ai-unit-outline a').evaluateAll(nodes=>nodes.every(a=>document.getElementById(decodeURIComponent(a.hash.slice(1))))));
      await p.locator('.ai-unit-outline summary').click();
      await p.setViewportSize({width:320,height:1000});
      for (let i=0;i<3;i++) {
        await p.locator('.ai-guided-choices button').nth(i).click();
        await p.getByRole('button',{name:'Load this example',exact:true}).click();
        const example=data.cases[id].guides[i];
        await inputs(p,example.values); await readouts(p,example.stats); await figures(p);
        assert.ok(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
      }
      await p.setViewportSize({width:1440,height:1000});
      const saved=data.cases[id].guides[2].values;
      await p.reload(); await ready(p); await inputs(p,saved);
      for (const lang of ['KO','EN']) {
        await p.getByRole('button',{name:lang,exact:true}).first().click();
        await p.waitForFunction(l=>document.documentElement.lang===l,lang.toLowerCase()); await inputs(p,saved);
      }
      await p.getByRole('button',{name:'Reset all experiment settings',exact:true}).click();
      await inputs(p,spec.controls.map(c=>c.initial));
      for (const [i,control] of spec.controls.entries()) {
        assert.equal(control.step,1);
        const field=p.locator('.ai-parameters input').nth(i), before=await field.inputValue();
        const readout=await p.locator('[data-testid=lab-readout]').innerText(), draft=String(control.min+.5);
        await field.fill(draft); await field.blur();
        assert.equal(await field.inputValue(),draft); assert.equal(await field.getAttribute('aria-invalid'),'true');
        assert.equal(await p.locator('[data-testid=lab-readout]').innerText(),readout);
        await field.press('Escape'); assert.equal(await field.inputValue(),before);
      }
      for (const [j,boundary] of data.cases[id].boundaries.entries()) {
        for (const [i,v] of boundary.values.entries()) await p.locator('.ai-parameters input').nth(i).fill(String(v));
        await inputs(p,boundary.values); await readouts(p,boundary.stats);
        await p.setViewportSize({width:320,height:1000}); await audit(p,id+'-boundary-'+j,'en',320);
        await p.setViewportSize({width:1440,height:1000});
      }
      for (let i=0;i<2;i++) {
        const q=p.locator('.ai-question').nth(i); await q.getByRole('radio').nth(unit.questions[i].answer).check();
        await q.getByRole('button').click(); assert.ok((await q.innerText()).includes('Correct.'));
      }
      await p.locator('.ai-completion input').check(); await p.reload(); await ready(p);
      assert.ok(await p.locator('.ai-completion input').isChecked());
      await p.getByRole('button',{name:'Reset all experiment settings',exact:true}).click();
      await inputs(p,spec.controls.map(c=>c.initial));
      scenarios.push(id+': four steps, three guided adapters, stored settings, language, integer drafts, minimum/maximum vectors, quizzes and completion'); save();
    }
    await c.close();
    const dark=await context('en',true), dp=await dark.newPage();
    for (const id of [...(part===0?[null]:[]),...ids]) {
      await go(dp,id); await dp.waitForFunction(()=>document.documentElement.classList.contains('dark'));
      await audit(dp,id,'en',1440,'dark');
    }
    await dark.close();
    if (part===0) {
      const mc=await context(), mp=await mc.newPage(); await go(mp,null);
      await mp.locator('.ai-curriculum > summary').click(); await mp.locator('.ai-curriculum input').fill('Mathematics of Privacy');
      assert.equal(await mp.locator('.ai-curriculum-domain').count(),1);
      await mp.locator('.ai-curriculum-domain > summary').click();
      assert.equal(await mp.locator('.ai-curriculum-domain-body a').count(),allIds.length);
      await audit(mp,'privacy-domain-map','en',1440); await mc.close();
      scenarios.push('Privacy curriculum links and English search');
    }
    assert.deepEqual(errors,[]); save();
    console.log(`Privacy research checks passed (${partition}): ${rows.length} states, ${scenarios.length} groups and ${ids.length*3} guided examples.`);
  } finally { await browser.close(); }
})().catch(e=>{console.error(e);process.exitCode=1;});
