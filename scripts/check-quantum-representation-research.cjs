const assert = require('node:assert/strict');
const fs = require('node:fs');
const { execFileSync } = require('node:child_process');
const { chromium } = require('playwright');
const AxeBuilder = require('@axe-core/playwright').default;
const base = process.env.SUANLAB_TEST_BASE || 'http://127.0.0.1:4175';
const report = process.env.SUANLAB_QUANTUM_REPRESENTATION_RESEARCH_REPORT || '.runtime/learning/quantum-representation-research/browser.json';
const data = JSON.parse(execFileSync(process.cwd() + '/node_modules/.bin/tsx', ['-e', `
import {learningGuides,learningUnits} from './src/data/interactive-book';
import {curriculumDomains} from './src/data/learning-curriculum';
import {bosonTeachingPresets} from './src/lib/learning-many-body-boson-teaching';
import {mpsTeachingPresets} from './src/lib/learning-many-body-mps-teaching';
import {manyBodyInputKey} from './src/lib/learning-many-body-input';
import {bosonFockModel} from './src/lib/learning-boson-fock';
import {bosonCoherentModel} from './src/lib/learning-boson-coherent';
import {mpsModel} from './src/lib/learning-mps';
const fullDomain=curriculumDomains.find(d=>d.id==='quantum-physics')!;
const allIds=["boson-fock-cutoff-boundary-algebra", "matrix-product-state-contraction-gauge"];
const requested=process.env.SUANLAB_QUANTUM_REPRESENTATION_RESEARCH_UNIT;
if(requested&&!allIds.includes(requested))throw Error('Unknown quantum representation research unit');
const ids=allIds.filter(id=>!requested||id===requested);
if(!ids.every(id=>fullDomain.unitIds.includes(id)))throw new Error('Quantum representation units must remain in quantum physics');
const domain={...fullDomain,unitIds:ids};
const labs=Object.fromEntries(ids.map(id=>{const ps=id===allIds[0]?bosonTeachingPresets:mpsTeachingPresets;return[id,{presets:ps.map(p=>{let metrics;if(p.input.kind==='mps'){const r=mpsModel(p.input.parameters);metrics={'data-mps-norm':String(r.original.normSquared.value),'data-mps-defined':String(r.validity.physicalStateDefined)};}else{const r=p.input.kind==='boson-fock'?bosonFockModel(p.input.parameters):bosonCoherentModel(p.input.parameters).normalizedProjection;metrics={'data-boson-energy':String(r.energy.observation.value),'data-boson-variance':String(r.variance.observation.value)};}return{...p,key:manyBodyInputKey(p.input),metrics};})}];}));
console.log(JSON.stringify({domain,labs:Object.fromEntries(domain.unitIds.map(id=>[id,labs[id]])),guides:Object.fromEntries(domain.unitIds.map(id=>[id,learningGuides[id]])),units:learningUnits.filter(u=>domain.unitIds.includes(u.id))}));
`], { encoding: 'utf8', maxBuffer: 10 * 1024 * 1024 }));

const selectors=id=>{const prefix=id==='boson-fock-cutoff-boundary-algebra'?'boson':'mps';return{prefix,lab:`data-${prefix}-lab`,current:`data-${prefix}-input`,status:`data-${prefix}-state`,result:`data-${prefix}-result`,guide:`data-${prefix}-guide`,concept:'data-many-body-concept',step:'data-many-body-concept-step'};};

(async () => {
  fs.mkdirSync(require('node:path').dirname(report), { recursive: true });
  fs.mkdirSync('.runtime/learning', { recursive: true });
  assert.equal(data.domain.unitIds.length, process.env.SUANLAB_QUANTUM_REPRESENTATION_RESEARCH_UNIT ? 1 : 2);
  assert.equal(data.domain.topics.length, 11);
  const rows = [], errors = [], guides = [], questions = [], concepts = [], interactions = [], assetBounds = [];
  const save = () => fs.writeFileSync(report, JSON.stringify({ unitIds: data.domain.unitIds, rows, errors, guides, questions, concepts, interactions, assetBounds }, null, 2));
  const browser = await chromium.launch({ executablePath: process.env.SUANLAB_BROWSER_PATH || '/usr/bin/google-chrome', args: ['--no-sandbox', '--enable-unsafe-swiftshader'] });
  try {
    const figurePage = await browser.newPage({ viewport: { width: 1200, height: 1400 } });
    for (const id of data.domain.unitIds) {
      const name = {"boson-fock-cutoff-boundary-algebra": "boundary-history-dynamics", "matrix-product-state-contraction-gauge": "transfer-correlation-gauge"}[id];
      for (const lang of ['en', 'ko']) {
        const path = `public/assets/learning/quantum-representation-research-${name}.${lang}.svg`;
        const source = fs.readFileSync(path, 'utf8');
        await figurePage.setContent('<style>body{margin:0}</style>' + source.slice(source.indexOf('<svg')));
        await figurePage.evaluate(() => document.fonts.ready);
        const outside = await figurePage.locator('svg').first().evaluate(svg => {
          const box = svg.getBoundingClientRect();
          return [...svg.querySelectorAll('text')].filter(node => {
            const r = node.getBoundingClientRect();
            return r.left < box.left - 1 || r.right > box.right + 1 || r.top < box.top - 1 || r.bottom > box.bottom + 1;
          }).map(node => node.textContent);
        });
        assert.deepEqual(outside, [], path + ': SVG text must stay within its viewport');
        assetBounds.push({ path, outside });
      }
    }
    await figurePage.close();
    const combinations = [
      ...['ko', 'en'].flatMap(lang => [320, 768, 1440].map(width => ({ lang, width, dark: false }))),
      ...['ko', 'en'].map(lang => ({ lang, width: 1440, dark: true })),
    ];
    for (const { lang, width, dark } of combinations) {
      const context = await browser.newContext({ viewport: { width, height: 1000 }, reducedMotion: 'reduce', colorScheme: dark ? 'dark' : 'light' });
      await context.addInitScript(({ lang, dark, origin }) => {
        if (location.origin !== origin) return;
        localStorage.setItem('suanlab-language', lang);
        localStorage.setItem('theme', dark ? 'dark' : 'light');
      }, { lang, dark, origin: new URL(base).origin });
      await context.route('**/*', route => new URL(route.request().url()).origin === new URL(base).origin ? route.continue() : route.abort());
      const page = await context.newPage();
      page.on('pageerror', error => errors.push({ url: page.url(), message: error.message }));
      for (const id of data.domain.unitIds) {
        const sel = selectors(id);
        const url = `${base}/book/online/physics-math-for-ai/${id}/`;
        assert.equal((await page.goto(url)).status(), 200, id);
        await page.waitForFunction(lang => document.documentElement.lang === lang, lang);
        await page.waitForFunction(sel => {const lab=document.querySelector(`[${sel.lab}]`);return lab?.getAttribute(sel.status)==='ready' && !!lab.querySelector(`[${sel.result}]`);},sel);
        await page.evaluate(() => document.fonts.ready);
        await page.waitForFunction(dark => document.documentElement.classList.contains('dark') === dark, dark);
        assert.equal(await page.locator('.katex-error').count(), 0, id);
        assert.ok(await page.locator('.ai-prose h2').count() >= 6, id);
        // The language attribute can update before React replaces the manuscript.
        // Wait for the requested locale's figure, avoiding decode() on a changing src.
        const figure = {"boson-fock-cutoff-boundary-algebra": "quantum-representation-research-boundary-history-dynamics", "matrix-product-state-contraction-gauge": "quantum-representation-research-transfer-correlation-gauge"}[id];
        const expectedSource = `/assets/learning/${figure}.${lang}.svg`;
        await page.waitForFunction(expected => {
          const images = [...document.querySelectorAll('.ai-prose img')];
          return images.some(node => node.getAttribute('src') === expected && node.complete && node.naturalWidth > 0);
        }, expectedSource);
        for (const image of await page.locator('.ai-prose img').all()) {
          await image.evaluate(node => node.decode());
          assert.ok(await image.evaluate(node => node.naturalWidth > 0), id);
        }
        await page.locator('.ai-unit-outline summary').click();
        assert.ok(await page.locator('.ai-unit-outline a').evaluateAll(nodes => nodes.every(a => document.getElementById(decodeURIComponent(a.hash.slice(1))))), id);
        await page.locator('.ai-unit-outline summary').click();
        assert.ok(await page.locator('.ai-lab svg path').evaluateAll(nodes => nodes.every(node => !/(NaN|Infinity)/.test(node.getAttribute('d') || ''))), id);
        if (width === 320) {
          for (const element of await page.locator('.ai-prose pre, .ai-prose table, .ai-prose .katex-display, .ai-foundation-scroll').all()) {
            if (await element.evaluate(node => node.scrollWidth > node.clientWidth + 2)) {
              await element.focus();
              await page.keyboard.press('ArrowRight');
              await page.waitForFunction(node => node.scrollLeft > 0, await element.elementHandle());
            }
          }
          const firstImage = page.locator(`.ai-prose img[src="${expectedSource}"]`);
          if (await firstImage.count()) await firstImage.evaluate(node => scrollTo(0,node.getBoundingClientRect().top+scrollY-120));
          // Let keyboard-scroll and compositor updates finish before capturing nearby text.
          await page.waitForTimeout(800);
          await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
          await page.screenshot({ path: `.runtime/learning/quantum-representation-research-${id}-${lang}-320.png` });
        }
        if (lang === 'en') assert.ok(!/[가-힣]/.test(await page.locator('.ai-book').innerText()), id);
        await page.waitForFunction(() => document.documentElement.scrollWidth <= innerWidth);
        if(lang==='en'&&width===320&&!dark){
          const guide=data.guides[id];assert.equal(guide.steps.length,3);assert.equal(guide.experiments.length,3);
          const lab=page.locator(`[${sel.lab}]`);
          const ready=(key=null)=>page.waitForFunction(({sel,key})=>{const node=document.querySelector(`[${sel.lab}]`);return node?.getAttribute(sel.status)==='ready'&&(!key||node.getAttribute(sel.current)===key)&&!!node.querySelector(`[${sel.result}]`);},{sel,key});
          for(let i=0;i<3;i++){
            await page.locator('.ai-concept-steps button').nth(i).click();
            assert.equal(await page.locator(`[${sel.concept}]`).getAttribute(sel.step),String(i));
            await page.locator(`[${sel.concept}]`).getByRole('button',{name:'Read explanation after predicting',exact:true}).click();
            assert.ok((await page.locator('[data-many-body-concept-explanation]').innerText()).length>20);concepts.push({id,step:i});
          }
          const owner=sel.prefix==='boson'?page.locator('[data-boson-workbench]'):lab;
          const panel=owner.locator('details').filter({has:page.locator(`[${sel.guide}="0"]`)}).first();
          if(await panel.getAttribute('open')===null)await panel.locator(':scope > summary').click();
          let lastPreset;
          for(let i=0;i<3;i++){
            const preset=data.labs[id].presets[guide.experiments[i].preset];
            if(sel.prefix==='boson'){await page.locator(`[data-boson-kind="${preset.input.kind}"]`).click();await ready();}
            const article=panel.locator(`[${sel.guide}="${i}"]`);
            const controls=sel.prefix==='boson'?lab:article;
            await controls.getByRole('button',{name:preset.label[lang],exact:true}).click();
            await ready(preset.key);assert.equal(await lab.getAttribute(sel.current),preset.key);
            for(const [attr,expected]of Object.entries(preset.metrics))assert.equal(await lab.locator(`[${sel.result}]`).getAttribute(attr),expected,id+': '+attr);
            const answer=article.locator('details');if(await answer.getAttribute('open')===null)await answer.locator('summary').click();
            assert.ok(await panel.locator('p').evaluateAll(nodes=>nodes.every(p=>{const b=p.getBoundingClientRect();const w=document.createTreeWalker(p,NodeFilter.SHOW_TEXT);while(w.nextNode()){const r=document.createRange();r.selectNodeContents(w.currentNode);if([...r.getClientRects()].some(q=>q.width>0&&(q.left<b.left-1||q.right>b.right+1)))return false;}return true;})),id+': guided text wraps');
            lastPreset=preset;guides.push({id,example:i,model:preset.input.kind});
          }
          const saved=await lab.getAttribute(sel.current);
          await page.reload();
          if(sel.prefix==='boson')await page.locator(`[data-boson-kind="${lastPreset.input.kind}"]`).click();
          await ready(saved);assert.equal(await lab.getAttribute(sel.current),saved);
          if(sel.prefix==='boson'){
            await page.locator('[data-boson-kind="boson-fock"]').click();await ready();
            assert.equal(await lab.getAttribute(sel.current),data.labs[id].presets[guide.experiments[1].preset].key);
            await page.locator('[data-boson-kind="boson-coherent"]').click();await ready(saved);assert.equal(await lab.getAttribute(sel.current),saved);
          }
          const unit=data.units.find(u=>u.id===id);assert.equal(unit.questions.length,3);
          for(let i=0;i<3;i++){const question=page.locator('#check .ai-question').nth(i);await question.getByRole('radio').nth(unit.questions[i].answer).check();await question.getByRole('button').click();await question.locator('.ai-answer-correct').waitFor();assert.ok((await question.locator('.ai-answer-correct').innerText()).includes('Correct.'));questions.push({id,question:i});}
          await page.locator('.ai-completion input').check();await page.reload();
          if(sel.prefix==='boson')await page.locator(`[data-boson-kind="${lastPreset.input.kind}"]`).click();
          await ready();assert.ok(await page.locator('.ai-completion input').isChecked());
          await lab.getByRole('button',{name:'Reset to first example',exact:true}).click();await ready();
          const initial=sel.prefix==='boson'?data.labs[id].presets.find(p=>p.input.kind===lastPreset.input.kind):data.labs[id].presets[0];assert.equal(await lab.getAttribute(sel.current),initial.key);
          interactions.push({id,concepts:3,guides:3,quizzes:3,persistence:true,completion:true,independentBosonModels:sel.prefix==='boson',reset:'first example of selected model'});
        }
        const audit = await new AxeBuilder({ page }).include('.ai-book').withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze();
        assert.deepEqual(audit.violations.map(v => ({ id: v.id, nodes: v.nodes.map(node => node.target) })), [], `${id}: ${lang}/${width}/${dark}`);
        rows.push({ id, lang, width, theme: dark ? 'dark' : 'light', violations: 0 });
        save();
      }
      await context.close();
    }
    assert.equal(rows.length, 8*data.domain.unitIds.length);
    assert.equal(guides.length, 3*data.domain.unitIds.length);
    assert.equal(questions.length, 3*data.domain.unitIds.length);
    assert.equal(concepts.length, 3*data.domain.unitIds.length);
    assert.equal(interactions.length, data.domain.unitIds.length);
    assert.deepEqual(errors, []);
    save();
    console.log(`Quantum representation research checks passed: ${rows.length} reader states, ${guides.length} expanded examples, ${questions.length} explained questions, ${concepts.length} concept selections, ${interactions.length} persistence/completion groups; no page errors or accessibility violations.`);
  } finally {
    save();
    console.log('Closing verification browser.');
    await browser.close();
    console.log('Verification browser closed.');
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
