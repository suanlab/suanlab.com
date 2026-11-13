const assert = require('node:assert/strict');
const fs = require('node:fs');
const { execFileSync } = require('node:child_process');
const { chromium } = require('playwright');
const AxeBuilder = require('@axe-core/playwright').default;
const base = process.env.SUANLAB_TEST_BASE || 'http://127.0.0.1:4175';
const report = process.env.SUANLAB_QUANTUM_MANY_BODY_RESEARCH_REPORT || '.runtime/learning/quantum-many-body-research/browser.json';
const data = JSON.parse(execFileSync(process.cwd() + '/node_modules/.bin/tsx', ['-e', `
import {learningGuides,learningUnits} from './src/data/interactive-book';
import {curriculumDomains} from './src/data/learning-curriculum';
import {spinChainTeachingPresets} from './src/lib/learning-many-body-spin-teaching';
import {fermionTeachingPresets} from './src/lib/learning-many-body-fermion-teaching';
import {manyBodyInputKey} from './src/lib/learning-many-body-input';
import {spinChainModel} from './src/lib/learning-spin-chain';
import {fermionFockModel} from './src/lib/learning-fermion-fock';
const fullDomain=curriculumDomains.find(d=>d.id==='quantum-physics')!;
const allIds=["finite-spin-chain-local-hamiltonians", "fermion-fock-parity-operator-algebra"];
const requested=process.env.SUANLAB_QUANTUM_MANY_BODY_RESEARCH_UNIT;
if(requested&&!allIds.includes(requested))throw Error('Unknown quantum many-body research unit');
const ids=allIds.filter(id=>!requested||id===requested);
if(!ids.every(id=>fullDomain.unitIds.includes(id)))throw new Error('Quantum many-body units must remain in quantum physics');
const domain={...fullDomain,unitIds:ids};
const labs=Object.fromEntries(ids.map(id=>{const ps=id===allIds[0]?spinChainTeachingPresets:fermionTeachingPresets;return[id,{presets:ps.map(p=>{const r=p.input.kind==='spin-chain'?spinChainModel(p.input.parameters):fermionFockModel(p.input.parameters);return{...p,key:manyBodyInputKey(p.input),energy:r.energy.value,variance:r.variance.value};})}];}));
console.log(JSON.stringify({domain,labs:Object.fromEntries(domain.unitIds.map(id=>[id,labs[id]])),guides:Object.fromEntries(domain.unitIds.map(id=>[id,learningGuides[id]])),units:learningUnits.filter(u=>domain.unitIds.includes(u.id))}));
`], { encoding: 'utf8', maxBuffer: 10 * 1024 * 1024 }));

const selectors = id => {const prefix=id==='finite-spin-chain-local-hamiltonians'?'spin':'fermion';return {lab:`data-${prefix}-lab`,current:`data-${prefix}-input`,status:`data-${prefix}-state`,result:`data-${prefix}-result`,energy:`data-${prefix}-energy`,variance:`data-${prefix}-variance`,guide:`data-${prefix}-guide`,concept:'data-many-body-concept',step:'data-many-body-concept-step'};};

(async () => {
  fs.mkdirSync(require('node:path').dirname(report), { recursive: true });
  fs.mkdirSync('.runtime/learning', { recursive: true });
  assert.equal(data.domain.unitIds.length, process.env.SUANLAB_QUANTUM_MANY_BODY_RESEARCH_UNIT ? 1 : 2);
  assert.equal(data.domain.topics.length, 11);
  const rows = [], errors = [], guides = [], questions = [], concepts = [], interactions = [], assetBounds = [];
  const save = () => fs.writeFileSync(report, JSON.stringify({ unitIds: data.domain.unitIds, rows, errors, guides, questions, concepts, interactions, assetBounds }, null, 2));
  const browser = await chromium.launch({ executablePath: process.env.SUANLAB_BROWSER_PATH || '/usr/bin/google-chrome', args: ['--no-sandbox', '--enable-unsafe-swiftshader'] });
  try {
    const figurePage = await browser.newPage({ viewport: { width: 1200, height: 1400 } });
    for (const id of data.domain.unitIds) {
      const name = {"finite-spin-chain-local-hamiltonians": "finite-gap-response", "fermion-fock-parity-operator-algebra": "majorana-covariance-limits"}[id];
      for (const lang of ['en', 'ko']) {
        const path = `public/assets/learning/quantum-many-body-research-${name}.${lang}.svg`;
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
        const figure = {"finite-spin-chain-local-hamiltonians": "quantum-many-body-research-finite-gap-response", "fermion-fock-parity-operator-algebra": "quantum-many-body-research-majorana-covariance-limits"}[id];
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
          await page.screenshot({ path: `.runtime/learning/quantum-many-body-research-${id}-${lang}-320.png` });
        }
        if (lang === 'en') assert.ok(!/[가-힣]/.test(await page.locator('.ai-book').innerText()), id);
        await page.waitForFunction(() => document.documentElement.scrollWidth <= innerWidth);
        if (lang === 'en' && width === 320 && !dark) {
          const guide = data.guides[id];
          assert.equal(guide.steps.length, 3, id);
          assert.equal(guide.experiments.length, 3, id);
          for (let i = 0; i < guide.steps.length; i++) {
            const step = page.locator('.ai-concept-steps button').nth(i);
            await step.click();
            assert.equal(await step.getAttribute('aria-pressed'), 'true');
            await page.waitForFunction(({i,sel}) => document.querySelector(`[${sel.concept}]`)?.getAttribute(sel.step) === String(i), {i,sel});
            await page.locator(`[${sel.concept}]`).getByRole('button',{name:'Read explanation after predicting',exact:true}).click();
            assert.ok((await page.locator('[data-many-body-concept-explanation]').innerText()).length>20);
            concepts.push({ id, step: i });
          }
          const guidedPanel=page.locator(`[${sel.lab}] details`).filter({has:page.locator(`[${sel.guide}="0"]`)}).first();
          if(await guidedPanel.getAttribute('open')===null)await guidedPanel.locator(':scope > summary').click();
          assert.equal(await guidedPanel.locator(`[${sel.guide}]`).count(),3,id);
          for (let i = 0; i < guide.experiments.length; i++) {
            const selectedPreset = data.labs[id].presets[guide.experiments[i].preset];
            await guidedPanel.locator(`[${sel.guide}="${i}"]`).getByRole('button', { name: selectedPreset.label[lang], exact: true }).click();
            const expected = data.labs[id].presets[guide.experiments[i].preset].key;
            await page.waitForFunction(({expected,sel}) => (() => { const lab=document.querySelector(`[${sel.lab}]`);return lab?.getAttribute(sel.current)===expected && lab.getAttribute(sel.status)==='ready' && !!lab.querySelector(`[${sel.result}]`); })(), {expected,sel});
            const displayed=page.locator(`[${sel.lab}] [${sel.result}]`);assert.equal(Number(await displayed.getAttribute(sel.energy)),selectedPreset.energy);assert.equal(Number(await displayed.getAttribute(sel.variance)),selectedPreset.variance);
            const answer=guidedPanel.locator(`[${sel.guide}="${i}"] details`);if(await answer.getAttribute('open')===null)await answer.locator('summary').click();
            assert.ok(await guidedPanel.locator('p').evaluateAll(nodes => nodes.every(p => {
              const bounds = p.getBoundingClientRect();
              const walker = document.createTreeWalker(p, NodeFilter.SHOW_TEXT);
              let node;
              while ((node = walker.nextNode())) {
                const range = document.createRange(); range.selectNodeContents(node);
                if ([...range.getClientRects()].some(r => r.width > 0 && (r.left < bounds.left - 1 || r.right > bounds.right + 1))) return false;
              }
              return true;
            })), `${id}: expanded guide wraps at 320px`);
            guides.push({ id, example: i });
          }
          const saved = await page.locator(`[${sel.lab}]`).getAttribute(sel.current);
          await page.reload();
          await page.waitForFunction(sel => {const lab=document.querySelector(`[${sel.lab}]`);return lab?.getAttribute(sel.status)==='ready' && !!lab.querySelector(`[${sel.result}]`);},sel);
          await page.waitForFunction(({saved,sel}) => (() => { const lab=document.querySelector(`[${sel.lab}]`);return lab?.getAttribute(sel.current)===saved && lab.getAttribute(sel.status)==='ready' && !!lab.querySelector(`[${sel.result}]`); })(), {saved,sel});
          const unit = data.units.find(unit => unit.id === id);
          assert.equal(unit.questions.length, 3, id);
          for (let i = 0; i < unit.questions.length; i++) {
            const question = page.locator('#check .ai-question').nth(i);
            await question.getByRole('radio').nth(unit.questions[i].answer).check();
            await question.getByRole('button').click();
            await question.locator('.ai-answer-correct').waitFor();
            assert.ok((await question.locator('.ai-answer-correct').innerText()).includes('Correct.'), id);
            questions.push({ id, question: i });
          }
          await page.locator('.ai-completion input').check();
          await page.reload();
          await page.waitForFunction(sel => {const lab=document.querySelector(`[${sel.lab}]`);return lab?.getAttribute(sel.status)==='ready' && !!lab.querySelector(`[${sel.result}]`);},sel);
          assert.ok(await page.locator('.ai-completion input').isChecked());
          await page.locator(`[${sel.lab}]`).getByRole('button', { name: 'Reset to first example', exact: true }).click();
          await page.waitForFunction(sel => {const lab=document.querySelector(`[${sel.lab}]`);return lab?.getAttribute(sel.status)==='ready' && !!lab.querySelector(`[${sel.result}]`);},sel);
          const initial=data.labs[id].presets[0].key;
          assert.equal(await page.locator(`[${sel.lab}]`).getAttribute(sel.current),initial);
          assert.ok(await page.locator('.ai-completion input').isChecked());
          interactions.push({ id, concepts: 3, guides: 3, quizzes: 3, persistence: true, completion: true, reset: 'first example' });
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
    console.log(`Quantum many-body research checks passed: ${rows.length} reader states, ${guides.length} expanded examples, ${questions.length} explained questions, ${concepts.length} concept selections, ${interactions.length} persistence/completion groups; no page errors or accessibility violations.`);
  } finally {
    save();
    console.log('Closing verification browser.');
    await browser.close();
    console.log('Verification browser closed.');
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
