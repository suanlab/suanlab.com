const assert = require('node:assert/strict');
const fs = require('node:fs');
const { execFileSync } = require('node:child_process');
const { chromium } = require('playwright');
const AxeBuilder = require('@axe-core/playwright').default;
const base = process.env.SUANLAB_TEST_BASE || 'http://127.0.0.1:4175';
const report = process.env.SUANLAB_RESPONSE_RESEARCH_REPORT || '.runtime/learning/response-research/browser.json';
const data = JSON.parse(execFileSync(process.cwd() + '/node_modules/.bin/tsx', ['-e', `
import {learningGuides,learningUnits} from './src/data/interactive-book';
import {curriculumDomains} from './src/data/learning-curriculum';
import {fdtPresets,fdtPresetInput,fdtInputKey} from './src/lib/learning-fluctuation-dissipation';
import {lrPresets,lrPresetInput,lrInputKey} from './src/lib/learning-linear-response';
const fullDomain=curriculumDomains.find(d=>d.id==='nonequilibrium')!;
const allIds=["thermal-ou-fluctuation-response", "markov-perturbation-linear-response"];
const requested=process.env.SUANLAB_RESPONSE_RESEARCH_UNIT;
if(requested&&!allIds.includes(requested))throw Error('Unknown response research unit');
const ids=allIds.filter(id=>!requested||id===requested);
if(!ids.every(id=>fullDomain.unitIds.includes(id)))throw new Error('Response units must remain in nonequilibrium physics');
const domain={...fullDomain,unitIds:ids};
const c='thermal-ou-fluctuation-response',r='markov-perturbation-linear-response';
const labs={
 [c]:{presets:fdtPresets[c].map((p,i)=>({...p,key:fdtInputKey(fdtPresetInput(c,i))}))},
 [r]:{presets:lrPresets[r].map((p,i)=>({...p,key:lrInputKey(lrPresetInput(r,i))}))},
};
console.log(JSON.stringify({domain,labs:Object.fromEntries(domain.unitIds.map(id=>[id,labs[id]])),guides:Object.fromEntries(domain.unitIds.map(id=>[id,learningGuides[id]])),units:learningUnits.filter(u=>domain.unitIds.includes(u.id))}));
`], { encoding: 'utf8', maxBuffer: 10 * 1024 * 1024 }));

const selectors = id => id === 'thermal-ou-fluctuation-response'
 ? {lab:'data-fdt-lab',current:'data-fdt-current',concept:'data-fluctuation-dissipation-concept',step:'data-fluctuation-dissipation-step',state:'data-fluctuation-dissipation-concept-state'}
 : {lab:'data-lr-lab',current:'data-lr-current',concept:'data-linear-response-concept',step:'data-linear-response-step',state:'data-linear-response-concept-state'};

(async () => {
  fs.mkdirSync(require('node:path').dirname(report), { recursive: true });
  fs.mkdirSync('.runtime/learning', { recursive: true });
  assert.equal(data.domain.unitIds.length, process.env.SUANLAB_RESPONSE_RESEARCH_UNIT ? 1 : 2);
  assert.equal(data.domain.topics.length, 10);
  const rows = [], errors = [], guides = [], questions = [], concepts = [], interactions = [], assetBounds = [];
  const save = () => fs.writeFileSync(report, JSON.stringify({ unitIds: data.domain.unitIds, rows, errors, guides, questions, concepts, interactions, assetBounds }, null, 2));
  const browser = await chromium.launch({ executablePath: process.env.SUANLAB_BROWSER_PATH || '/usr/bin/google-chrome', args: ['--no-sandbox'] });
  try {
    const figurePage = await browser.newPage({ viewport: { width: 1200, height: 1400 } });
    for (const id of data.domain.unitIds) {
      const name = {"thermal-ou-fluctuation-response": "thermal-response-spectrum", "markov-perturbation-linear-response": "control-path-information"}[id];
      for (const lang of ['en', 'ko']) {
        const path = `public/assets/learning/response-research-${name}.${lang}.svg`;
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
        await page.waitForFunction(() => document.querySelector('.ai-presets')?.disabled === false);
        await page.evaluate(() => document.fonts.ready);
        await page.waitForFunction(dark => document.documentElement.classList.contains('dark') === dark, dark);
        assert.equal(await page.locator('.katex-error').count(), 0, id);
        assert.ok(await page.locator('.ai-prose h2').count() >= 6, id);
        // The language attribute can update before React replaces the manuscript.
        // Wait for the requested locale's figure, avoiding decode() on a changing src.
        const figure = {"thermal-ou-fluctuation-response": "response-research-thermal-response-spectrum", "markov-perturbation-linear-response": "response-research-control-path-information"}[id];
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
          await page.screenshot({ path: `.runtime/learning/response-research-${id}-${lang}-320.png` });
        }
        if (lang === 'en') assert.ok(!/[가-힣]/.test(await page.locator('.ai-book').innerText()), id);
        await page.waitForFunction(() => document.documentElement.scrollWidth <= innerWidth);
        if (lang === 'en' && width === 320 && !dark) {
          const guide = data.guides[id];
          assert.equal(guide.steps.length, 6, id);
          assert.equal(guide.experiments.length, 3, id);
          for (let i = 0; i < guide.steps.length; i++) {
            const step = page.locator('.ai-concept-steps button').nth(i);
            await step.click();
            assert.equal(await step.getAttribute('aria-pressed'), 'true');
            await page.waitForFunction(({i,sel}) => document.querySelector(`[${sel.concept}]`)?.getAttribute(sel.step) === String(i) && document.querySelector(`[${sel.concept}]`)?.getAttribute(sel.state) === 'ready', {i,sel});
            concepts.push({ id, step: i });
          }
          for (let i = 0; i < guide.experiments.length; i++) {
            await page.locator('.ai-guided-choices button').nth(i).click();
            const selectedPreset = data.labs[id].presets[guide.experiments[i].preset];
            await page.locator('.ai-guided-lab').getByRole('button', { name: selectedPreset.label[lang], exact: true }).click();
            const expected = data.labs[id].presets[guide.experiments[i].preset].key;
            await page.waitForFunction(({expected,sel}) => document.querySelector(`[${sel.lab}]`)?.getAttribute(sel.current) === expected, {expected,sel});
            const answer = page.locator('.ai-guided-lab details');
            if (await answer.getAttribute('open') === null) await answer.locator('summary').click();
            assert.ok(await page.locator('.ai-guided-lab p').evaluateAll(nodes => nodes.every(p => {
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
          await page.waitForFunction(() => document.querySelector('.ai-presets')?.disabled === false);
          await page.waitForFunction(({saved,sel}) => document.querySelector(`[${sel.lab}]`)?.getAttribute(sel.current) === saved, {saved,sel});
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
          await page.waitForFunction(() => document.querySelector('.ai-presets')?.disabled === false);
          assert.ok(await page.locator('.ai-completion input').isChecked());
          await page.getByRole('button', { name: 'Reset all experiment settings', exact: true }).click();
          await page.waitForFunction(() => document.querySelector('.ai-presets')?.disabled === false);
          interactions.push({ id, concepts: 6, guides: 3, quizzes: 3, persistence: true, completion: true });
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
    assert.equal(concepts.length, 6*data.domain.unitIds.length);
    assert.equal(interactions.length, data.domain.unitIds.length);
    assert.deepEqual(errors, []);
    save();
    console.log(`Response research checks passed: ${rows.length} reader states, ${guides.length} expanded examples, ${questions.length} explained questions, ${concepts.length} concept selections, ${interactions.length} persistence/completion groups; no page errors or accessibility violations.`);
  } finally {
    save();
    console.log('Closing verification browser.');
    await browser.close();
    console.log('Verification browser closed.');
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
