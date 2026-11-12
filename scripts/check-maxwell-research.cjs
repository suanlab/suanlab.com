const assert = require('node:assert/strict');
const fs = require('node:fs');
const { execFileSync } = require('node:child_process');
const { chromium } = require('playwright');
const AxeBuilder = require('@axe-core/playwright').default;
const base = process.env.SUANLAB_TEST_BASE || 'http://127.0.0.1:4175';
const report = process.env.SUANLAB_MAXWELL_RESEARCH_REPORT || '.runtime/learning/maxwell-research/browser.json';
const data = JSON.parse(execFileSync(process.cwd() + '/node_modules/.bin/tsx', ['-e', `
import {electromagnetismLabs,learningGuides,learningUnits} from './src/data/interactive-book';
import {curriculumDomains} from './src/data/learning-curriculum';
const fullDomain=curriculumDomains.find(d=>d.id==='electromagnetism')!;
const ids=['maxwell-plane-wave-local-integral-laws','maxwell-charge-displacement-continuity'];
if(!ids.every(id=>fullDomain.unitIds.includes(id)))throw new Error('Maxwell research units must remain in electromagnetism');
const domain={...fullDomain,unitIds:ids};
const labs=electromagnetismLabs;
console.log(JSON.stringify({domain,labs:Object.fromEntries(domain.unitIds.map(id=>[id,labs[id]])),guides:Object.fromEntries(domain.unitIds.map(id=>[id,learningGuides[id]])),units:learningUnits.filter(u=>domain.unitIds.includes(u.id))}));
`], { encoding: 'utf8', maxBuffer: 10 * 1024 * 1024 }));

(async () => {
  fs.mkdirSync(require('node:path').dirname(report), { recursive: true });
  fs.mkdirSync('.runtime/learning', { recursive: true });
  assert.equal(data.domain.unitIds.length, 2);
  assert.equal(data.domain.topics.length, 10);
  const rows = [], errors = [], guides = [], questions = [], concepts = [], interactions = [];
  const save = () => fs.writeFileSync(report, JSON.stringify({ unitIds: data.domain.unitIds, rows, errors, guides, questions, concepts, interactions }, null, 2));
  const browser = await chromium.launch({ executablePath: process.env.SUANLAB_BROWSER_PATH || '/usr/bin/google-chrome', args: ['--no-sandbox'] });
  try {
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
        const url = `${base}/book/online/physics-math-for-ai/${id}/`;
        assert.equal((await page.goto(url)).status(), 200, id);
        await page.waitForFunction(lang => document.documentElement.lang === lang, lang);
        await page.waitForFunction(() => document.querySelector('.ai-presets')?.disabled === false);
        await page.evaluate(() => document.fonts.ready);
        await page.waitForFunction(dark => document.documentElement.classList.contains('dark') === dark, dark);
        assert.equal(await page.locator('.katex-error').count(), 0, id);
        assert.ok(await page.locator('.ai-prose h2').count() >= 6, id);
        assert.equal(await page.locator('.ai-parameters input').count(), data.labs[id].controls.length, id);
        // The language attribute can update before React replaces the manuscript.
        // Wait for the requested locale's figure, avoiding decode() on a changing src.
        const figure = id === 'maxwell-plane-wave-local-integral-laws' ? 'staggered-wave-stability' : 'periodic-gauss-repair';
        const expectedSource = `/assets/learning/electromagnetism-${figure}.${lang}.svg`;
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
        assert.ok(await page.locator('.ai-lab svg text').evaluateAll(nodes => nodes.every(node => {
          const b = node.getBBox(), v = node.ownerSVGElement.viewBox.baseVal;
          return b.x >= 0 && b.x + b.width <= v.width && b.y >= 0 && b.y + b.height <= v.height;
        })), id);
        if (width === 320) {
          for (const element of await page.locator('.ai-prose pre, .ai-prose table, .ai-prose .katex-display, .ai-foundation-scroll').all()) {
            if (await element.evaluate(node => node.scrollWidth > node.clientWidth + 2)) {
              await element.focus();
              await page.keyboard.press('ArrowRight');
              await page.waitForFunction(node => node.scrollLeft > 0, await element.elementHandle());
            }
          }
          const firstImage = page.locator('.ai-prose img').first();
          if (await firstImage.count()) await firstImage.scrollIntoViewIfNeeded();
          await page.screenshot({ path: `.runtime/learning/maxwell-research-${id}-${lang}-320.png` });
        }
        if (lang === 'en') assert.ok(!/[가-힣]/.test(await page.locator('.ai-book').innerText()), id);
        await page.waitForFunction(() => document.documentElement.scrollWidth <= innerWidth);
        if (lang === 'en' && width === 320 && !dark) {
          const guide = data.guides[id];
          assert.equal(guide.steps.length, 4, id);
          assert.equal(guide.experiments.length, 3, id);
          for (let i = 0; i < guide.steps.length; i++) {
            const step = page.locator('.ai-concept-steps button').nth(i);
            await step.click();
            assert.equal(await step.getAttribute('aria-pressed'), 'true');
            concepts.push({ id, step: i });
          }
          for (let i = 0; i < guide.experiments.length; i++) {
            await page.locator('.ai-guided-choices button').nth(i).click();
            await page.getByRole('button', { name: 'Load this example', exact: true }).click();
            const expected = data.labs[id].presets[guide.experiments[i].preset].values;
            await page.waitForFunction(expected => [...document.querySelectorAll('.ai-parameters input')].every((node, i) => Number(node.value) === expected[i]), expected);
            assert.deepEqual(await page.locator('.ai-parameters input').evaluateAll(nodes => nodes.map(node => Number(node.value))), expected);
            await page.locator('.ai-guided-answer summary').click();
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
          const saved = await page.locator('.ai-parameters input').evaluateAll(nodes => nodes.map(node => node.value));
          await page.reload();
          await page.waitForFunction(() => document.querySelector('.ai-presets')?.disabled === false);
          assert.deepEqual(await page.locator('.ai-parameters input').evaluateAll(nodes => nodes.map(node => node.value)), saved);
          const unit = data.units.find(unit => unit.id === id);
          assert.equal(unit.questions.length, 2, id);
          for (let i = 0; i < unit.questions.length; i++) {
            const question = page.locator('.ai-question').nth(i);
            await question.getByRole('radio').nth(unit.questions[i].answer).check();
            await question.getByRole('button').click();
            assert.ok((await question.innerText()).includes('Correct.'), id);
            questions.push({ id, question: i });
          }
          await page.locator('.ai-completion input').check();
          await page.reload();
          await page.waitForFunction(() => document.querySelector('.ai-presets')?.disabled === false);
          assert.ok(await page.locator('.ai-completion input').isChecked());
          await page.getByRole('button', { name: 'Reset all experiment settings', exact: true }).click();
          await page.waitForFunction(() => document.querySelector('.ai-presets')?.disabled === false);
          interactions.push({ id, concepts: 4, guides: 3, quizzes: 2, persistence: true, completion: true });
        }
        const audit = await new AxeBuilder({ page }).include('.ai-book').withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze();
        assert.deepEqual(audit.violations.map(v => ({ id: v.id, nodes: v.nodes.map(node => node.target) })), [], `${id}: ${lang}/${width}/${dark}`);
        rows.push({ id, lang, width, theme: dark ? 'dark' : 'light', violations: 0 });
        save();
      }
      await context.close();
    }
    assert.equal(rows.length, 16);
    assert.equal(guides.length, 6);
    assert.equal(questions.length, 4);
    assert.equal(concepts.length, 8);
    assert.equal(interactions.length, 2);
    assert.deepEqual(errors, []);
    save();
    console.log('Maxwell research checks passed: 16 reader states, 6 expanded guided examples, 4 explained questions, 8 concept selections, 2 persistence/completion groups; no page errors or accessibility violations.');
  } finally {
    save();
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
