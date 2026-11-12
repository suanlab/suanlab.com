const assert = require('node:assert/strict');
const fs = require('node:fs');
const { chromium } = require('playwright');
const AxeBuilder = require('@axe-core/playwright').default;
const base = process.env.SUANLAB_TEST_BASE || 'http://127.0.0.1:43299';
const route = '/book/online/physics-math-for-ai/mechanics-energy/';
const report = process.env.SUANLAB_MECHANICS_ENERGY_REPORT || '.runtime/learning/classical-energy-browser.json';
(async () => {
  fs.mkdirSync('.runtime/learning', { recursive: true });
  const browser = await chromium.launch({ executablePath: process.env.SUANLAB_BROWSER_PATH || '/usr/bin/google-chrome', args: ['--no-sandbox'] });
  const rows = [], errors = [];
  async function open(lang, width, dark = false) {
    const context = await browser.newContext({ viewport: { width, height: 1000 }, reducedMotion: 'reduce', colorScheme: dark ? 'dark' : 'light' });
    await context.addInitScript(({ lang, dark }) => {
      localStorage.setItem('suanlab-language', lang);
      localStorage.setItem('theme', dark ? 'dark' : 'light');
    }, { lang, dark });
    await context.route('**/*', r => new URL(r.request().url()).origin === new URL(base).origin ? r.continue() : r.abort());
    const page = await context.newPage();
    page.on('pageerror', e => errors.push(e.message));
    assert.equal((await page.goto(base + route)).status(), 200);
    await page.waitForFunction(lang => document.documentElement.lang === lang && document.querySelector('.ai-presets') && !document.querySelector('.ai-presets').disabled, lang);
    return { context, page };
  }
  async function audit(page, lang, width, dark) {
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    const content = await page.locator('.ai-book').innerText();
    if (lang === 'en') assert.ok(!/[가-힣]/.test(content));
    assert.ok(content.includes('160802'));
    assert.equal(await page.locator('.katex-error').count(), 0);
    const heading = page.getByRole('heading', { name: lang === 'en' ? 'Research extension: a conserved quadratic form can fail to control motion' : '연구 확장: 보존량이 일정해도 운동을 제어하지 못할 수 있다', exact: true });
    assert.equal(await heading.count(), 1);
    const image = page.locator(`img[src$="classical-energy-invariant.${lang}.svg"]`);
    assert.equal(await image.count(), 1);
    await image.scrollIntoViewIfNeeded();
    await page.waitForFunction(lang => {
      const img = document.querySelector(`img[src$="classical-energy-invariant.${lang}.svg"]`);
      return img && img.complete && img.naturalWidth > 400;
    }, lang);
    assert.ok((await image.getAttribute('alt')).length > 30);
    if (width === 320 && !dark) {
      await image.screenshot({ path: `.runtime/learning/classical-energy-mobile-${lang}.png` });
      for (const region of [page.locator('.ai-book pre').last(), page.locator('.ai-book table[tabindex]').last()]) {
        assert.equal(await region.getAttribute('tabindex'), '0');
        if (await region.evaluate(node => node.scrollWidth > node.clientWidth)) {
          await region.evaluate(node => { node.scrollLeft = 0; });
          await region.focus(); await page.keyboard.press('ArrowRight');
          await page.waitForFunction(node => node.scrollLeft > 0, await region.elementHandle());
        }
      }
    }
    const violations = (await new AxeBuilder({ page }).include('.ai-book').withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze()).violations;
    assert.deepEqual(violations.map(v => v.id), []);
    rows.push({ lang, width, dark, violations: 0 });
  }
  try {
    for (const lang of ['ko', 'en']) for (const width of [320, 768, 1440]) {
      const { context, page } = await open(lang, width);
      await audit(page, lang, width, false); await context.close();
    }
    const dark = await open('en', 1440, true);
    await dark.page.waitForFunction(() => document.documentElement.classList.contains('dark'));
    await audit(dark.page, 'en', 1440, true); await dark.context.close();
    const { context, page } = await open('en', 1440);
    const presets = page.locator('.ai-presets button');
    for (let i = 0; i < 3; i++) {
      await presets.nth(i).click();
      const readout = await page.locator('[data-testid="lab-readout"]').innerText();
      assert.ok(readout.includes(i === 2 ? '160,802' : 'E₀ [J]\n0.5'));
    }
    const choices = page.locator('.ai-guided-choices button');
    assert.equal(await choices.count(), 3);
    for (let i = 0; i < 3; i++) {
      await choices.nth(i).click();
      await page.getByRole('button', { name: 'Load this example', exact: true }).click();
      const readout = await page.locator('[data-testid="lab-readout"]').innerText();
      assert.ok(readout.includes(i === 2 ? '160,802' : 'E₀ [J]\n0.5'));
    }
    await page.getByRole('button', { name: 'Reset all experiment settings', exact: true }).click();
    const step = page.getByRole('spinbutton', { name: 'Time step Δt [s]', exact: true });
    await step.fill('0.125'); await step.press('Tab');
    assert.equal(await step.inputValue(), '0.125');
    assert.equal(await step.getAttribute('aria-invalid'), 'false');
    const readout = await page.locator('[data-testid="lab-readout"]').innerText();
    await step.fill('0'); await step.press('Tab');
    assert.equal(await step.getAttribute('aria-invalid'), 'true');
    assert.equal(await page.locator('[data-testid="lab-readout"]').innerText(), readout);
    await step.focus(); await step.press('Escape'); assert.equal(await step.inputValue(), '0.125');
    await page.reload(); await page.waitForFunction(() => document.querySelector('.ai-presets') && !document.querySelector('.ai-presets').disabled);
    assert.equal(await step.inputValue(), '0.125');
    await page.getByRole('button', { name: 'KO', exact: true }).first().click();
    await page.waitForFunction(() => document.documentElement.lang === 'ko');
    assert.equal(await page.locator('.ai-parameters input').first().inputValue(), '0.125');
    await page.getByRole('button', { name: 'EN', exact: true }).first().click();
    await page.waitForFunction(() => document.documentElement.lang === 'en');
    for (const [i, answer] of [0, 1].entries()) {
      const question = page.locator('.ai-question').nth(i);
      await question.getByRole('radio').nth(answer).check(); await question.getByRole('button').click();
      assert.ok((await question.innerText()).includes('Correct.'));
    }
    for (const region of [page.locator('.ai-book pre').last(), page.locator('.ai-book table[tabindex]').last()]) {
      assert.equal(await region.getAttribute('tabindex'), '0');
      await region.focus();
      assert.ok(await region.evaluate(node => document.activeElement === node));
    }
    await page.locator('.ai-completion input').check(); await page.reload();
    await page.waitForFunction(() => document.querySelector('.ai-presets') && !document.querySelector('.ai-presets').disabled);
    assert.ok(await page.locator('.ai-completion input').isChecked());
    await context.close(); assert.deepEqual(errors, []);
    fs.writeFileSync(report, JSON.stringify({ rows, errors, interactionGroups: 1, guidedExamples: 3, scope: 'mechanics-energy only; other20 classical units not covered' }, null, 2) + '\n');
    console.log('mechanics-energy: seven locale/responsive/dark states, three guided examples, fractional controls, persistence, quizzes and completion passed.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
