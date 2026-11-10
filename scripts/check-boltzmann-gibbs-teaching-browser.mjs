/* Isolated real teaching components. Does not prove Next reader integration. */
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import AxeBuilder from '@axe-core/playwright';
const root = '.runtime/learning/boltzmann-gibbs-teaching';
const browser = await chromium.launch({ headless: true });
const result = { scope: 'isolated components with real worker and exported styles', combinations: 0, conceptSteps: 0, cases: 0, guides: 0, presets: 0, quizChoices: 0, axes: 0, errors: [], metrics: [] };
const ready = async page => {
  await page.locator('[data-boltzmann-concept-state="ready"]').waitFor();
  await page.locator('[data-boltzmann-state="ready"]').waitFor();
};
async function layout(page, english) {
  const issues = await page.evaluate(english => {
    const main = document.querySelector('main'), issues = [];
    if (document.documentElement.scrollWidth > innerWidth + 1) issues.push(`page overflow ${document.documentElement.scrollWidth}/${innerWidth}`);
    if (english && /[가-힣]/.test(main.innerText)) issues.push('Korean in English UI');
    if (/NaN|Infinity|undefined/.test(main.innerText)) issues.push('invalid numeric output');
    for (const svg of main.querySelectorAll('svg[role="img"]')) {
      const box = svg.getBoundingClientRect();
      if (!box.width || !box.height) issues.push('empty SVG');
      for (const node of svg.querySelectorAll('[points]')) if (/NaN|Infinity/.test(node.getAttribute('points'))) issues.push('invalid contour');
    }
    return issues;
  }, english);
  assert.deepEqual(issues, []);
}
try {
  for (const language of ['ko', 'en']) for (const theme of ['light', 'dark']) for (const width of [320, 1440]) {
    const context = await browser.newContext({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
    await context.addInitScript(language => localStorage.setItem('suanlab-language', language), language);
    const page = await context.newPage(); page.on('pageerror', e => result.errors.push(e.message));
    page.on('response', response => { if (response.status() >= 400) result.errors.push(`${response.status()} ${response.url()}`); });
    await page.goto('http://127.0.0.1:4187/'); await page.waitForFunction(() => !!window.boltzmannTeaching);
    await page.evaluate(theme => window.boltzmannTeaching.setTheme(theme), theme);
    await page.evaluate(() => document.fonts.ready);
    const kinds = await page.evaluate(() => window.boltzmannTeaching.kinds);
    for (const kind of kinds) {
      await page.evaluate(kind => { window.boltzmannTeaching.setKind(kind); window.boltzmannTeaching.setStep(0); }, kind);
      await page.locator(`[data-boltzmann-lab="${kind}"]`).waitFor(); await ready(page);
      const lab = page.locator('[data-boltzmann-lab]');
      for (let step = 0; step < 4; step++) {
        await page.evaluate(step => window.boltzmannTeaching.setStep(step), step);
        await page.locator(`[data-boltzmann-step="${step}"][data-boltzmann-concept-state="ready"]`).waitFor();
        const concept = page.locator('[data-boltzmann-concept]');
        const choices = concept.locator('[role="group"] > button');
        for (let i = 0; i < await choices.count(); i++) { await choices.nth(i).click(); assert.equal(await choices.nth(i).getAttribute('aria-pressed'), 'true'); await layout(page, language === 'en'); result.cases++; }
        result.conceptSteps++;
      }
      const guideChoices = lab.locator('.ai-guided-choices button');
      for (let i = 0; i < await guideChoices.count(); i++) {
        await guideChoices.nth(i).click(); assert.equal(await guideChoices.nth(i).getAttribute('aria-pressed'), 'true');
        const reveal = lab.locator('details details'); await reveal.locator('summary').click(); assert.equal(await reveal.getAttribute('open'), ''); result.guides++;
      }
      const presets = lab.locator('.ai-presets button');
      for (let i = 0; i < await presets.count(); i++) {
        await presets.nth(i).click(); await page.waitForFunction(i => document.querySelector('[data-boltzmann-lab]').dataset.boltzmannFrame === String(i), i); await ready(page);
        await layout(page, language === 'en'); result.presets++;
      }
      const quizzes = page.locator('.ai-question');
      for (let i = 0; i < await quizzes.count(); i++) {
        const quiz = quizzes.nth(i), radios = quiz.locator('input[type="radio"]');
        for (let j = 0; j < await radios.count(); j++) { await radios.nth(j).check(); await quiz.locator('button').click(); await quiz.locator('.ai-answer').waitFor(); await layout(page, language === 'en'); result.quizChoices++; }
      }
      await layout(page, language === 'en');
      const audit = await new AxeBuilder({ page }).analyze();
      if (audit.violations.length) await fs.writeFile(`${root}/axe-${language}-${theme}-${width}-${kinds.indexOf(kind)}.json`, JSON.stringify(audit.violations, null, 2));
      assert.deepEqual(audit.violations.map(v => ({ id: v.id, targets: v.nodes.map(n => n.target) })), []);
      result.axes++; result.combinations++;
      await page.screenshot({ path: `${root}/${language}-${theme}-${width}-${kinds.indexOf(kind)}.png`, fullPage: true });
      console.log(`Verified ${language}/${theme}/${width}/${kind}`);
    }
    await page.evaluate(() => window.boltzmannTeaching.setMounted(false));
    await page.waitForFunction(() => window.boltzmannTeaching.metrics.live === 0);
    const metrics = await page.evaluate(() => window.boltzmannTeaching.metrics); assert.equal(metrics.started, metrics.terminated); result.metrics.push(metrics);
    await context.close();
  }
  assert.deepEqual(result.errors, []);
  await fs.writeFile(`${root}/browser-result.json`, JSON.stringify(result, null, 2) + '\n');
  console.log(JSON.stringify(result));
} finally { await browser.close(); }
