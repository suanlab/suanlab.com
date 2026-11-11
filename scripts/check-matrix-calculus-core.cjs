const assert = require('node:assert/strict');
const fs = require('node:fs');
const { execFileSync } = require('node:child_process');
const { chromium } = require(process.cwd() + '/node_modules/playwright');
const AxeBuilder = require(process.cwd() + '/node_modules/@axe-core/playwright').default;
const base = process.env.SUANLAB_TEST_BASE || 'http://127.0.0.1:4175';
const ids = ['gradients', 'chain-rule', 'jacobians', 'backpropagation'];
const metadata = JSON.parse(execFileSync(process.cwd() + '/node_modules/.bin/tsx', ['-e',
  `import {learningUnits} from './src/data/interactive-book'; console.log(JSON.stringify(learningUnits.filter(u=>${JSON.stringify(ids)}.includes(u.id)).map(u=>({id:u.id,questions:u.questions}))));`
], { encoding: 'utf8' }));
const expected = {
  gradients: ['2.125', '(0.5, 0.125)', '9.125'],
  'chain-rule': ['1.24', '0.04', '-0.36'],
  jacobians: ['(-0.12, 0.21)', '(-0.01, 0.02)', '(-0.01, 0.01)'],
  backpropagation: ['0.5', '-1', '0.5'],
};
(async () => {
  fs.mkdirSync('.runtime/learning', { recursive: true });
  const browser = await chromium.launch({ executablePath: process.env.SUANLAB_BROWSER_PATH || '/usr/bin/google-chrome', args: ['--no-sandbox', '--enable-unsafe-swiftshader'] });
  const rows = [], errors = [], interactions = [];
  try {
    for (const [language, width, theme] of [['en', 1440, 'light'], ['en', 320, 'light'], ['ko', 1440, 'dark'], ['ko', 320, 'dark']]) {
      const context = await browser.newContext({ viewport: { width, height: 1000 }, colorScheme: theme, reducedMotion: 'reduce' });
      context.setDefaultTimeout(15000);
      await context.addInitScript(([language, theme]) => {
        localStorage.setItem('suanlab-language', language);
        localStorage.setItem('theme', theme);
      }, [language, theme]);
      await context.route('**/*', route => new URL(route.request().url()).origin === new URL(base).origin ? route.continue() : route.abort());
      const page = await context.newPage();
      page.on('pageerror', error => errors.push({ url: page.url(), message: error.message }));
      for (const id of ids) {
        const response = await page.goto(`${base}/book/online/physics-math-for-ai/${id}/`);
        assert.equal(response.status(), 200);
        await page.waitForFunction(language => document.documentElement.lang === language, language);
        await page.waitForFunction(() => document.querySelector('.ai-presets')?.disabled === false);
        await page.waitForFunction(dark => document.documentElement.classList.contains('dark') === dark, theme === 'dark');
        assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
        if (language === 'en') assert(!/[가-힣]/.test(await page.locator('.ai-book').innerText()));
        await page.locator('.ai-unit-outline summary').click();
        assert(await page.locator('.ai-unit-outline a').evaluateAll(nodes => nodes.every(node => document.getElementById(decodeURIComponent(node.hash.slice(1))))));
        await page.locator('.ai-unit-outline summary').click();
        if (id === 'gradients' || id === 'jacobians') {
          const table = page.locator('.ai-prose table').filter({ hasText: id === 'gradients' ? '−2.828427' : '32/1025' });
          assert.equal(await table.count(), 1);
          assert.equal(await table.locator('tbody tr').count(), id === 'gradients' ? 2 : 3);
          assert.equal(await page.locator('.ai-prose .katex-error').count(), 0);
          assert.equal(await page.locator('.ai-prose pre').count(), 1);
          const code = page.locator('.ai-prose pre').first();
          if (await code.evaluate(node => node.scrollWidth > node.clientWidth)) {
            await code.evaluate(node => { node.scrollLeft = 0; });
            await code.focus();
            await page.keyboard.press('ArrowRight');
            await page.waitForFunction(node => node.scrollLeft > 0, await code.elementHandle());
          }
        }
        if (language === 'en' && width === 1440) {
          const steps = page.locator('.ai-concept-steps button');
          for (let i = 0; i < await steps.count(); i++) {
            await steps.nth(i).click();
            assert.equal(await steps.nth(i).getAttribute('aria-pressed'), 'true');
          }
          for (let i = 0; i < 3; i++) {
            await page.locator('.ai-guided-choices button').nth(i).click();
            await page.getByRole('button', { name: 'Load this example', exact: true }).click();
            await page.waitForFunction(expected => document.querySelector('[data-testid="lab-readout"]')?.textContent.includes(expected), expected[id][i]);
          }
          const inputs = page.locator('.ai-parameters input');
          const saved = await inputs.evaluateAll(nodes => nodes.map(node => node.value));
          const unit = metadata.find(unit => unit.id === id);
          for (let i = 0; i < unit.questions.length; i++) {
            const question = page.locator('.ai-question').nth(i);
            await question.getByRole('radio').nth(unit.questions[i].answer).check();
            await question.getByRole('button').click();
            assert((await question.innerText()).includes('Correct.'));
          }
          await page.locator('.ai-completion input').check();
          await page.reload();
          await page.waitForFunction(() => document.querySelector('.ai-presets')?.disabled === false);
          assert.deepEqual(await inputs.evaluateAll(nodes => nodes.map(node => node.value)), saved);
          assert(await page.locator('.ai-completion input').isChecked());
          interactions.push(id);
        }
        const result = await new AxeBuilder({ page }).include('.ai-book').withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze();
        assert.deepEqual(result.violations.map(v => ({ id: v.id, nodes: v.nodes.map(node => node.target) })), []);
        rows.push({ id, language, width, theme });
        fs.writeFileSync('.runtime/learning/matrix-calculus-core-browser.json', JSON.stringify({ rows, errors, interactions }, null, 2));
      }
      await context.close();
    }
    assert.deepEqual(errors, []);
    console.log(`${rows.length} reader states, ${interactions.length} interaction groups, ${interactions.length * 3} guided examples and ${interactions.reduce((sum, id) => sum + metadata.find(unit => unit.id === id).questions.length, 0)} explained quizzes passed.`);
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
