const assert = require('node:assert/strict');
const fs = require('node:fs');
const { chromium } = require('playwright');
const AxeBuilder = require('@axe-core/playwright').default;
const base = process.env.SUANLAB_TEST_BASE || 'http://127.0.0.1:4175';
const root = '/book/online/physics-math-for-ai/';
const units = ['vectors-and-forces', 'linear-transforms', 'gradients'];
const algebraUnits = ['inner-products', 'least-squares', 'eigenvalues'];
const calculusUnits = ['chain-rule', 'jacobians', 'backpropagation'];
const allUnits = [units[0], algebraUnits[0], units[1], algebraUnits[1], algebraUnits[2], units[2], ...calculusUnits];
(async () => {
  fs.mkdirSync('.runtime/learning', { recursive: true });
  const browser = await chromium.launch({ executablePath: process.env.SUANLAB_BROWSER_PATH || '/usr/bin/google-chrome', args: ['--no-sandbox', '--enable-unsafe-swiftshader', '--disable-gpu-rasterization'] });
  const errors = [], rows = [], scenarios = [];
  async function context(language = 'en', options = {}) {
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 1000 }, reducedMotion: 'reduce', ...options });
    await ctx.addInitScript(lang => { try { localStorage.setItem('suanlab-language', lang); } catch {} }, language);
    await ctx.route('**/*', route => new URL(route.request().url()).origin === new URL(base).origin ? route.continue() : route.abort());
    ctx.on('page', page => {
      page.on('pageerror', e => errors.push({ url: page.url(), message: e.message }));
      page.on('console', msg => { if (msg.type() === 'error' && /Hydration|did not match|Encountered two children/.test(msg.text())) errors.push({ url: page.url(), message: msg.text() }); });
    });
    return ctx;
  }
  async function go(page, route, lang = 'en') {
    const response = await page.goto(base + route); assert.equal(response.status(), 200);
    await page.waitForFunction(lang => document.documentElement.lang === lang, lang);
    if (allUnits.some(id => route.endsWith(id + '/'))) await page.locator('.ai-presets button').first().waitFor({ state: 'visible' });
    await page.evaluate(() => document.fonts.ready);
  }
  async function readout(page) { return page.locator('[data-testid="lab-readout"]').innerText(); }
  async function ready(page) { await page.waitForFunction(() => !document.querySelector('.ai-presets')?.disabled); }
  async function guidedLearning(page, kind) {
    const intuition = page.locator('.ai-intuition');
    const steps = intuition.locator('.ai-concept-steps button'); const count = await steps.count();
    assert.ok(await intuition.getByRole('button', { name: 'Previous step', exact: true }).isDisabled());
    for (let i = 0; i < count; i++) {
      await steps.nth(i).click(); assert.equal(await steps.nth(i).getAttribute('aria-pressed'), 'true');
      assert.ok((await intuition.getByRole('img').getAttribute('aria-label')).length > 20);
    }
    assert.ok(await intuition.getByRole('button', { name: 'Next step', exact: true }).isDisabled());
    if (kind === 'backpropagation') assert.ok((await intuition.getByRole('img').textContent()).includes('Lnew=0.28'));
    if (kind === 'jacobians') assert.ok((await intuition.getByRole('img').textContent()).includes('Jd=(-0.1, 0.2)'));
    await page.getByRole('button', { name: 'KO', exact: true }).first().click(); await page.waitForFunction(() => document.documentElement.lang === 'ko');
    assert.equal(await steps.nth(count - 1).getAttribute('aria-pressed'), 'true');
    await page.getByRole('button', { name: 'EN', exact: true }).first().click(); await page.waitForFunction(() => document.documentElement.lang === 'en');
    await intuition.evaluate(node => node.scrollIntoView({ block: 'center' })); await intuition.screenshot({ path: `.runtime/learning/${kind}-concept-en.png` });
    await page.locator('.ai-unit-outline > summary').click();
    const validAnchors = await page.locator('.ai-unit-outline a').evaluateAll(nodes => nodes.every(a => document.getElementById(decodeURIComponent(a.hash.slice(1)))));
    assert.ok(validAnchors); await page.locator('.ai-unit-outline > summary').click();
    const expected = { 'vectors-and-forces': ['(3, 4, 0)', '(2, 0, 0)', '25'], 'linear-transforms': ['(3, 2, 0)', '-1', 'None · dimensional collapse'], gradients: ['2.125', '(0.5, 0.125)', '9.125'], 'inner-products': ['(3, 0, 0)', '90°', '—'], 'least-squares': ['0.75', '0.179', '(1, 2, 3)'], eigenvalues: ['(3, 1)', '75%', '50%'], 'chain-rule': ['1.24', '0.04', '-0.36'], jacobians: ['(-0.12, 0.21)', '(-0.01, 0.02)', '(-0.01, 0.01)'], backpropagation: ['0.5', '-1', '0.5'] }[kind];
    const choices = page.locator('.ai-guided-choices button');
    for (let i = 0; i < expected.length; i++) {
      await choices.nth(i).click(); await page.getByRole('button', { name: 'Load this example', exact: true }).click();
      assert.ok((await readout(page)).includes(expected[i]), `${kind} guided example ${i + 1}`);
      const explanation = page.locator('.ai-guided-answer'); assert.equal(await explanation.getAttribute('open'), null);
      await explanation.locator('summary').click(); assert.ok(!/[가-힣]/.test(await explanation.innerText()));
    }
    await page.getByRole('button', { name: 'Reset all experiment settings', exact: true }).click();
    scenarios.push(`${kind}: concept steps, locale preservation, valid theory contents, and three reproducible guided examples.`);
  }

  try {
    for (const lang of ['ko', 'en']) {
      const ctx = await context(lang); const page = await ctx.newPage();
      for (const width of [320, 768, 1440]) {
        await page.setViewportSize({ width, height: 1000 });
        for (const route of [root, ...allUnits.map(id => root + id + '/')]) {
          await go(page, route, lang);
          assert.equal(await page.locator('main h1').count(), 1, route);
          assert.ok((await page.locator('main h1').innerText()).length > 5);
          const overflow = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth); assert.ok(overflow <= 1, `${route}: ${width}px overflow ${overflow}`);
          if (lang === 'en') assert.ok(!/[가-힣]/.test(await page.locator('.ai-book').innerText()), `${route}: Korean text in English`);
          if(width===320 && calculusUnits.some(id=>route.endsWith(id+'/'))){
            const graph=page.locator('.ai-calculus-diagram').first();
            assert.ok(await graph.evaluate(node=>node.scrollWidth>node.clientWidth));
            await graph.focus();await page.keyboard.press('ArrowRight');
            await page.waitForFunction(()=>document.querySelector('.ai-calculus-diagram').scrollLeft>0);
          }
          const audit = await new AxeBuilder({ page }).include('.ai-book').withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze();
          const violations = audit.violations.map(v => ({ id: v.id, nodes: v.nodes.map(n => n.target) }));
          rows.push({ lang, width, route, violations }); fs.writeFileSync('.runtime/learning/browser-check.json', JSON.stringify({ rows, errors, scenarios }, null, 2));
          assert.deepEqual(violations, [], `${lang} ${width} ${route}`);
        }
      }
      await ctx.close();
    }
    const ctx = await context(); const page = await ctx.newPage();
    await go(page, root); await page.screenshot({ path: '.runtime/learning/book-overview-en.png', fullPage: true });
    await go(page, root + units[0] + '/'); await ready(page); await guidedLearning(page, units[0]);
    await page.locator('.ai-viewport').scrollIntoViewIfNeeded(); await page.locator('.ai-viewport[data-renderer="3d"]').waitFor({ timeout: 20000 });
    assert.ok((await readout(page)).includes('(3, 4, 0)')); assert.ok((await readout(page)).includes('(1.5, 2, 0)'));
    await page.getByRole('slider', { name: 'Time:', exact: false }).fill('2');
    assert.ok((await readout(page)).includes('25')); assert.ok((await readout(page)).includes('(3, 4, 0)'));
    await page.locator('.ai-presets').getByRole('button', { name: 'Opposing · moving', exact: true }).click();
    await page.getByRole('slider', { name: 'Time:', exact: false }).fill('2');
    const moving = await readout(page); assert.ok(moving.includes('(2, 0, 0)')); assert.ok(moving.includes('(1, 0, 0)'));
    await page.reload(); await ready(page); assert.ok((await readout(page)).includes('(2, 0, 0)'));
    await page.getByRole('button', { name: 'KO', exact: true }).first().click();
    await page.waitForFunction(() => document.documentElement.lang === 'ko'); assert.ok((await readout(page)).includes('(2, 0, 0)'));
    await page.getByRole('button', { name: 'EN', exact: true }).first().click();
    await page.waitForFunction(() => document.documentElement.lang === 'en');
    await page.getByRole('spinbutton', { name: 'Mass (kg)', exact: true }).fill('4'); assert.equal(await page.getByRole('slider', { name: 'Time:', exact: false }).inputValue(), '0');
    await page.locator('.ai-presets').getByRole('button', { name: '3–4–5 resultant', exact: true }).click();
    await page.locator('.ai-viewport').scrollIntoViewIfNeeded(); await page.locator('.ai-viewport[data-renderer="3d"]').waitFor();
    await page.screenshot({ path: '.runtime/learning/vector-lab-en.png' });
    await page.getByRole('button', { name: 'Show 2D', exact: true }).click(); assert.equal(await page.locator('.ai-viewport').getAttribute('data-renderer'), '2d');
    await page.getByRole('button', { name: 'Show 3D', exact: true }).click(); await page.locator('.ai-viewport[data-renderer="3d"]').waitFor();
    await page.locator('.ai-canvas-host canvas').evaluate(canvas => canvas.getContext('webgl2').getExtension('WEBGL_lose_context').loseContext());
    await page.locator('.ai-viewport[data-renderer="2d"]').waitFor(); assert.ok(await page.getByText('3D is unavailable.', { exact: false }).isVisible());
    await page.getByRole('button', { name: 'Retry 3D', exact: true }).click(); await page.locator('.ai-viewport[data-renderer="3d"]').waitFor();
    assert.ok(await page.locator('.ai-playback').getByRole('button', { name: 'Play', exact: true }).isDisabled());
    await page.locator('.ai-playback').getByRole('button', { name: '+0.1 s', exact: true }).click(); assert.equal(await page.getByRole('slider', { name: 'Time:', exact: false }).inputValue(), '0.1');
    const question = page.locator('.ai-question').first(); await question.getByRole('radio', { name: '7 N', exact: true }).check(); await question.getByRole('button').click(); assert.ok((await question.innerText()).includes('Try again.'));
    await question.getByRole('radio', { name: '5 N', exact: true }).check(); await question.getByRole('button').click(); assert.ok((await question.innerText()).includes('Correct.'));
    await page.locator('.ai-completion input').check(); await page.reload(); await ready(page); assert.ok(await page.locator('.ai-completion input').isChecked()); assert.ok(await page.getByRole('radio', { name: '5 N', exact: true }).isChecked());
    scenarios.push('Vector math, moving cancellation, input resets, saved state, locale preservation, quiz, completion, WebGL loss/retry and reduced-motion manual stepping.');
    await go(page, root + units[1] + '/'); await ready(page); await guidedLearning(page, units[1]);
    await page.locator('.ai-presets').getByRole('button', { name: '90° rotation', exact: true }).click(); assert.ok((await readout(page)).includes('(0, 1, 0)'));
    await page.locator('.ai-presets').getByRole('button', { name: 'Reflection', exact: true }).click(); assert.ok((await readout(page)).includes('−1') || (await readout(page)).includes('-1'));
    await page.locator('.ai-presets').getByRole('button', { name: 'Collapse', exact: true }).click(); assert.ok((await readout(page)).includes('None · dimensional collapse'));
    await page.locator('.ai-viewport').scrollIntoViewIfNeeded(); await page.locator('.ai-viewport[data-renderer="3d"]').waitFor(); await page.screenshot({ path: '.runtime/learning/matrix-lab-en.png' });
    await go(page, root + units[2] + '/'); await ready(page); await guidedLearning(page, units[2]);
    await page.locator('.ai-playback').getByRole('button', { name: 'One step', exact: true }).click(); assert.ok((await readout(page)).includes('(1, 1.5)')); assert.ok((await readout(page)).includes('2.125'));
    await page.locator('.ai-viewport').scrollIntoViewIfNeeded(); await page.locator('.ai-viewport[data-renderer="3d"]').waitFor(); await page.screenshot({ path: '.runtime/learning/gradient-lab-en.png' });
    await page.locator('.ai-presets').getByRole('button', { name: 'Boundary · 1', exact: true }).click(); await page.locator('.ai-playback').getByRole('button', { name: 'One step', exact: true }).click(); assert.ok((await readout(page)).includes('(-2, 0)'));
    await page.locator('.ai-presets').getByRole('button', { name: 'Diverging · 1.25', exact: true }).click();
    for (let i = 0; i < 50; i++) { const button = page.locator('.ai-playback').getByRole('button', { name: 'One step', exact: true }); if (await button.isDisabled()) break; await button.click(); }
    assert.ok((await page.locator('.ai-lab').innerText()).includes('coordinate exceeded 1,000'));
    await page.setViewportSize({ width: 320, height: 850 }); await page.waitForFunction(() => document.documentElement.scrollWidth <= innerWidth);
    await page.getByRole('button', { name: 'KO', exact: true }).first().click(); await page.waitForFunction(() => document.documentElement.lang === 'ko');
    await page.locator('.ai-presets button').first().click(); await page.locator('.ai-viewport').scrollIntoViewIfNeeded(); await page.screenshot({ path: '.runtime/learning/gradient-mobile-ko.png' });
    scenarios.push('Matrix rotation/reflection/singular output and gradient updates, oscillation, divergence stop, and unclipped mobile values.');
    await page.setViewportSize({ width: 1440, height: 1000 });
    for (const kind of algebraUnits) {
      await go(page, root + kind + '/'); await ready(page); await guidedLearning(page, kind);
      await page.locator('.ai-viewport').scrollIntoViewIfNeeded(); await page.locator('.ai-viewport[data-renderer="3d"]').waitFor();
      if (kind === 'inner-products') {
        await page.locator('.ai-presets').getByRole('button', { name: 'Opposite direction', exact: true }).click(); assert.ok((await readout(page)).includes('(-3, 0, 0)'));
        await page.getByRole('spinbutton', { name: 'u · y', exact: true }).fill('1'); await page.reload(); await ready(page); assert.equal(await page.getByRole('spinbutton', { name: 'u · y', exact: true }).inputValue(), '1');
      } else if (kind === 'least-squares') {
        await page.getByRole('button', { name: 'Apply least-squares solution', exact: true }).click(); assert.ok((await readout(page)).includes('0.179'));
        assert.ok(await page.getByRole('spinbutton', { name: 'Slope a', exact: true }).evaluate(node => node.validity.valid));
        await page.getByRole('spinbutton', { name: 'y3', exact: true }).fill('5'); await page.getByRole('button', { name: 'Apply least-squares solution', exact: true }).click();
        assert.ok((await readout(page)).includes('0.714')); await page.reload(); await ready(page); assert.equal(await page.getByRole('spinbutton', { name: 'y3', exact: true }).inputValue(), '5');
      } else {
        await page.locator('.ai-presets').getByRole('button', { name: 'Covariance', exact: true }).click(); assert.ok((await readout(page)).includes('(2, 1, 2)'));
        await page.locator('.ai-presets').getByRole('button', { name: 'Negative eigenvalue', exact: true }).click(); assert.ok((await readout(page)).includes('(3, -1)'));
        await page.getByRole('button', { name: 'Covariance point model', exact: true }).click(); assert.ok((await page.locator('.ai-model-note').innerText()).includes('negative eigenvalue'));
        await page.getByRole('button', { name: 'Show 2D', exact: true }).click(); assert.equal(await page.locator('.ai-lab-projection circle').count(), 0);
        await page.locator('.ai-presets').getByRole('button', { name: 'Zero matrix', exact: true }).click(); assert.ok((await page.locator('.ai-model-note').innerText()).includes('Total variance is zero'));
        await page.locator('.ai-presets').getByRole('button', { name: 'Covariance', exact: true }).click(); await page.getByRole('button', { name: 'Show 3D', exact: true }).click();
      }
      await page.locator('.ai-viewport').scrollIntoViewIfNeeded(); await page.locator('.ai-viewport[data-renderer="3d"]').waitFor(); await page.screenshot({ path: `.runtime/learning/${kind}-lab-en.png` });
    }
    for (const kind of calculusUnits) {
      await go(page, root + kind + '/'); await ready(page); await guidedLearning(page, kind);
      await page.locator('.ai-calculus-stages button').nth(1).click();
      await page.reload(); await ready(page);
      assert.equal(await page.locator('.ai-calculus-stages button').nth(1).getAttribute('aria-pressed'), 'true');
      await page.locator('.ai-calculus-stages button').last().click();
      if (kind === 'chain-rule') {
        await page.getByRole('spinbutton', {name:'δ',exact:true}).fill('0.01');
        assert.ok((await readout(page)).includes('0.12'));
        await page.locator('.ai-presets').getByRole('button',{name:'Constant function',exact:true}).click();
        assert.ok((await readout(page)).includes('Linear prediction\n0'));
      } else if (kind === 'jacobians') {
        await page.locator('.ai-presets').getByRole('button',{name:'Large displacement',exact:true}).click();
        assert.ok((await readout(page)).includes('(0.25, 1.75)'));
        await page.getByRole('spinbutton',{name:'Output weights q1',exact:true}).fill('2');
        await page.reload();await ready(page);assert.equal(await page.getByRole('spinbutton',{name:'Output weights q1',exact:true}).inputValue(),'2');
      } else {
        await page.getByRole('button',{name:'Apply one update',exact:true}).click();
        assert.ok((await readout(page)).includes('0.28'));
        assert.ok((await page.getByRole('status').first().innerText()).includes('0.5 → new loss 0.28'));
        assert.ok(await page.getByRole('spinbutton',{name:'w1',exact:true}).evaluate(n=>n.validity.valid));
        await page.locator('.ai-presets').getByRole('button',{name:'Large learning rate',exact:true}).click();
        await page.getByRole('button',{name:'Apply one update',exact:true}).click();assert.ok((await readout(page)).includes('L\n2'));
        await page.locator('.ai-presets').getByRole('button',{name:'ReLU kink',exact:true}).click();
        assert.ok((await page.locator('.ai-model-note').first().innerText()).includes('no classical derivative'));
        await page.locator('.ai-presets').getByRole('button',{name:'tanh',exact:true}).click();
        assert.equal(await page.locator('.ai-calculus-table tbody tr').count(),5);
        assert.ok((await page.locator('.ai-calculus-table tbody').innerText()).includes('Yes'));
        await page.locator('.ai-presets').getByRole('button',{name:'Reference',exact:true}).click();
        for(const name of ['w1','w2','b','v','c'])await page.getByRole('spinbutton',{name,exact:true}).fill('10');
        await page.getByRole('spinbutton',{name:'Learning rate η',exact:true}).fill('2');
        await page.getByRole('button',{name:'Apply one update',exact:true}).click();
        assert.ok((await page.getByRole('status').first().innerText()).includes('Update not applied'));
        assert.equal(await page.getByRole('spinbutton',{name:'v',exact:true}).inputValue(),'10');
      }
      await page.getByRole('button',{name:'Reset all experiment settings',exact:true}).click();
      await page.locator('.ai-lab').evaluate(n=>n.scrollIntoView({block:'start'}));
      await page.screenshot({path:`.runtime/learning/${kind}-lab-en.png`});
    }
    scenarios.push('Calculus labs: saved stages and weights, finite changes, simultaneous updates, oversized steps, ReLU kink checks and unclipped rejected candidates.');
    scenarios.push('New algebra labs: saved inputs, least-squares optimum and exact fractions, valid covariance moments, indefinite/zero cases, 3D and 2D.');
    await ctx.close();
    const live = await context('en', { reducedMotion: 'no-preference' }); const animated = await live.newPage();
    await go(animated, root + units[2] + '/'); await ready(animated); await animated.locator('.ai-playback').scrollIntoViewIfNeeded();
    const play = animated.locator('.ai-playback').getByRole('button', { name: 'Play', exact: true }); await play.waitFor(); await animated.waitForFunction(() => !document.querySelector('.ai-playback button')?.disabled); await play.click();
    await animated.waitForFunction(() => document.querySelector('.ai-readout dd')?.textContent === '1');
    await animated.getByRole('button', { name: 'Pause site motion', exact: true }).first().click();
    await animated.waitForTimeout(850); const stopped = await readout(animated); await animated.waitForTimeout(850); assert.equal(await readout(animated), stopped);
    await animated.getByRole('button', { name: 'Play site motion', exact: true }).first().click(); await animated.locator('.ai-playback').scrollIntoViewIfNeeded();
    await animated.locator('.ai-playback').getByRole('button', { name: 'Play', exact: true }).click(); await animated.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await animated.waitForTimeout(850); const offscreen = await readout(animated); await animated.waitForTimeout(850); assert.equal(await readout(animated), offscreen);
    scenarios.push('Automatic playback respects global pause and stops offscreen.'); await live.close();
    const denied = await context('ko'); await denied.addInitScript(() => { Object.defineProperty(window, 'localStorage', { get() { throw new DOMException('Denied', 'SecurityError'); } }); });
    const blocked = await denied.newPage(); await go(blocked, root + units[0] + '/', 'ko'); await ready(blocked);
    await blocked.locator('.ai-playback').getByRole('button', { name: '+0.1 s', exact: true }).click(); assert.ok((await blocked.locator('.ai-lab').innerText()).includes('현재 화면에서만 유지'));
    await blocked.getByRole('button', { name: 'EN', exact: true }).first().click(); await blocked.waitForFunction(() => document.documentElement.lang === 'en'); assert.ok(!/[가-힣]/.test(await blocked.locator('.ai-book').innerText())); await denied.close();
    const malformed = await context('en');
    await malformed.addInitScript(() => {
      try {
        localStorage.setItem('suanlab-learning:vectors-and-forces:lab', '{invalid');
        localStorage.setItem('suanlab-learning:vectors-and-forces:study', 'null');
      } catch {}
    });
    const recovered = await malformed.newPage(); await go(recovered, root + units[0] + '/'); await ready(recovered);
    assert.ok((await readout(recovered)).includes('(3, 4, 0)')); assert.ok((await recovered.locator('.ai-lab').innerText()).includes('saved in this browser'));
    await malformed.close(); scenarios.push('Malformed saved state falls back to defaults without disabling browser persistence.');
    const unavailable = await context('en');
    await unavailable.addInitScript(() => {
      const original = HTMLCanvasElement.prototype.getContext;
      HTMLCanvasElement.prototype.getContext = function(type, ...args) { return /^webgl/.test(type) ? null : original.call(this, type, ...args); };
    });
    const fallback = await unavailable.newPage(); await go(fallback, root + units[0] + '/'); await ready(fallback);
    await fallback.locator('.ai-viewport').scrollIntoViewIfNeeded(); await fallback.getByText('3D is unavailable.', { exact: false }).waitFor();
    await fallback.locator('.ai-playback').getByRole('button', { name: '+0.1 s', exact: true }).click(); assert.ok((await readout(fallback)).includes('(0.15, 0.2, 0)')); await unavailable.close();
    const dark = await context('en', { colorScheme: 'dark' });
    await dark.addInitScript(() => { try { localStorage.setItem('theme', 'dark'); } catch {} }); const darkPage = await dark.newPage();
    for (const id of [null, ...allUnits]) {
      await go(darkPage, id ? root + id + '/' : root);
      const audit = await new AxeBuilder({ page: darkPage }).include('.ai-book').withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze();
      assert.deepEqual(audit.violations.map(v => v.id), []); rows.push({ lang: 'en', width: 1440, route: id ? root + id + '/' : root, theme: 'dark', violations: [] });
    }
    await go(darkPage, root + 'gradients/'); await ready(darkPage);
    await darkPage.locator('.ai-playback').getByRole('button', { name: 'One step', exact: true }).click();
    const beforeTheme = await readout(darkPage);
    await darkPage.locator('.ai-viewport').scrollIntoViewIfNeeded(); await darkPage.locator('.ai-viewport[data-renderer="3d"]').waitFor();
    await darkPage.locator('.ai-canvas-host canvas').evaluate(node => { window.__learningCanvas = node; });
    const darkColor = await darkPage.locator('.ai-orange').evaluate(node => getComputedStyle(node).color);
    await darkPage.screenshot({ path: '.runtime/learning/gradient-lab-dark.png' });
    await darkPage.getByRole('button', { name: 'Toggle theme', exact: true }).first().click();
    await darkPage.waitForFunction(() => !document.documentElement.classList.contains('dark'));
    assert.equal(await readout(darkPage), beforeTheme);
    assert.ok(await darkPage.locator('.ai-canvas-host canvas').evaluate(node => node === window.__learningCanvas));
    const lightColor = await darkPage.locator('.ai-orange').evaluate(node => getComputedStyle(node).color); assert.notEqual(lightColor, darkColor);
    for (const theme of ['light', 'dark']) {
      if (theme === 'dark') { await darkPage.getByRole('button', { name: 'Toggle theme', exact: true }).first().click(); await darkPage.waitForFunction(() => document.documentElement.classList.contains('dark')); }
      assert.ok(await darkPage.evaluate(() => getComputedStyle(document.querySelector('.ai-book')).backgroundColor === getComputedStyle(document.body).backgroundColor));
    }
    await darkPage.getByRole('button', { name: 'Show 2D', exact: true }).click();
    await darkPage.locator('.ai-viewport').scrollIntoViewIfNeeded(); await darkPage.screenshot({ path: '.runtime/learning/gradient-fallback-dark.png' });
    await dark.close(); scenarios.push('Unavailable WebGL keeps manual calculations; site colors follow the theme in 3D, 2D and dark mode without resetting the canvas or model.');
    const nojs = await browser.newContext({ javaScriptEnabled: false }); const staticPage = await nojs.newPage(); await staticPage.goto(base + root + units[0] + '/'); assert.ok((await staticPage.locator('.ai-theory').innerText()).includes('성분으로 더하기')); await nojs.close();
    scenarios.push('Storage-denied controls and language work; theory and fallback diagram are present without JavaScript.');
    fs.writeFileSync('.runtime/learning/browser-check.json', JSON.stringify({ rows, errors, scenarios }, null, 2));
    assert.deepEqual(errors, []); console.log(`Learning checks passed: ${rows.length} bilingual layout/accessibility states and ${scenarios.length} interaction groups.`);
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
