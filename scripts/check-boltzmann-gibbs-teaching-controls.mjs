import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import { chromium } from 'playwright';
const root = '.runtime/learning/boltzmann-gibbs-teaching', base = 'http://127.0.0.1:4187/';
const browser = await chromium.launch({ headless: true });
const result = { scope: 'isolated actual teaching controls', checks: [], errors: [] };
try {
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, reducedMotion: 'no-preference' });
  await context.addInitScript(() => localStorage.setItem('suanlab-language', 'en'));
  const page = await context.newPage(); page.on('pageerror', e => result.errors.push(e.message));
  page.on('response', response => { if (response.status() >= 400) result.errors.push(`${response.status()} ${response.url()}`); });
  const ready = () => page.locator('[data-boltzmann-state="ready"]').waitFor();
  const lab = page.locator('[data-boltzmann-lab]');
  const setKind = async index => { await page.evaluate(index => window.boltzmannTeaching.setKind(window.boltzmannTeaching.kinds[index]), index); await lab.filter({ has: page.locator('input') }).waitFor(); await page.waitForFunction(index => document.querySelector('[data-boltzmann-lab]')?.dataset.boltzmannLab === window.boltzmannTeaching.kinds[index], index); await ready(); };
  const field = async (name, value) => { const input = lab.locator('label').filter({ has: page.getByText(name, { exact: true }) }).locator('input'); await input.fill(value); await input.press('Enter'); return input; };
  await page.goto(base); await ready();
  await page.addInitScript(() => {
    window.invalidFrames = 0;
    new MutationObserver(() => {
      const lab = document.querySelector('[data-boltzmann-lab]'), concept = document.querySelector('[data-boltzmann-concept]');
      if (lab && lab.dataset.boltzmannState !== 'ready' && lab.querySelector('[data-boltzmann-result]')) window.invalidFrames++;
      if (concept && concept.dataset.boltzmannConceptState !== 'ready' && concept.querySelector('[data-boltzmann-concept-result]')) window.invalidFrames++;
    }).observe(document, { subtree: true, childList: true, attributes: true });
  });
  await page.reload(); await ready();
  let input = await field('Temperature T (K)', '1e-400'); assert.equal(await input.getAttribute('aria-invalid'), 'true');
  assert.equal(JSON.parse(await lab.getAttribute('data-boltzmann-current')).parameters.temperature, 1);
  await field('Temperature T (K)', '2'); await ready();
  await page.reload(); await ready(); assert.equal(await lab.getByLabel('Temperature T (K)', { exact: true }).inputValue(), '2');
  await lab.getByLabel('Relative energy list (J) · 1–32 levels', { exact: true }).fill('0,1e-400');
  await lab.getByRole('button', { name: 'Apply energy and reference lists', exact: true }).click(); await lab.getByRole('alert').waitFor();
  assert.equal(JSON.parse(await lab.getAttribute('data-boltzmann-current')).parameters.energyLevels.length, 3);
  await lab.getByRole('button', { name: 'Reset all experiment settings', exact: true }).click(); await ready();
  console.log('Control group passed'); result.checks.push('strict scalar/list underflow rejection and valid saved-setting reload');

  // Exercise the visible pending/cancel/retry branch before its debounce expires.
  await lab.locator('.ai-presets button').nth(1).click();
  await lab.getByRole('button', { name: 'Stop calculation', exact: true }).click();
  assert.equal(await lab.locator('[data-boltzmann-result]').count(), 0);
  await lab.getByRole('button', { name: 'Retry calculation', exact: true }).click(); await ready();
  await page.evaluate(() => { window.boltzmannTeaching.metrics.blocked = true; });
  await lab.locator('.ai-presets button').nth(2).click(); await page.locator('[data-boltzmann-state="error"]').waitFor();
  assert.equal(await lab.locator('[data-boltzmann-result]').count(), 0);
  await page.evaluate(() => { window.boltzmannTeaching.metrics.blocked = false; });
  await lab.getByRole('button', { name: 'Retry calculation', exact: true }).click(); await ready();
  console.log('Control group passed'); result.checks.push('debounced cancellation, worker creation error, retry and stale-result hiding');

  await setKind(1);
  input = await field('Temperature T (K)', '0'); assert.equal(await input.getAttribute('aria-invalid'), 'true');
  await field('Temperature T (K)', '1');
  await lab.getByLabel('Allowed support', { exact: true }).selectOption('fixed-zero-sum'); await ready();
  assert.equal(await lab.getByLabel('Initial state index', { exact: true }).locator('option').count(), 2);
  await lab.getByLabel('Initial state index', { exact: true }).selectOption('1'); await ready();
  await lab.getByLabel('Allowed support', { exact: true }).selectOption('full-product'); await ready();
  assert.equal(await lab.getByLabel('Initial state index', { exact: true }).inputValue(), '0');
  await lab.getByLabel('Observation steps · increasing in 1–64', { exact: true }).fill('1,2,4,8');
  await lab.getByRole('button', { name: 'Apply observation steps', exact: true }).click(); await ready();
  const kernel = lab.getByLabel('Theoretical kernel to inspect', { exact: true });
  for (const name of ['first', 'second', 'random', 'systematic', 'blocked']) { await kernel.selectOption(name); assert.equal(await kernel.inputValue(), name); }
  await lab.getByLabel('Theoretical distribution observation step', { exact: true }).selectOption('3');
  console.log('Control group passed'); result.checks.push('Gibbs support changes, valid initial-state reset, observation steps and five kernels');

  await lab.getByLabel('Generate a finite-bit example path', { exact: true }).check(); await ready();
  const path = lab.locator('[data-boltzmann-path]'); await path.waitFor();
  await field('Path seed · 0–4294967295', '4294967295'); await ready();
  await field('Random-grid bits · 1–32', '32'); await ready();
  await field('Path steps · 1–2048', '2048'); await ready();
  for (const mode of ['first', 'second', 'systematic', 'blocked', 'random']) { await lab.getByLabel('Path update rule', { exact: true }).selectOption(mode); await ready(); await path.waitFor(); }
  await path.getByRole('button', { name: 'Step path', exact: true }).click(); assert.equal(await path.getAttribute('data-path-frame'), '1');
  await path.getByRole('button', { name: 'Reset path cursor', exact: true }).click(); assert.equal(await path.getAttribute('data-path-frame'), '0');
  await path.getByRole('button', { name: 'Play path', exact: true }).click();
  await page.waitForFunction(() => Number(document.querySelector('#experiment [data-path-frame]').dataset.pathFrame) >= 2);
  await page.getByRole('button', { name: 'Pause site motion', exact: true }).evaluate(node => node.click());
  const pausedFrame = await path.getAttribute('data-path-frame'); await page.waitForTimeout(400);
  assert.equal(await path.getAttribute('data-path-frame'), pausedFrame);
  await path.getByRole('button', { name: 'Step path', exact: true }).click();
  assert.equal(Number(await path.getAttribute('data-path-frame')), Number(pausedFrame) + 1);
  await page.emulateMedia({ reducedMotion: 'reduce' });
  assert.equal(await path.getByRole('button', { name: 'Play path', exact: true }).isDisabled(), true);
  await page.emulateMedia({ reducedMotion: 'no-preference' });
  await page.getByRole('button', { name: 'Play site motion', exact: true }).click();
  console.log('Control group passed'); result.checks.push('maximum finite-bit path, all update rules, manual cursor, site pause and reduced motion');

  // The automatic scan is an experimental comparison; it stops off screen.
  await lab.getByRole('button', { name: 'Scan comparison examples', exact: true }).scrollIntoViewIfNeeded();
  await lab.getByRole('button', { name: 'Scan comparison examples', exact: true }).click();
  await page.locator('[data-boltzmann-playing="true"]').waitFor();
  await page.evaluate(() => { const spacer = document.createElement('div'); spacer.style.height = '2000px'; spacer.id = 'offscreen-spacer'; document.body.append(spacer); scrollTo(0, document.body.scrollHeight); });
  await page.locator('[data-boltzmann-playing="false"]').waitFor();
  await lab.getByRole('button', { name: 'Step comparison example', exact: true }).click(); await ready();
  console.log('Control group passed'); result.checks.push('example scan stops off screen and manual comparison remains available');

  await setKind(2);
  input = await field('s1', '0'); assert.equal(await input.getAttribute('aria-invalid'), 'true');
  await field('s1', '-2'); await ready();
  await lab.getByRole('button', { name: 'Transformed coordinates z', exact: true }).click();
  assert.equal(await lab.getByRole('button', { name: 'Transformed coordinates z', exact: true }).getAttribute('aria-pressed'), 'true');
  await field('A₁₁ (J)', '0'); await ready(); await lab.locator('[data-boltzmann-normalization="absent"]').waitFor();
  assert.equal(await lab.locator('polygon').count(), 0);
  await lab.getByRole('button', { name: 'Reset all experiment settings', exact: true }).click(); await ready();
  console.log('Control group passed'); result.checks.push('Gaussian noninvertible map rejection, orientation change and no cloud for improper precision');

  const before = await page.evaluate(() => window.boltzmannTeaching.metrics.started);
  await page.evaluate(() => { window.boltzmannTeaching.setLanguage('ko'); window.boltzmannTeaching.setTheme('dark'); });
  await page.waitForFunction(() => document.documentElement.lang === 'ko');
  assert.equal(await page.evaluate(() => window.boltzmannTeaching.metrics.started), before);
  assert.equal(await page.evaluate(() => window.invalidFrames), 0);
  await page.evaluate(() => window.boltzmannTeaching.setMounted(false)); await page.waitForFunction(() => window.boltzmannTeaching.metrics.live === 0);
  result.metrics = await page.evaluate(() => window.boltzmannTeaching.metrics); assert.equal(result.metrics.started, result.metrics.terminated);
  await context.close();

  const unavailable = await browser.newContext();
  await unavailable.addInitScript(() => { Object.defineProperty(window, 'localStorage', { get() { throw new Error('Explicit storage denial'); } }); });
  const fallback = await unavailable.newPage(); fallback.on('pageerror', e => result.errors.push(e.message));
  await fallback.goto(base); await fallback.locator('[data-boltzmann-state="ready"]').waitFor();
  await fallback.locator('#experiment').getByText('브라우저 저장을 사용할 수 없습니다.', { exact: true }).waitFor();
  await fallback.getByRole('radio').first().check(); await fallback.getByRole('button', { name: '답과 해설 확인', exact: true }).first().click();
  await fallback.locator('.ai-answer').waitFor(); await unavailable.close();
  console.log('Control group passed'); result.checks.push('language/theme preserve current calculation; worker cleanup; storage-denied lab and quiz work');
  assert.deepEqual(result.errors, []);
  await fs.writeFile(`${root}/controls-result.json`, JSON.stringify(result, null, 2) + '\n'); console.log(JSON.stringify(result));
} finally { await browser.close(); }
