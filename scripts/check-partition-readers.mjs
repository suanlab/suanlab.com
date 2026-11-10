/* Actual exported reader routes, using the existing local preview server. */
import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import AxeBuilder from '@axe-core/playwright';
const root = '.runtime/learning/partition-integration', base = 'http://127.0.0.1:4175';
const { teaching, presets } = JSON.parse(await fs.readFile(`${root}/specs.json`, 'utf8'));
const result = { concepts: 0, guides: 0, quizChoices: 0, presets: 0, axeStates: 0, reloads: 0, errors: [], workerUrls: [], cases: [] };
let saveQueue = Promise.resolve();
const save = () => { const data = JSON.stringify(result, null, 2) + '\n'; saveQueue = saveQueue.then(() => fs.writeFile(`${root}/reader-progress.json`, data)); return saveQueue; };
const browser = await chromium.launch({ headless: true, args: ['--disable-dev-shm-usage'] });
try {
  async function verify({ language, dark, width }) {
    const context = await browser.newContext({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
    await context.addInitScript(({ language, dark }) => { localStorage.setItem('suanlab-language', language); localStorage.setItem('theme', dark ? 'dark' : 'light'); document.addEventListener('DOMContentLoaded', () => document.documentElement.classList.toggle('dark', dark)); }, { language, dark });
    const page = await context.newPage(); page.on('pageerror', error => result.errors.push(error.message));
    page.on('worker', worker => { if (!result.workerUrls.includes(worker.url())) result.workerUrls.push(worker.url()); });
    const ready = (kind, key) => page.waitForFunction(({ kind, key }) => { const lab = document.querySelector('[data-partition-lab]'); return lab?.getAttribute('data-partition-lab') === kind && lab.getAttribute('data-partition-current') === key; }, { kind, key }, { timeout: 60000 });
    const audit = async () => {
      const a = await new AxeBuilder({ page }).include('main').analyze();
      assert.deepEqual(a.violations.map(v => ({ id: v.id, nodes: v.nodes.map(n => n.target) })), []);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
      assert.equal(await page.locator('main svg').evaluateAll(els => els.some(el => /NaN|Infinity|="undefined"/.test(el.outerHTML))), false);
      if (language === 'en') assert.equal(/[가-힣]/.test(await page.locator('main').innerText()), false);
      await page.evaluate(() => document.fonts.ready);
      const clipped = await page.locator('[data-partition-concept] svg').evaluateAll(svgs => svgs.flatMap(svg => {
        const box = svg.viewBox.baseVal;
        return [...svg.querySelectorAll('text')].filter(text => {
          const b = text.getBBox(); return b.x < box.x - .5 || b.x + b.width > box.x + box.width + .5 || b.y < box.y - .5 || b.y + b.height > box.y + box.height + .5;
        }).map(text => text.textContent);
      }));
      assert.deepEqual(clipped, [], 'Concept labels must remain inside the SVG viewport');
      const wrappedNumbers = await page.locator('main table td').evaluateAll(cells => cells.filter(cell => {
        if (!/^[+−-]?\d+(?:\.\d+)?e[+-]?\d+$/i.test(cell.textContent.trim())) return false;
        const range = document.createRange(); range.selectNodeContents(cell); return range.getClientRects().length > 1;
      }).map(cell => cell.textContent));
      assert.deepEqual(wrappedNumbers, [], 'Scientific notation stays on one line');
      result.axeStates++;
    };
    for (const kind of Object.keys(teaching)) {
      await page.goto(`${base}/book/online/physics-math-for-ai/${kind}/`);
      await page.waitForFunction(language => document.documentElement.lang === language && document.querySelector('[data-partition-lab]')?.getAttribute('data-partition-ready') === 'true', language);
      await ready(kind, presets[kind][0].key);
      const lab = page.locator('[data-partition-lab]');
      for (let step = 0; step < 4; step++) {
        await page.locator('.ai-concept-steps button').nth(step).click();
        await page.waitForFunction(step => { const concept = document.querySelector('[data-partition-concept]'); return concept?.getAttribute('data-partition-step') === String(step) && concept.getAttribute('data-partition-concept-state') === 'ready' && concept.querySelector('svg'); }, step);
        await audit(); result.concepts++;
      }
      for (let guide = 0; guide < 3; guide++) {
        await lab.locator('.ai-guided-choices button').nth(guide).click();
        await lab.getByRole('button', { name: language === 'en' ? 'Load this example' : '이 예제 열기', exact: true }).click();
        await ready(kind, presets[kind][teaching[kind].guide.experiments[guide].preset].key);
        const answer = lab.locator('.ai-guided-lab details'); if ((await answer.getAttribute('open')) === null) await answer.locator('summary').click();
        assert.ok((await answer.innerText()).includes(teaching[kind].guide.experiments[guide].explain[language])); result.guides++;
      }
      for (const [i, preset] of presets[kind].entries()) {
        await lab.locator('.ai-presets button').nth(i).click(); await ready(kind, preset.key); await audit(); result.presets++;
      }
      await page.waitForFunction(() => document.querySelectorAll('.ai-question').length === 2 && [...document.querySelectorAll('.ai-question')].every(el => !el.disabled));
      for (let q = 0; q < 2; q++) for (let choice = 0; choice < 3; choice++) {
        const field = page.locator('.ai-question').nth(q); await field.locator('input[type=radio]').nth(choice).check();
        await field.getByRole('button').click(); await field.locator('[role=status]').waitFor();
        assert.equal(await field.locator('.ai-answer-correct').count() === 1, choice === teaching[kind].unit.questions[q].answer); result.quizChoices++;
      }
      await page.locator('.ai-completion input').check(); await audit();
      await page.waitForFunction(kind => { const saved = JSON.parse(localStorage.getItem('suanlab-learning:' + kind + ':study')); return saved?.value?.complete === true && Object.values(saved.value.answers).length === 2 && Object.values(saved.value.answers).every(v => v === 2); }, kind);
      await page.reload(); await ready(kind, presets[kind].at(-1).key);
      await page.waitForFunction(() => document.querySelectorAll('.ai-question').length === 2 && [...document.querySelectorAll('.ai-question')].every(el => !el.disabled));
      assert.deepEqual(await page.locator('.ai-question input[type=radio]:checked').evaluateAll(els => els.map(el => el.value)), ['2', '2']);
      assert.equal(await page.locator('.ai-completion input').isChecked(), true); result.reloads++;
      const caseResult = { kind, language, dark, width }; result.cases.push(caseResult); await save(); console.log(JSON.stringify({ ...caseResult, axe: result.axeStates }));
    }
    for (const url of result.workerUrls) { assert.ok(url.startsWith(base + '/_next/'), url); const response = await context.request.get(url); assert.equal(response.status(), 200); }
    await context.close();
  }
  const settings = ['ko', 'en'].flatMap(language => [false, true].flatMap(dark => [320, 1440].map(width => ({ language, dark, width }))));
  let cursor = 0;
  // Independent storage/theme contexts; at most four active pages in one owned browser.
  await Promise.all(Array.from({ length: 4 }, async () => { while (cursor < settings.length) { const setting = settings[cursor++]; await verify(setting); } }));
  assert.equal(result.errors.length, 0); assert.equal(result.cases.length, 24); assert.equal(result.concepts, 96); assert.equal(result.guides, 72); assert.equal(result.quizChoices, 144); assert.equal(result.presets, 216);
  await fs.writeFile(`${root}/reader-result.json`, JSON.stringify(result, null, 2) + '\n');
} finally { await browser.close(); }
