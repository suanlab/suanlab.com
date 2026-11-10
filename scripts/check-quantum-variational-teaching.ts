/** Local component QA; does not register units or rebuild the existing export. */
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { build } from 'esbuild';
import { chromium } from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import { quantumVariationalTeaching as teaching } from '../src/data/quantum-variational-teaching';
import { quantumVariationalSteps as steps } from '../src/lib/learning-quantum-variational-concepts';
import { quantumVariationalKinds as kinds, quantumVariationalPresets as presets } from '../src/lib/learning-quantum-variational';
const root = process.env.SUANLAB_QV_QA_OUTPUT ?? '.runtime/learning/quantum-variational-teaching';
async function main() {
  const source = `import React from 'react';import{createRoot}from'react-dom/client';import{LanguageProvider,useLanguage}from'./src/components/language-provider';import{SiteMotionProvider}from'./src/components/motion/motion-provider';import Lab from'./src/components/learning/quantum-variational-lab';import Concept from'./src/components/learning/quantum-variational-concept';import Quiz from'./src/components/learning/learning-quiz';import{quantumVariationalTeaching}from'./src/data/quantum-variational-teaching';function App(){const{language}=useLanguage();return <div className="ai-book"><main id="main-content" style={{maxWidth:1100,margin:'auto',padding:16}}><h1>Quantum variational experiments</h1><Concept kind={window.qvKind} language={language}/><Lab kind={window.qvKind}/><Quiz unit={quantumVariationalTeaching.find(v=>v.unit.id===window.qvKind).unit}/></main></div>}createRoot(document.getElementById('root')).render(<LanguageProvider><SiteMotionProvider><App/></SiteMotionProvider></LanguageProvider>);`;
  const client = await build({ stdin: { contents: source, resolveDir: process.cwd(), loader: 'tsx' }, bundle: true, write: false, metafile: true, jsx: 'automatic', format: 'esm', platform: 'browser', target: 'es2022', define: { 'process.env.NODE_ENV': '"production"' } });
  const worker = await build({ entryPoints: ['src/lib/learning-quantum-variational.worker.ts'], bundle: true, write: false, format: 'esm', platform: 'browser', target: 'es2022' });
  const graph = Object.keys(client.metafile!.inputs);
  assert(!graph.some(p => /learning-(rayleigh|ritz|tensor-roots|quantum-variational-evaluator)\.ts$/.test(p)));
  writeFileSync(`${root}/browser-dependency-graph.json`, JSON.stringify(graph, null, 2));
  const exported = readFileSync('out/book/online/physics-math-for-ai/index.html', 'utf8');
  const css = [...new Set([...exported.matchAll(/href="([^"]+\.css)"/g)].map(m => m[1]))].map(href => `<link rel="stylesheet" href="${href}">`).join('');
  assert(css);
  const html = `<!doctype html><html lang="en"><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>Quantum teaching QA</title>${css}</head><body><div id="root"></div><script type="module" src="/__qv-teaching.js"></script></body></html>`;
  writeFileSync(`${root}/teaching-client.js`, client.outputFiles[0].text);
  const browser = await chromium.launch({ headless: true });
  const report = { cases: 0, examples: 0, reloads: 0, invalidInputs: 0, jsonEdits: 0, dimensions: 0, axeCases: 0, predictions: 0, quizzes: 0, guides: 0, errors: [] as string[], readerIntegrated: false };
  try {
    for (const kind of kinds) for (const language of ['ko', 'en'] as const) for (const width of [320, 1440]) {
      const context = await browser.newContext({ viewport: { width, height: 960 }, reducedMotion: 'reduce' });
      await context.addInitScript(({ kind, language, width }) => {
        (window as any).qvKind = kind;
        localStorage.setItem('suanlab-language', language);
        document.addEventListener('DOMContentLoaded', () => document.documentElement.classList.toggle('dark', width === 1440));
      }, { kind, language, width });
      const page = await context.newPage();
      page.on('pageerror', e => report.errors.push(e.message));
      await page.route('**/__qv-teaching/', r => r.fulfill({ contentType: 'text/html', body: html }));
      await page.route('**/__qv-teaching.js', r => r.fulfill({ contentType: 'text/javascript', body: client.outputFiles[0].text }));
      await page.route('**/learning-quantum-variational.worker.ts', r => r.fulfill({ contentType: 'text/javascript', body: worker.outputFiles[0].text }));
      await page.goto('http://localhost:4175/__qv-teaching/');
      const lab = page.locator('[data-qv-lab]');
      const exampleGroup = lab.getByRole('group', { name: language === 'ko' ? '실험 예제' : 'Experiment examples', exact: true });
      const ready = () => page.locator('[data-qv-lab][data-qv-state="ready"]').waitFor({ timeout: 120000 });
      await ready();
      await page.locator('[data-qv-concept-state="ready"]').waitFor({ timeout: 120000 });
      const examples = width === 1440 && language === 'en' ? presets[kind].map((_, i) => i) : [0, 1, 3];
      for (const i of examples) {
        await exampleGroup.getByRole('button', { name: presets[kind][i].label[language], exact: true }).click();
        await ready();
        assert.equal(await lab.locator('[data-qv-result]').count(), 1);
        report.examples++;
      }
      await exampleGroup.getByRole('button', { name: presets[kind][0].label[language], exact: true }).click();
      await ready();
      for (let i = 0; i < 3; i++) {
        await page.locator('.ai-concept-steps').getByRole('button').nth(i).click();
        await page.locator('[data-qv-concept-state="ready"]').waitFor({ timeout: 120000 });
        const question = page.locator('[data-qv-concept] fieldset.ai-question');
        await question.getByRole('radio').nth(steps[kind][i].answer).check();
        await question.getByRole('button').click();
        assert.equal(await question.locator('.ai-answer-correct').count(), 1);
        report.predictions++;
      }
      for (const [i, q] of teaching.find(v => v.unit.id === kind)!.unit.questions.entries()) {
        const field = page.locator('#check fieldset').nth(i);
        await field.getByRole('radio').nth(q.answer).check();
        await field.getByRole('button').click();
        assert.equal(await field.locator('.ai-answer-correct').count(), 1);
        report.quizzes++;
      }
      const guideDetails = lab.locator('details').filter({ has: page.locator('[data-qv-guide]') });
      await guideDetails.locator('summary').click();
      for (let i = 0; i < 3; i++) {
        await lab.locator(`[data-qv-guide="${i}"]`).getByRole('button').first().click(); await ready(); report.guides++;
      }
      await exampleGroup.getByRole('button', { name: presets[kind][0].label[language], exact: true }).click(); await ready();
      const initialKey = await lab.getAttribute('data-qv-input');
      const editDetails = lab.locator('details').filter({ has: page.locator('[data-qv-field]') });
      await editDetails.locator('summary').click();
      const field = lab.locator('[data-qv-field="hDenominator"]');
      await field.fill('0'); await field.press('Enter');
      assert.equal(await field.getAttribute('aria-invalid'), 'true');
      assert.equal(await lab.getAttribute('data-qv-input'), initialKey);
      await field.fill('2'); await field.press('Enter'); await ready();
      report.invalidInputs++;
      const changedKey = await lab.getAttribute('data-qv-input');
      await page.reload(); await ready();
      assert.equal(await lab.getAttribute('data-qv-input'), changedKey);
      report.reloads++;
      for (const [i, q] of teaching.find(v => v.unit.id === kind)!.unit.questions.entries()) {
        assert(await page.locator('#check fieldset').nth(i).getByRole('radio').nth(q.answer).isChecked());
      }
      const jsonDetails = lab.locator('details').filter({ has: page.locator('[data-qv-json]') });
      await jsonDetails.locator('summary').click();
      await lab.locator('[data-qv-json]').fill('{}');
      await lab.getByRole('button', { name: language === 'ko' ? 'JSON 적용' : 'Apply JSON', exact: true }).click();
      assert.equal(await lab.locator('[data-qv-json]').getAttribute('aria-invalid'), 'true');
      assert.equal(await lab.getAttribute('data-qv-input'), changedKey);
      await lab.locator('[data-qv-json]').fill(JSON.stringify(presets[kind][0].input));
      await lab.getByRole('button', { name: language === 'ko' ? 'JSON 적용' : 'Apply JSON', exact: true }).click();
      await ready(); report.jsonEdits++;
      const dimensions = lab.locator('details').filter({ has: page.locator('select') });
      await dimensions.locator('summary').click();
      await dimensions.locator('select').first().selectOption('3');
      if (kind === kinds[1]) await dimensions.locator('select').nth(1).selectOption('2');
      await dimensions.getByRole('button', { name: language === 'ko' ? '새 공간 적용' : 'Apply new space', exact: true }).click();
      await ready(); report.dimensions++;
      if (language === 'en') assert(!/[가-힣]/.test(await page.locator('body').innerText()));
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
      const axe = await new AxeBuilder({ page }).include('#root').withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze();
      writeFileSync(`${root}/axe-${kinds.indexOf(kind)}-${language}-${width}.json`, JSON.stringify(axe.violations, null, 2));
      assert.deepEqual(axe.violations, []);
      report.axeCases++;
      await page.screenshot({ path: `${root}/preview-${kinds.indexOf(kind)}-${language}-${width}.png` });
      report.cases++; console.log(`PASS ${kind}/${language}/${width}`);
      await context.close();
    }
    assert.deepEqual(report.errors, []);
    writeFileSync(`${root}/browser-verification.json`, JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report));
  } finally { await browser.close(); }
}
main().catch(e => { console.error(e); process.exitCode = 1; });
