/** Actual registered readers with native Next worker assets and client navigation. */
import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
import { chromium, type Page, type BrowserContext } from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import { finiteSizeKinds as kinds, finiteSizePresets as presets, finiteSizeGuides as guides, finiteSizePresetInput as preset } from '../src/lib/learning-finite-size';
import { finiteSizeInputKey as key, type FiniteSizeInput } from '../src/lib/learning-finite-size-protocol';
import { finiteSizeFields as fields, advanceFiniteSize as advance, editFiniteSizeDraft as edit, editFiniteSizeChoice as choose } from '../src/lib/learning-finite-size-controls';
const destination = '.runtime/learning/finite-size-integration';
import {finiteSizeConceptSteps as concepts,finiteSizeConceptInputs as samples} from '../src/lib/learning-finite-size-concepts';
const controlsOnly = process.argv.includes('--controls-only');
interface Stats { created: number; terminated: number; started: number; stoppedAtAck: number; stale: string[] }
async function main() {
 mkdirSync(destination,{recursive:true});
 const base=process.env.FINITE_SIZE_READER_BASE_URL??'http://localhost:4175',book='/book/online/physics-math-for-ai/';let failWorker=false;const workerUrls=new Set<string>();
 const browser = await chromium.launch({ headless: true, args: ['--no-sandbox'] });
 const report = { scope: controlsOnly ? 'Additional native choice/constraint/replacement controls' : 'Actual homepage reader teaching flow and native worker assets', conceptSteps:0,predictionChoices:0,fixedPreparationChoices:0,controlCases: 0, inactiveFieldChecks: 0, choiceConstraintCases: 0, choiceReloads: 0, displayCases: 0, guideExplanations: 0, guideSelections: 0, presetSelections: 0, manualSteps: 0, resets: 0, quizChoices: 0, reloads: 0, invalidDrafts: 0, customEdits: 0, axeStates: 0, storageCases: 0, ackLinkedStops: 0, retryRecoveries: 0, workerFailureRecoveries: 0, inFlightReplacements: 0, autoProgressCases: 0, globalPauses: 0, offscreenPauses: 0, reducedMotionPauses: 0, focusPauses: 0, unmountRemounts: 0, workersCreated: 0, workersTerminated: 0, workersStarted: 0, staleResults: [] as string[], errors: [] as string[] };
 const storageKey = (kind: string) => `suanlab-finite-size-${kind}`;
 async function instrument(context: BrowserContext, language: string, dark: boolean) {
  await context.route('**/_next/static/**',route=>failWorker&&workerUrls.has(route.request().url())?route.abort('failed'):route.continue());
  await context.addInitScript(({ language, dark }) => {
   localStorage.setItem('suanlab-language', language); localStorage.setItem('theme',dark?'dark':'light'); localStorage.setItem('suanlab-motion-paused', 'false');
   document.addEventListener('DOMContentLoaded', () => document.documentElement.classList.toggle('dark', dark));
   const stats = (window as any).labStats = { created: 0, terminated: 0, started: 0, stoppedAtAck: 0, live: new Set(), urls:new Set<string>(), stale: [] as string[], cancelAt: null as string | null, replaceAt: null as string | null, replacedAtAck: 0 };
   const Native = window.Worker;
   window.Worker = class extends Native {
    ended = false; inputKey: string | null = null;
    constructor(url: string | URL, options?: WorkerOptions) { super(url, options); stats.urls.add(new URL(String(url),location.href).href); stats.created++; stats.live.add(this); this.addEventListener('message', event => { if (event.data?.phase !== 'started') return; stats.started++; if (stats.cancelAt === this.inputKey) { stats.cancelAt = null; queueMicrotask(() => { const lab = document.querySelector('[data-finite-size-lab]'), stop = Array.from(lab?.querySelectorAll('button') ?? []).find(button => button.textContent === (language === 'ko' ? '계산 중단' : 'Stop calculation')); if (stop) { stop.click(); stats.stoppedAtAck++; } }); } else if (stats.replaceAt === this.inputKey) { stats.replaceAt = null; queueMicrotask(() => { const first = document.querySelector('[data-finite-size-lab] .ai-presets button') as HTMLButtonElement | null; if (first) { first.click(); stats.replacedAtAck++; } }); } }); }
    postMessage(value: any) { this.inputKey = value?.input ? JSON.stringify(value.input) : null; super.postMessage(value); }
    terminate() { if (!this.ended) { this.ended = true; stats.terminated++; stats.live.delete(this); } super.terminate(); }
   };
   new MutationObserver(() => { const lab = document.querySelector('[data-finite-size-lab]'), result = lab?.querySelector('[data-finite-size-result-key]'); if (result && result.getAttribute('data-finite-size-result-key') !== lab!.getAttribute('data-finite-size-input')) stats.stale.push('old result under new input'); }).observe(document, { attributes: true, childList: true, subtree: true });
  }, { language, dark });
 }
 const ready = async (page: Page, input: FiniteSizeInput) => { await page.waitForFunction(expected => document.querySelector('[data-finite-size-lab]')?.getAttribute('data-finite-size-current') === expected, key(input), { timeout: 90000 }); };
 const state = async (page: Page, expected: string) => { await page.waitForFunction(expected => document.querySelector('[data-finite-size-lab]')?.getAttribute('data-finite-size-state') === expected, expected, { timeout: 90000 }); };
 async function setup(context: BrowserContext, family: number, language: string, chooseLanguage = false) { const page = await context.newPage(); page.on('pageerror', error => report.errors.push(error.message)); await page.goto(`${base}${book}${kinds[family]}/`); await ready(page, preset(kinds[family], 0)); if(chooseLanguage) await page.getByRole('button',{name:language==='en'?'EN':'KO',exact:true}).click(); await page.waitForFunction(language => document.documentElement.lang === language, language); await page.locator('[data-finite-size-concept-state="ready"]').waitFor({ timeout: 90000 }); for(const url of await page.evaluate(()=>[...(window as any).labStats.urls]as string[]))workerUrls.add(url);return page; }
 async function check(page: Page, language: string, axe = true) { assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Document overflow'); if (language === 'en') assert.doesNotMatch(await page.locator('body').innerText(), /[가-힣]/); if (axe) { const result = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']).analyze(); assert.deepEqual(result.violations.map(v => ({ id: v.id, nodes: v.nodes.map(n => n.target) })), []); report.axeStates++; } }
 async function finish(context: BrowserContext, page: Page) { await page.locator('main .ai-reader-top a').first().click(); await page.waitForURL(`**${book}`); await page.waitForFunction(() => (window as any).labStats.live.size === 0); const stats: Stats = await page.evaluate(() => { const s = (window as any).labStats; return { created: s.created, terminated: s.terminated, started: s.started, stoppedAtAck: s.stoppedAtAck, stale: s.stale }; }); assert.equal(stats.created, stats.terminated); report.workersCreated += stats.created; report.workersTerminated += stats.terminated; report.workersStarted += stats.started; report.staleResults.push(...stats.stale); await context.close(); }
 async function field(page: Page, id: string, value: string) { const locator = page.locator(`[data-finite-size-field="${id}"]`); await locator.fill(value); await locator.press('Enter'); }
 async function show(page: Page, kind: typeof kinds[number], index: number) { await page.locator('.ai-presets button').nth(index).click(); await ready(page, preset(kind, index)); }
 try {
  if (!controlsOnly) {
  for (const language of ['ko', 'en'] as const) for (const dark of [false, true]) for (const width of [320, 1440]) for (const [family, kind] of kinds.entries()) {
   const context = await browser.newContext({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' }); await instrument(context, language, dark); const page = await setup(context, family, language), lab = page.locator('[data-finite-size-lab]');
   // Covers belong to the contents cards and are verified by check-finite-size-navigation.ts.
   await check(page, language); report.displayCases++;
   for (let guideIndex = 0; guideIndex < 3; guideIndex++) {
    const guide = guides[kind][guideIndex]; await lab.locator('.ai-guided-choices button').nth(guideIndex).click(); const details = lab.locator('.ai-guided-lab details'); await details.locator('summary').click(); assert.ok((await details.innerText()).includes(guide.explain[language])); report.guideExplanations++;
    await lab.locator('.ai-guided-lab .ai-lab-actions button').first().click(); await ready(page, preset(kind, guide.presetIndices[0])); report.guideSelections++;
   }
   await lab.getByRole('button', { name: language === 'ko' ? '모든 실험 설정 초기화' : 'Reset all experiment settings', exact: true }).click(); await ready(page, preset(kind, 0));
   const manual = advance(preset(kind, 0))!;
   await lab.getByRole('button', { name: language === 'ko' ? '공간 크기 한 단계' : 'One spatial-size step', exact: true }).click(); await ready(page, manual); report.manualSteps++;
   assert.equal(await lab.locator('.ai-presets button[aria-pressed="true"]').count(), presets[kind].some(p => key(p.input) === key(manual)) ? 1 : 0);
   await lab.getByRole('button', { name: language === 'ko' ? '모든 실험 설정 초기화' : 'Reset all experiment settings', exact: true }).click(); await ready(page, preset(kind, 0)); report.resets++;
   const auto = lab.getByRole('button', { name: language === 'ko' ? '공간 크기 자동 순회' : 'Scan spatial sizes automatically', exact: true }); assert.equal(await auto.isDisabled(), true);
   for (const descriptor of fields(preset(kind, 0))) assert.equal(await page.locator(`[data-finite-size-field="${descriptor.id}"]`).inputValue(), String(descriptor.value));
   const quiz = page.locator('#check'); for (let q = 0; q < 3; q++) for (let answer = 0; answer < 3; answer++) { const question = quiz.locator('fieldset').nth(q); await question.locator('input[type=radio]').nth(answer).check(); await question.getByRole('button').click(); assert.ok(await question.locator('.ai-answer').isVisible()); report.quizChoices++; }
   await quiz.locator('input[type=checkbox]').check();
   // Every preset is selected in both languages in the narrow dark layout.
   if (dark && width === 320) {
    for(let step=0;step<6;step++){
     await page.locator('.ai-concept-steps button').nth(step).click();const concept=page.locator(`[data-finite-size-step="${step}"][data-finite-size-concept-state="ready"]`);await concept.waitFor({timeout:90000});
     const fixed=concept.getByRole('group',{name:language==='ko'?'고정 준비 선택':'Choose a fixed preparation'}).getByRole('button');for(let i=0;i<samples(kind,step).length;i++){await fixed.nth(i).click();assert.equal(await concept.locator('[data-finite-size-result-key]').getAttribute('data-finite-size-result-key'),key(samples(kind,step)[i]));report.fixedPreparationChoices++;}
     for(let i=0;i<3;i++){await concept.locator('input[type=radio]').nth(i).check();await concept.getByRole('button',{name:language==='ko'?'예측 확인':'Check prediction',exact:true}).click();assert.ok((await concept.locator('.ai-answer strong').innerText()).includes(i===concepts[kind][step].check.answer?language==='ko'?'맞았습니다.':'Correct.':language==='ko'?'다시 생각해보세요.':'Try again.'));report.predictionChoices++;}report.conceptSteps++;
    }
    for (let index = 0; index < presets[kind].length; index++) { await show(page, kind, index); assert.equal(await lab.locator('.ai-presets button[aria-pressed="true"]').innerText(), presets[kind][index].label[language]); await check(page, language, false); report.presetSelections++; }
    await show(page, kind, 0);
    const first = fields(preset(kind, 0))[0];
    for (const invalid of ['1e-999', '1.0', '-1', '1000001']) { await field(page, first.id, invalid); assert.equal(await page.locator(`[data-finite-size-field="${first.id}"]`).getAttribute('aria-invalid'), 'true'); assert.equal(await lab.getAttribute('data-finite-size-input'), key(preset(kind, 0))); report.invalidDrafts++; }
    const invalidId = family === 0 ? 'width' : family === 1 ? 'commonNumerator' : 'activityDenominator'; await field(page, invalidId, family === 1 ? '3' : '0'); assert.equal(await page.locator(`[data-finite-size-field="${invalidId}"]`).getAttribute('aria-invalid'), 'true'); report.invalidDrafts++;
    await lab.getByRole('button', { name: language === 'ko' ? '모든 실험 설정 초기화' : 'Reset all experiment settings', exact: true }).click(); await ready(page, preset(kind, 0)); assert.equal(await lab.locator('[aria-invalid="true"]').count(), 0); report.resets++;
    const custom = edit(preset(kind, 0), first.id, String(first.value + 1))!; await field(page, first.id, String(first.value + 1)); await ready(page, custom); assert.equal(await lab.locator('.ai-presets button[aria-pressed="true"]').count(), 0); report.customEdits++;
    await field(page, first.id, String(first.value)); await ready(page, preset(kind, 0)); assert.equal(await lab.locator('.ai-presets button[aria-pressed="true"]').innerText(), presets[kind][0].label[language]);
    await check(page, language);
    if (language === 'en') { await lab.evaluate(element => element.scrollIntoView({ block: 'start' })); await page.screenshot({ path: `${destination}/${family}-lab-en-dark-mobile.png` }); }
   }
   const before = await lab.getAttribute('data-finite-size-input'); const prior: Stats = await page.evaluate(() => { const s = (window as any).labStats; return { created: s.created, terminated: s.terminated, started: s.started, stoppedAtAck: s.stoppedAtAck, stale: s.stale }; }); await page.reload(); await ready(page, JSON.parse(before!)); await page.locator('[data-finite-size-concept-state="ready"]').waitFor({ timeout: 90000 }); report.reloads++;
   assert.equal(await page.locator('#check input[type=radio]:checked').count(), 3); assert.equal(await page.locator('#check input[type=checkbox]').isChecked(), true); report.workersCreated += prior.created; report.workersTerminated += prior.terminated; report.workersStarted += prior.started; report.staleResults.push(...prior.stale);
   await finish(context, page); console.log(JSON.stringify({ kind, language, dark, width, displayCases: report.displayCases }));
  }
  // Explicit native UI lifecycle and motion checks, separate from reduced-motion display cases.
  for (const [family, kind] of kinds.entries()) {
   const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, reducedMotion: 'no-preference' }); await instrument(context, 'en', false); const page = await setup(context, family, 'en'), lab = page.locator('[data-finite-size-lab]');
   const heavyIndex = [10,14,12][family], heavy = preset(kind, heavyIndex);
   await page.evaluate(expected => { (window as any).labStats.cancelAt = expected; }, key(heavy));
   await lab.locator('.ai-presets button').nth(heavyIndex).click();
   await state(page, 'stopped'); assert.equal(await lab.locator('[data-finite-size-result]').count(), 0); assert.equal(await page.evaluate(() => (window as any).labStats.stoppedAtAck), 1); report.ackLinkedStops++;
   await lab.getByRole('button', { name: 'Retry calculation', exact: true }).click(); await ready(page, heavy); report.retryRecoveries++;
   await show(page, kind, 0);
   failWorker = true; await lab.locator('.ai-presets button').nth(1).click(); await state(page, 'error'); assert.equal(await lab.locator('[data-finite-size-result]').count(), 0); failWorker = false; await lab.getByRole('button', { name: 'Retry calculation', exact: true }).click(); await ready(page, preset(kind, 1)); report.workerFailureRecoveries++;
   await show(page, kind, 0);
   const playName = 'Scan spatial sizes automatically', pauseName = 'Pause spatial-size scan';
   const play = () => lab.getByRole('button', { name: playName, exact: true });
   await play().scrollIntoViewIfNeeded(); await play().waitFor(); await page.waitForFunction(() => Array.from(document.querySelectorAll('[data-finite-size-lab] button')).some(b => ['Scan spatial sizes automatically'].includes(b.textContent!) && !(b as HTMLButtonElement).disabled)); await play().click();
   await ready(page, advance(preset(kind, 0))!); await lab.getByRole('button', { name: pauseName, exact: true }).click(); report.autoProgressCases++;
   await show(page, kind, 0); await play().click(); await page.locator('footer').getByRole('button', { name: 'Pause site motion', exact: true }).evaluate((button: HTMLButtonElement) => button.click()); await page.waitForFunction(() => document.querySelector('[data-finite-size-lab]')?.getAttribute('data-finite-size-playing') === 'false'); assert.equal(await play().isDisabled(), true); report.globalPauses++; await page.locator('footer').getByRole('button', { name: 'Play site motion', exact: true }).evaluate((button: HTMLButtonElement) => button.click());
   await play().click(); await page.evaluate(() => window.scrollTo(0, 0)); await page.waitForFunction(() => document.querySelector('[data-finite-size-lab]')?.getAttribute('data-finite-size-playing') === 'false'); report.offscreenPauses++;
   await play().scrollIntoViewIfNeeded(); await play().click(); await page.emulateMedia({ reducedMotion: 'reduce' }); await page.waitForFunction(() => document.querySelector('[data-finite-size-lab]')?.getAttribute('data-finite-size-playing') === 'false'); assert.equal(await play().isDisabled(), true); report.reducedMotionPauses++;
   await page.emulateMedia({ reducedMotion: 'no-preference' }); await play().click(); await lab.locator('input').first().focus(); await page.waitForFunction(() => document.querySelector('[data-finite-size-lab]')?.getAttribute('data-finite-size-playing') === 'false'); report.focusPauses++;
   const saved = await lab.getAttribute('data-finite-size-input'); await page.locator('main .ai-reader-top a').first().click(); await page.waitForURL(`**${book}`); await page.waitForFunction(() => (window as any).labStats.live.size === 0); await page.locator(`main a.ai-unit-card[href="${book}${kind}/"]`).click(); await page.waitForURL(`**${book}${kind}/`); await ready(page, JSON.parse(saved!)); await page.locator('[data-finite-size-concept-state="ready"]').waitFor({ timeout: 90000 }); report.unmountRemounts++; await check(page, 'en'); await finish(context, page); console.log(JSON.stringify({ lifecycle: kind }));
  }
  // Malformed storage is replaced by defaults; unavailable storage keeps manual controls functional.
  const bad = ['not-json', JSON.stringify({ version: 2, value: preset(kinds[2], 0) }), JSON.stringify({ version: 1, value: preset(kinds[0], 0) }), JSON.stringify({ version: 1, value: { ...preset(kinds[2], 0), options: {} } }), JSON.stringify({ version: 1, value: { kind: kinds[2], parameters: { ...preset(kinds[2], 0).parameters, activityDenominator: 0 } } })];
  for (const raw of [...bad, 'read-blocked', 'write-blocked']) {
   const context = await browser.newContext({ viewport: { width: 320, height: 1000 }, reducedMotion: 'reduce' }); await instrument(context, 'en', true);
   await context.addInitScript(({ raw, key }) => { if (raw === 'read-blocked') Object.defineProperty(Storage.prototype, 'getItem', { value() { throw Error('Storage blocked'); } }); else if (raw === 'write-blocked') Object.defineProperty(Storage.prototype, 'setItem', { value() { throw Error('Quota exceeded'); } }); else localStorage.setItem(key, raw); }, { raw, key: storageKey(kinds[2]) });
   const page = await setup(context, 2, 'en', raw === 'read-blocked'), lab = page.locator('[data-finite-size-lab]'); assert.equal(await lab.getAttribute('data-finite-size-storage'), raw.endsWith('blocked') ? 'false' : 'true'); await lab.getByRole('button', { name: 'One spatial-size step', exact: true }).click(); await ready(page, advance(preset(kinds[2], 0))!); await check(page, 'en'); report.storageCases++; await finish(context, page);
  }
  }
  // Explicit choice editing, inactive inputs and replacements use a separate
  // six-case matrix; --controls-only supplements the full standalone audit.
  for (const language of ['ko', 'en'] as const) for (const [family, kind] of kinds.entries()) {
   const context = await browser.newContext({ viewport: { width: 320, height: 1000 }, reducedMotion: 'reduce' }); await instrument(context, language, true); const page = await setup(context, family, language), lab = page.locator('[data-finite-size-lab]'); let current = preset(kind, 0);
   const reset = async () => { await lab.getByRole('button', { name: language === 'ko' ? '모든 실험 설정 초기화' : 'Reset all experiment settings', exact: true }).click(); current = preset(kind, 0); await ready(page, current); };
   const select = async (id: string, raw: string) => { const next = choose(current, id, raw); assert.ok(next); await lab.locator(`[data-finite-size-choice="${id}"]`).selectOption(raw); current = next; await ready(page, current); };
   if (family === 0) {
    await lab.locator('[data-finite-size-choice="dimension"]').selectOption('1'); assert.equal(await lab.locator('[data-finite-size-choice="dimension"]').getAttribute('aria-invalid'), 'true'); assert.equal(await lab.getAttribute('data-finite-size-input'), key(current)); report.choiceConstraintCases++; await check(page, language);
    await reset(); current = edit(current, 'width', '1')!; await field(page, 'width', '1'); await ready(page, current); await select('dimension', '1'); await lab.locator('[data-finite-size-choice="growthMode"]').selectOption('fixed-thickness'); assert.equal(await lab.locator('[data-finite-size-choice="growthMode"]').getAttribute('aria-invalid'), 'true'); assert.equal(await lab.getAttribute('data-finite-size-input'), key(current)); report.choiceConstraintCases++; await check(page, language);
    await reset(); await select('growthMode', 'fixed-thickness'); await select('spinTwo', '1');
   } else if (family === 1) {
    await select('commonScaling', 'inverse-size'); current = edit(current, 'commonNumerator', '2')!; await field(page, 'commonNumerator', '2'); await ready(page, current); await field(page, 'commonDenominator', '1'); assert.equal(await lab.locator('[data-finite-size-field="commonDenominator"]').getAttribute('aria-invalid'), 'true'); assert.equal(await lab.getAttribute('data-finite-size-input'), key(current)); report.choiceConstraintCases++; await check(page, language);
    await reset(); await select('commonScaling', 'inverse-size-squared');
   } else {
    await select('sizePath', 'inverse-size');
    for (const descriptor of fields(current)) { assert.equal(await lab.locator(`[data-finite-size-field="${descriptor.id}"]`).isDisabled(), !descriptor.active); report.inactiveFieldChecks++; }
    await field(page, 'offsetNumerator', '3'); assert.equal(await lab.locator('[data-finite-size-field="offsetNumerator"]').getAttribute('aria-invalid'), 'true'); assert.equal(await lab.getAttribute('data-finite-size-input'), key(current)); report.choiceConstraintCases++; await check(page, language);
    await select('approachSide', '-1'); current = edit(current, 'offsetNumerator', '0')!; await field(page, 'offsetNumerator', '0'); await ready(page, current); await select('sizePath', 'fixed');
    assert.equal(await lab.locator('[data-finite-size-choice="approachSide"]').isDisabled(), true);
    for (const descriptor of fields(current)) { assert.equal(await lab.locator(`[data-finite-size-field="${descriptor.id}"]`).isDisabled(), !descriptor.active); report.inactiveFieldChecks++; }
    await select('sizePath', 'inverse-size'); assert.equal(await lab.locator('[data-finite-size-field="offsetNumerator"]').inputValue(), '0');
   }
   const prior: Stats = await page.evaluate(() => { const s = (window as any).labStats; return { created: s.created, terminated: s.terminated, started: s.started, stoppedAtAck: s.stoppedAtAck, stale: s.stale }; }); await page.reload(); await ready(page, current); await page.locator('[data-finite-size-concept-state="ready"]').waitFor({ timeout: 90000 }); report.workersCreated += prior.created; report.workersTerminated += prior.terminated; report.workersStarted += prior.started; report.staleResults.push(...prior.stale); report.choiceReloads++;
   const heavy = preset(kind, [10,14,12][family]); await page.evaluate(expected => { (window as any).labStats.replaceAt = expected; }, key(heavy)); await lab.locator('.ai-presets button').nth([10,14,12][family]).click(); await page.waitForFunction(() => (window as any).labStats.replacedAtAck === 1); await ready(page, preset(kind, 0)); report.inFlightReplacements++;
   await check(page, language); report.controlCases++; await finish(context, page); console.log(JSON.stringify({ controls: kind, language }));
  }
  if (!controlsOnly) { assert.equal(report.displayCases, 24); assert.equal(report.presetSelections, 88); assert.equal(report.ackLinkedStops, 3); assert.equal(report.autoProgressCases, 3); } assert.equal(report.controlCases, 6); assert.equal(report.inFlightReplacements, 6); assert.deepEqual(report.errors, []); assert.deepEqual(report.staleResults, []); assert.equal(report.workersCreated, report.workersTerminated);
  for(const url of workerUrls)assert.equal((await fetch(url)).status,200);writeFileSync(`${destination}/worker-assets.json`,JSON.stringify([...workerUrls],null,2));
  writeFileSync(`${destination}/${controlsOnly ? 'controls-browser-verification' : 'browser-verification'}.json`, JSON.stringify(report, null, 2)); console.log(JSON.stringify(report));
 } finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
