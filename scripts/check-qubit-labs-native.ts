import assert from 'node:assert/strict';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { build } from 'esbuild';
import { chromium, type Page } from 'playwright';
import AxeBuilder from '@axe-core/playwright';
import { qubitLabBrowserInstrumentation } from './learning/qubit-lab-browser-instrumentation';
import { qubitTeachingIds, qubitTeachingPresets } from '../src/lib/learning-qubit-teaching-input';

const runtime = '.runtime/learning/quantum-qubit-labs';
const smoke = process.argv.includes('--smoke');
const sha = (value: string | Buffer) => createHash('sha256').update(value).digest('hex');
async function main() {
  mkdirSync(runtime, { recursive: true });
  const files: Record<string, string> = {}, graphs: unknown[] = [];
  for (const [name, entry] of [['calculate', 'src/lib/learning-qubit.worker.ts'], ['validate', 'src/lib/learning-qubit-validation.worker.ts']]) {
    const result = await build({ entryPoints: [entry], bundle: true, write: false, platform: 'browser', format: 'iife', metafile: true, logLevel: 'silent' });
    files[name + '.js'] = result.outputFiles[0].text;
    graphs.push({ name, inputs: Object.keys(result.metafile!.inputs), dependencyHashes: Object.fromEntries(Object.keys(result.metafile!.inputs).map(p => [p, sha(readFileSync(p))])), sha256: sha(files[name + '.js']) });
  }
  const preview = await build({ entryPoints: ['scripts/learning/qubit-lab-preview.tsx'], bundle: true, write: false, outdir: runtime + '/assets', entryNames: 'preview', chunkNames: 'chunk-[hash]', platform: 'browser', format: 'esm', splitting: true, jsx: 'automatic', define: { 'process.env.NODE_ENV': '"production"' }, metafile: true, logLevel: 'silent', plugins: [{
    name: 'local-qa-worker-assets', setup(builder) {
      builder.onLoad({ filter: /learning-qubit-worker-factory\.ts$/ }, () => ({ loader: 'ts', contents: `export function createQubitWorker(){return new Worker('/__qubit_lab__/calculate.js',{type:'module'});}export function createQubitValidationWorker(){return new Worker('/__qubit_lab__/validate.js',{type:'module'});}` }));
    },
  }] });
  for (const file of preview.outputFiles) files['assets/' + file.path.split('/').at(-1)!] = file.text;
  const mainInputs = Object.keys(preview.metafile!.outputs[runtime + '/assets/preview.js'].inputs);
  const allowed = ['learning-qubit-input.ts', 'learning-qubit-protocol.ts', 'learning-qubit-controller.ts', 'learning-qubit-wire.ts', 'learning-qubit-worker-factory.ts', 'learning-qubit-teaching-input.ts', 'utils.ts'];
  assert(mainInputs.filter(p => p.startsWith('src/lib/')).every(p => allowed.includes(p.split('/').at(-1)!)));
  assert(!mainInputs.some(p => p.includes('node_modules/three/') || p.endsWith('hilbert-bloch-engine.ts')));
  assert(!Object.keys(preview.metafile!.inputs).includes('src/data/interactive-book.ts'));
  const cssUrls = [...readFileSync('out/book/interactive/index.html', 'utf8').matchAll(/href="([^\"]+\.css)"/g)].map(m => m[1]);
  for (const [name, code] of Object.entries(files)) { mkdirSync(runtime + '/' + name.split('/').slice(0, -1).join('/'), { recursive: true }); writeFileSync(runtime + '/' + name, code); }
  writeFileSync(runtime + '/dependency-graphs.json', JSON.stringify({ graphs, mainInputs, previewInputs: Object.keys(preview.metafile!.inputs), dependencyHashes: Object.fromEntries(Object.keys(preview.metafile!.inputs).map(p => [p, sha(readFileSync(p))])), files: Object.fromEntries(Object.entries(files).map(([p, code]) => [p, sha(code)])), css: cssUrls.map(url => ({ path: 'out' + url, sha256: sha(readFileSync('out' + url)) })), mainNumericalEnginesExcluded: true, threeOnlyInDynamicChunk: true, registeredRegistryExcluded: true, qaWorkerFactoryOverride: true }, null, 2) + '\n');
  const browser = await chromium.launch({ headless: true });
  const errors: string[] = [], states: unknown[] = [], flows: unknown[] = [], stateRoleChecks: unknown[] = [];
  const instrumentation = qubitLabBrowserInstrumentation;
  const ready = async (page: Page) => { await page.waitForFunction(() => document.querySelector('[data-qubit-lab]')?.getAttribute('data-qubit-phase') === 'ready'); };
  const assetRoutes = async (page: Page, language: string) => {
    page.on('pageerror', e => errors.push(e.message));
    await page.route('**/__qubit_lab__/**', route => {
      const name = new URL(route.request().url()).pathname.replace('/__qubit_lab__/', '');
      if (name === 'broken.js') return route.fulfill({ contentType: 'text/javascript', body: 'throw Error("intentional Worker startup failure")' });
      if (files[name]) return route.fulfill({ contentType: name.endsWith('.css') ? 'text/css' : 'text/javascript', body: files[name] });
      return route.fulfill({ contentType: 'text/html', body: `<!doctype html><html lang="${language}"><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qubit lab review</title>${cssUrls.map(url => `<link rel="stylesheet" href="${url}">`).join('')}<link rel="stylesheet" href="/__qubit_lab__/assets/preview.css"></head><body><div id="root"></div><script type="module" src="/__qubit_lab__/assets/preview.js"></script></body></html>` });
    });
    await page.route('**/_next/static/css/*.css', route => route.fulfill({ contentType: 'text/css', body: readFileSync('out' + new URL(route.request().url()).pathname) }));
  };
  try {
    for (const language of smoke ? ['en'] as const : ['ko', 'en'] as const) for (const dark of smoke ? [false] : [false, true]) for (const width of smoke ? [320] : [320, 1440]) {
      const context = await browser.newContext({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await context.addInitScript({ content: instrumentation });
      const page = await context.newPage(); await assetRoutes(page, language);
      for (const kind of qubitTeachingIds) {
        await page.goto(`http://localhost:4175/__qubit_lab__/?unit=${kind}&lang=${language}&theme=${dark ? 'dark' : 'light'}`);
        await ready(page); await page.waitForFunction(l => document.querySelector('main')?.getAttribute('data-language') === l, language);
        const all = page.getByLabel(language === 'ko' ? '전체 입력 예제' : 'All input examples', { exact: true });
        for (let index = 0; index < (smoke ? 1 : qubitTeachingPresets[kind].length); index++) {
          await all.selectOption(String(index)); await ready(page);
          assert.equal(await page.locator('[data-qubit-result]').getAttribute('data-qubit-result'), qubitTeachingPresets[kind][index].input.kind);
          if (language === 'en') assert(!/[가-힣]/.test(await page.locator('main').innerText()), `${kind}/${index} untranslated`);
          assert(!await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), `${kind}/${index} overflow`);
          const lifecycle = await page.evaluate(() => (window as unknown as { __qubitLifecycle: { created: number; terminated: number } }).__qubitLifecycle);
          assert.equal(lifecycle.created, lifecycle.terminated);
          const a11y = await new AxeBuilder({ page }).include('main').withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze();
          assert.deepEqual(a11y.violations, [], `${kind}/${index}/${language}/${dark}/${width} accessibility`);
          states.push({ kind, index, language, dark, width, accessibilityPassed: true, bodyOverflow: false, workersReleased: true });
        }
        // At least one representative Bloch interaction per lesson/view combination.
        await all.selectOption('0'); await ready(page);
        const figure = page.locator('[data-qubit-bloch]').first(); await figure.scrollIntoViewIfNeeded();
        await page.waitForFunction(() => document.querySelector('[data-qubit-bloch]')?.getAttribute('data-qubit-bloch') === '3d');
        await figure.getByRole('button', { name: language === 'ko' ? '정면' : 'Front', exact: true }).click();
        await figure.getByRole('button', { name: language === 'ko' ? '위에서' : 'Top', exact: true }).click();
        await figure.getByRole('button', { name: language === 'ko' ? '2D 보기' : 'Show 2D', exact: true }).click();
        assert.equal(await figure.getAttribute('data-qubit-bloch'), '2d');
        if (width === 320) { const region = figure.getByRole('region'); await region.focus(); await region.press('ArrowRight'); await page.waitForFunction(() => (document.querySelector('[data-qubit-bloch] .qubit-scroll')?.scrollLeft ?? 0) > 0); }
        if (language === 'en') await page.screenshot({ path: `${runtime}/${kind}-${dark ? 'dark' : 'light'}-${width}.png` });
        await figure.getByRole('button', { name: language === 'ko' ? '3D 보기' : 'Show 3D', exact: true }).click();
        await page.waitForFunction(() => document.querySelector('[data-qubit-bloch]')?.getAttribute('data-qubit-bloch') === '3d');
        await figure.locator('canvas').evaluate(canvas => { const gl = (canvas as HTMLCanvasElement).getContext('webgl2'); assertContext(gl); function assertContext(value: WebGL2RenderingContext | null) { if (!value?.getExtension('WEBGL_lose_context')) throw Error('WebGL loss extension unavailable'); value.getExtension('WEBGL_lose_context')!.loseContext(); } });
        await page.waitForFunction(() => document.querySelector('[data-qubit-bloch]')?.getAttribute('data-qubit-bloch') === '2d');
        await figure.getByRole('button', { name: language === 'ko' ? '3D 다시 시도' : 'Retry 3D', exact: true }).click();
        await page.waitForFunction(() => document.querySelector('[data-qubit-bloch]')?.getAttribute('data-qubit-bloch') === '3d');
        const inspect = page.getByLabel(language === 'ko' ? '자세히 볼 상태' : 'State to inspect', { exact: true });
        if (await inspect.count()) {
          const roles = await inspect.locator('option').evaluateAll(options => options.map(option => (option as HTMLOptionElement).value));
          for (const role of roles) {
            await inspect.selectOption(role);
            assert(!await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1));
            if (language === 'en') assert(!/[가-힣]/.test(await page.locator('main').innerText()));
            const check = await new AxeBuilder({ page }).include('main').withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze();
            assert.deepEqual(check.violations, [], `${kind}/${role} accessibility`);
            stateRoleChecks.push({ kind, role, language, dark, width, accessibilityPassed: true, bodyOverflow: false });
          }
        }
        // Persist a non-default example, reload, then inspect/restore an invalid draft.
        const last = qubitTeachingPresets[kind].length - 1;
        await all.selectOption(String(last)); await ready(page); await page.reload(); await ready(page);
        assert.equal(await page.getByLabel(language === 'ko' ? '전체 입력 예제' : 'All input examples', { exact: true }).inputValue(), String(last));
        await page.getByText(language === 'ko' ? '직접 입력 · 정확한 상태와 연산 수정' : 'Custom input · edit exact states and operations', { exact: true }).click();
        await page.getByLabel(language === 'ko' ? '정확 입력 JSON' : 'Exact input JSON', { exact: true }).fill('{');
        await page.getByRole('button', { name: language === 'ko' ? '입력 적용' : 'Apply input', exact: true }).click();
        assert.equal(await page.getByLabel(language === 'ko' ? '정확 입력 JSON' : 'Exact input JSON', { exact: true }).getAttribute('aria-invalid'), 'true');
        assert.equal(await page.locator('[data-qubit-lab]').getAttribute('data-qubit-phase'), 'ready');
        await page.getByRole('button', { name: language === 'ko' ? '현재 입력으로 되돌리기' : 'Restore current input', exact: true }).click();
        assert.equal(await page.getByLabel(language === 'ko' ? '정확 입력 JSON' : 'Exact input JSON', { exact: true }).getAttribute('aria-invalid'), 'false');
        await page.getByRole('button', { name: language === 'ko' ? '첫 예제로 초기화' : 'Reset to first example', exact: true }).click(); await ready(page);
        flows.push({ kind, language, dark, width, threeD: true, flatKeyboard: width === 320, webglLossAndRetry: true, inputReload: true, invalidDraftRetainedInput: true, reset: true });
        console.log(JSON.stringify({ progress: states.length, kind, language, dark, width }));
      }
      await context.close();
    }
    writeFileSync(runtime + (smoke ? '/native-smoke.json' : '/native-labs.json'), JSON.stringify({ states, flows, stateRoleChecks, accessibilityStates: states.length + stateRoleChecks.length, errors, actualNativeWorkers: true, actualReaderVerified: false, productionNextAssetsVerified: false, qaWorkerFactoryOverride: true }, null, 2) + '\n');
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ states: states.length, flows: flows.length, smoke }));
  } finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
