/** Check native text against every bilingual thermodynamics research SVG viewport. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { createHash } = require('node:crypto');
const { execFileSync } = require('node:child_process');
const { chromium } = require('playwright');

const report = process.env.SUANLAB_THERMO_FIGURES_REPORT || '.runtime/learning/thermo-svg-repair/figures.json';
const ids = JSON.parse(execFileSync('node_modules/.bin/tsx', ['-e', `
import { curriculumDomains } from './src/data/learning-curriculum';
console.log(JSON.stringify(curriculumDomains.find(d => d.id === 'thermo-statmech')!.unitIds));
`], { encoding: 'utf8' }));

(async () => {
  const { unified } = await import('unified');
  const { default: parse } = await import('remark-parse');
  const parser = unified().use(parse);
  const rows = [];
  const save = () => {
    fs.mkdirSync(path.dirname(report), { recursive: true });
    fs.writeFileSync(report, JSON.stringify({ units: ids, rows }, null, 2) + '\n');
  };
  const browser = await chromium.launch({
    executablePath: process.env.SUANLAB_BROWSER_PATH || '/usr/bin/google-chrome',
    args: ['--no-sandbox'],
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1200, height: 1400 } });
    assert.equal(ids.length, 32);
    for (const id of ids) {
      for (const lang of ['ko', 'en']) {
        const source = fs.readFileSync(`content/learning/physics-math-for-ai/${id}.${lang}.md`, 'utf8');
        const start = source.search(/^## (Research extension:|연구 확장:)/m);
        assert.ok(start >= 0, id);
        const end = source.indexOf('\n## ', start + 4);
        assert.ok(end > start, id);
        const images = [];
        const visit = node => {
          if (node.type === 'image') images.push(node);
          node.children?.forEach(visit);
        };
        visit(parser.parse(source.slice(start, end)));
        assert.equal(images.length, 1, id);
        const image = images[0];
        assert.ok(image.url.startsWith('/assets/learning/') && image.url.endsWith(`.${lang}.svg`), id);
        const filename = 'public' + image.url;
        const svg = fs.readFileSync(filename, 'utf8');
        await page.setContent('<style>body{margin:0}</style>' + svg.slice(svg.indexOf('<svg')));
        await page.evaluate(() => document.fonts.ready);
        const outside = await page.locator('svg').first().evaluate(svg => {
          const bounds = svg.getBoundingClientRect();
          return [...svg.querySelectorAll('text')].flatMap(node => {
            const box = node.getBoundingClientRect();
            const clipped = box.left < bounds.left - 1 || box.right > bounds.right + 1 ||
              box.top < bounds.top - 1 || box.bottom > bounds.bottom + 1;
            return clipped ? [node.textContent] : [];
          });
        });
        rows.push({ id, lang, path: filename, sha256: createHash('sha256').update(svg).digest('hex'), outside });
        save();
        assert.deepEqual(outside, [], filename + ': native SVG text must fit its viewport');
      }
    }
    assert.equal(rows.length, 64);
    assert.equal(new Set(rows.map(row => row.path)).size, 64);
    console.log('Thermodynamics research figures passed: 32 units, 64 bilingual SVGs, no clipped text.');
  } finally {
    save();
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
