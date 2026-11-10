/* Reuse exported styles/fonts; never replace the running site's assets. */
import { build } from 'esbuild';
import fs from 'node:fs/promises';
import { parse } from 'parse5';
import { createHash } from 'node:crypto';
const parent = '.runtime/learning/boltzmann-gibbs-teaching', root = `${parent}/browser`;
await fs.mkdir(root, { recursive: true });
const tree = parse(await fs.readFile('out/book/interactive/index.html', 'utf8'));
const styles = []; let htmlClass = '', bodyClass = '';
function visit(node) {
  const attrs = Object.fromEntries((node.attrs ?? []).map(a => [a.name, a.value]));
  if (node.tagName === 'link' && attrs.rel === 'stylesheet') styles.push(attrs.href);
  if (node.tagName === 'html') htmlClass = attrs.class ?? '';
  if (node.tagName === 'body') bodyClass = attrs.class ?? '';
  for (const child of node.childNodes ?? []) visit(child);
}
visit(tree);
if (!styles.length) throw Error('Missing actual exported styles');
for (const folder of ['css', 'media']) await fs.cp(`out/_next/static/${folder}`, `${root}/_next/static/${folder}`, { recursive: true });
await fs.cp('out/assets/fonts', `${root}/assets/fonts`, { recursive: true });
await build({ entryPoints: ['scripts/learning/boltzmann-gibbs-teaching-probe.tsx'], bundle: true, format: 'esm', platform: 'browser', target: 'es2022', jsx: 'automatic', outfile: `${root}/probe.js` });
await build({ entryPoints: ['src/lib/learning-boltzmann-gibbs.worker.ts'], bundle: true, format: 'esm', platform: 'browser', target: 'es2022', outfile: `${root}/worker.js` });
await fs.writeFile(`${root}/index.html`, `<!doctype html><html lang="en" class="${htmlClass}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Teaching component verification</title>${styles.map(href => `<link rel="stylesheet" href="${href}">`).join('')}</head><body class="${bodyClass}"><div id="root"></div><script type="module" src="./probe.js"></script></body></html>`);
const markers = ['Expected 64..8192 pi precision bits', 'All derivatives hold reference weights and energies fixed.', 'Left grid R/'];
const main = await fs.readFile(`${root}/probe.js`, 'utf8'), worker = await fs.readFile(`${root}/worker.js`, 'utf8');
for (const marker of markers) if (main.includes(marker) || !worker.includes(marker)) throw Error(`Calculation bundle boundary: ${marker}`);
await fs.writeFile(`${parent}/bundle-boundary.json`, JSON.stringify({ scope: 'isolated teaching components, not Next integration', styles, numericalMarkersOnlyInWorker: true, sha256: { main: createHash('sha256').update(main).digest('hex'), worker: createHash('sha256').update(worker).digest('hex') } }, null, 2) + '\n');
console.log('Teaching probe built with exported CSS/fonts and separate real worker.');
