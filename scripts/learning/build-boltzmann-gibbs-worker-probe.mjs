/* Reproducible isolated-worker browser harness; does not modify Next/out. */
import { build } from 'esbuild';
import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
const parent = '.runtime/learning/boltzmann-gibbs-worker', root = `${parent}/browser`;
await fs.mkdir(root, { recursive: true });
await build({ entryPoints: ['scripts/learning/boltzmann-gibbs-worker-probe.tsx'], bundle: true, format: 'esm', platform: 'browser', target: 'es2022', outfile: `${root}/probe.js` });
await build({ entryPoints: ['src/lib/learning-boltzmann-gibbs.worker.ts'], bundle: true, format: 'esm', platform: 'browser', target: 'es2022', outfile: `${root}/worker.js` });
await fs.writeFile(`${root}/index.html`, '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Isolated worker verification</title></head><body><div id="root"></div><script type="module" src="./probe.js"></script></body></html>');
const main = await fs.readFile(`${root}/probe.js`, 'utf8'), worker = await fs.readFile(`${root}/worker.js`, 'utf8');
const markers = ['Expected 64..8192 pi precision bits', 'All derivatives hold reference weights and energies fixed.', 'Left grid R/'];
for (const marker of markers) if (main.includes(marker) || !worker.includes(marker)) throw new Error(`Incorrect isolated bundle boundary: ${marker}`);
const result = { scope: 'isolated esbuild probe, not Next static export', markers, mainContainsNumericalMarkers: false, workerContainsAllNumericalMarkers: true,
  sha256: { main: createHash('sha256').update(main).digest('hex'), worker: createHash('sha256').update(worker).digest('hex') } };
await fs.writeFile(`${parent}/bundle-boundary.json`, JSON.stringify(result, null, 2)+'\n');
console.log('Isolated probe rebuilt; numerical implementation markers occur only in worker.');
