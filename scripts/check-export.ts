import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import sitemap from '../src/app/sitemap';
for (const entry of sitemap()) {
  const route = decodeURIComponent(new URL(entry.url).pathname);
  const filename = path.join('out', route, 'index.html');
  assert.ok(fs.existsSync(filename), `Missing exported page: ${route}`);
  const html = fs.readFileSync(filename, 'utf8');
  const canonical = html.match(/<link rel="canonical" href="([^"]+)"/);
  assert.equal(canonical?.[1].replace(/\/$/, ''), entry.url.replace(/\/$/, ''), `Incorrect canonical: ${route}`);
}
for (const file of ['out/index.html', 'out/lecture/ai/present/index.html', 'out/search/index.html']) {
  const html = fs.readFileSync(file, 'utf8');
  assert.ok(html.includes('<h1'), `${file}: missing heading`);
}
console.log('Export checks passed: every sitemap URL has a generated page.');
