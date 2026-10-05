import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import sitemap from '../src/app/sitemap';
import { parse, type DefaultTreeAdapterMap } from 'parse5';
import { site } from '../src/config/site';

function* htmlFiles(directory: string): Generator<string> {
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    const filename = path.join(directory, entry.name);
    if (entry.isDirectory()) yield* htmlFiles(filename);
    else if (entry.name.endsWith('.html')) yield filename;
  }
}

const missing = new Set<string>();
const checked = new Map<string, boolean>();
let pages = 0;
for (const filename of htmlFiles('out')) {
  pages++;
  const base = new URL(path.relative('out', filename), `${site.url}/`);
  const visit = (node: DefaultTreeAdapterMap['node']) => {
    if ('attrs' in node) for (const attribute of node.attrs) {
      if (!['href', 'src', 'poster'].includes(attribute.name)) continue;
      const url = new URL(attribute.value, base);
      if (url.origin !== new URL(site.url).origin) continue;
      const target = path.resolve('out', `.${decodeURIComponent(url.pathname)}`);
      if (!checked.has(target)) checked.set(target, fs.existsSync(target) && (fs.statSync(target).isFile() || fs.existsSync(path.join(target, 'index.html'))));
      if (!checked.get(target)) missing.add(`${url.pathname} ← ${filename}`);
    }
    if ('childNodes' in node) node.childNodes.forEach(visit);
  };
  visit(parse(fs.readFileSync(filename, 'utf8')));
}
assert.equal(missing.size, 0, `Broken internal links/assets:\n${[...missing].join('\n')}`);
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
console.log(`Export checks passed: ${pages} HTML pages, internal links/assets, sitemap routes, and canonicals.`);
