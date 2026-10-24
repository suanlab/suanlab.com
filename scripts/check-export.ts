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
const identifiers = new Map<string, Set<string>>();
const fragments: { target: string; fragment: string; source: string }[] = [];
const slideCounts = new Map<string, number>();
let pages = 0;
for (const filename of htmlFiles('out')) {
  pages++;
  const absoluteFilename = path.resolve(filename);
  const ids = new Set<string>();
  let mainCount = 0;
  identifiers.set(absoluteFilename, ids);
  const base = new URL(path.relative('out', filename), `${site.url}/`);
  const visit = (node: DefaultTreeAdapterMap['node'], insideMain = false) => {
    const isMain = 'tagName' in node && node.tagName === 'main';
    if (isMain) {
      mainCount++;
      assert.ok(!insideMain, `${filename}: nested main landmark`);
    }
    if ('attrs' in node) {
      if (node.attrs.some(a => a.name === 'class' && a.value.split(' ').includes('deck-slide'))) slideCounts.set(absoluteFilename, (slideCounts.get(absoluteFilename) || 0) + 1);
    }
    if ('attrs' in node) for (const attribute of node.attrs) {
      if (attribute.name === 'id' || (node.tagName === 'a' && attribute.name === 'name')) ids.add(attribute.value);
      if (!['href', 'src', 'poster'].includes(attribute.name)) continue;
      assert.ok(attribute.name !== 'href' || attribute.value.trim() !== '#', `${filename}: placeholder link`);
      const url = new URL(attribute.value, base);
      if (url.origin !== new URL(site.url).origin) continue;
      const target = path.resolve('out', `.${decodeURIComponent(url.pathname)}`);
      if (!checked.has(target)) checked.set(target, fs.existsSync(target) && (fs.statSync(target).isFile() || fs.existsSync(path.join(target, 'index.html'))));
      if (!checked.get(target)) missing.add(`${url.pathname} ← ${filename}`);
      else if (attribute.name === 'href' && url.hash && !url.hash.startsWith('#:~:text=')) {
        const htmlTarget = fs.statSync(target).isDirectory() ? path.join(target, 'index.html') : target;
        if (htmlTarget.endsWith('.html')) fragments.push({ target: htmlTarget, fragment: decodeURIComponent(url.hash.slice(1)), source: filename });
      }
    }
    if ('childNodes' in node) node.childNodes.forEach(child => visit(child, insideMain || isMain));
  };
  visit(parse(fs.readFileSync(filename, 'utf8')));
  assert.equal(mainCount, 1, `${filename}: expected one main landmark`);
}
for (const { target, fragment, source } of fragments) {
  const slides = slideCounts.get(target);
  const slideLink = slides && /^\d+$/.test(fragment) && Number(fragment) >= 1 && Number(fragment) <= slides;
  if (!slideLink && fragment.toLowerCase() !== 'top' && !identifiers.get(target)?.has(fragment)) missing.add(`${path.relative('out', target)}#${fragment} ← ${source}`);
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
const feed = fs.readFileSync('out/feed.xml', 'utf8');
assert.ok(feed.includes(`${site.contact.emails[0]} (이수안)`), 'RSS contact must use site configuration');
for (const match of feed.matchAll(/<enclosure url="([^"]+)" type="([^"]+)" length="(\d+)"\s*\/>/g)) {
  const url = new URL(match[1].replace(/&amp;/g, '&'));
  assert.equal(url.origin, new URL(site.url).origin);
  const file = path.join('out', decodeURIComponent(url.pathname));
  assert.equal(fs.statSync(file).size, Number(match[3]), `RSS enclosure size: ${url.pathname}`);
  const bytes = fs.readFileSync(file);
  if (match[2] === 'image/jpeg') assert.ok(bytes[0] === 0xff && bytes[1] === 0xd8, `RSS JPEG encoding: ${url.pathname}`);
  else if (match[2] === 'image/png') assert.ok(bytes.subarray(1, 4).equals(Buffer.from('PNG')), `RSS PNG encoding: ${url.pathname}`);
}
console.log(`Export checks passed: ${pages} HTML pages, internal links/assets/fragments, sitemap routes, and canonicals.`);
