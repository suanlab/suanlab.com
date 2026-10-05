import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import matter from 'gray-matter';
import sitemap from '../src/app/sitemap';
import { publications } from '../src/data/publications';
import { projects } from '../src/data/projects';
import { lectures } from '../src/data/lectures';
import { playlists } from '../src/data/youtube';
import { researchAreas } from '../src/data/research';
import { getQTSlugs } from '../src/lib/qt';
import { getBookSlugs } from '../src/lib/books';
import { getPostSlugs } from '../src/lib/blog';
import { mainNavigation } from '../src/data/site-navigation';

function unique(values: (number | string)[], label: string) {
  assert.equal(new Set(values).size, values.length, `${label}: duplicate identifiers`);
}
unique(publications.map(p => p.id), 'publications');
unique(projects.map(p => p.id), 'projects');
for (const [label, rows] of [['lectures', lectures], ['playlists', playlists], ['research', researchAreas]] as const) {
  unique(rows.map(p => p.slug), label);
  rows.forEach(p => assert.match(p.slug, /^[a-z0-9-]+$/, `${label}: invalid slug`));
}
for (const file of fs.readdirSync('content/blog').filter(f => f.endsWith('.md'))) {
  const { data } = matter(fs.readFileSync(path.join('content/blog', file), 'utf8'));
  assert.ok(data.title && data.date, `${file}: title and date required`);
  assert.ok(!Number.isNaN(Date.parse(data.date)), `${file}: invalid date`);
  assert.ok(Array.isArray(data.tags), `${file}: tags must be an array`);
}
const urls = sitemap().map(entry => new URL(entry.url).pathname.replace(/\/$/, '') || '/');
unique(urls, 'sitemap');
const routes = new Set(urls);
for (const [prefix, slugs] of [['qt', getQTSlugs()], ['book/online', getBookSlugs()], ['blog', getPostSlugs()]] as const) {
  for (const slug of slugs) assert.ok(routes.has(`/${prefix}/${slug.replace(/\.md$/, '')}`), `Missing sitemap route: ${prefix}/${slug}`);
}
for (const item of mainNavigation) {
  assert.ok(routes.has(item.href), `Missing navigation route: ${item.href}`);
  for (const child of 'children' in item ? item.children || [] : []) assert.ok(routes.has(child.href), `Missing child route: ${child.href}`);
}
for (const lecture of lectures) for (const link of lecture.relatedYoutube || []) assert.ok(routes.has(link), `Missing related video: ${link}`);
console.log(`Content validation passed: ${urls.length} URLs, ${publications.length} publications, ${projects.length} projects.`);
