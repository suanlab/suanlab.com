import { createHash } from 'node:crypto';
import { isCalendarDate } from '../src/lib/content-date';
import { parseProvenance } from '../src/lib/content-provenance';
import { getLectureContentSlugs } from '../src/lib/lecture-content';
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
import { getAllQTEntries, getQTSlugs } from '../src/lib/qt';
import { getBookSlugs } from '../src/lib/books';
import { getPostSlugs } from '../src/lib/blog';
import { mainNavigation } from '../src/data/site-navigation';
import { featuredPublications, featuredProjectIds, projectSummaries, researchLinks, learningPaths } from '../src/data/editorial';

function unique(values: (number | string)[], label: string) {
  assert.equal(new Set(values).size, values.length, `${label}: duplicate identifiers`);
}
unique(publications.map(p => p.id), 'publications');
unique(publications.map(p => [p.type, p.title.trim().toLowerCase(), p.authors, p.date].join('|')), 'publication records');
unique(projects.map(p => p.id), 'projects');
for (const [label, rows] of [['lectures', lectures], ['playlists', playlists], ['research', researchAreas]] as const) {
  unique(rows.map(p => p.slug), label);
  rows.forEach(p => assert.match(p.slug, /^[a-z0-9-]+$/, `${label}: invalid slug`));
}
const blogBodies: string[] = [];
for (const file of fs.readdirSync('content/blog').filter(f => f.endsWith('.md'))) {
  const { data, content } = matter(fs.readFileSync(path.join('content/blog', file), 'utf8'));
  blogBodies.push(createHash('sha256').update(content.trim()).digest('hex'));
  parseProvenance(data.provenance);
  assert.ok(data.title && data.date, `${file}: title and date required`);
  assert.ok(isCalendarDate(data.date), `${file}: invalid date`);
  assert.ok(Array.isArray(data.tags), `${file}: tags must be an array`);
  for (const key of ['title', 'excerpt', 'category']) assert.ok(typeof data[key] === 'string' && data[key].trim(), `${file}: ${key} must be a non-empty string`);
  assert.ok(data.tags.every((tag: unknown) => typeof tag === 'string' && tag.trim()), `${file}: tags must contain non-empty strings`);
}
unique(blogBodies, 'blog bodies');
for (const entry of getAllQTEntries()) assert.ok(isCalendarDate(entry.date), `Invalid QT date: ${entry.slug}`);
for (const dir of ['books', 'lectures']) for (const file of fs.readdirSync(`content/${dir}`).filter(f => f.endsWith('.md'))) {
  const { data } = matter(fs.readFileSync(`content/${dir}/${file}`, 'utf8'));
  assert.ok(data.title && isCalendarDate(data.date), `Invalid ${dir} title/date: ${file}`);
}
for (const slug of getLectureContentSlugs()) assert.ok(lectures.some(lecture => lecture.slug === slug), `Unknown lecture content: ${slug}`);
const urls = sitemap().map(entry => new URL(entry.url).pathname.replace(/\/$/, '') || '/');
unique(urls, 'sitemap');
const routes = new Set(urls);
for (const id of Object.keys(projectSummaries)) assert.ok(projects.some(p => p.id === Number(id)), `Unknown project summary: ${id}`);
unique(featuredPublications.map(p => p.id), 'featured publications');
for (const selection of featuredPublications) assert.ok(publications.some(p => p.id === selection.id), `Missing featured publication: ${selection.id}`);
for (const id of featuredProjectIds) assert.ok(projects.some(p => p.id === id && !p.completed), `Featured project must be active: ${id}`);
assert.deepEqual(Object.keys(researchLinks).sort(), researchAreas.map(p => p.slug).sort(), 'Every research area needs curated links');
for (const [slug, links] of Object.entries(researchLinks)) {
  for (const id of links.publications) assert.ok(publications.some(p => p.id === id), `${slug}: missing publication ${id}`);
  for (const id of links.projects) assert.ok(projects.some(p => p.id === id), `${slug}: missing project ${id}`);
  for (const course of links.lectures) assert.ok(lectures.some(p => p.slug === course), `${slug}: missing lecture ${course}`);
}
for (const learningPath of learningPaths) for (const step of learningPath.steps) assert.ok(routes.has(step.href.replace(/\/$/, '')), `Missing learning step: ${step.href}`);
for (const [prefix, slugs] of [['qt', getQTSlugs()], ['book/online', getBookSlugs()], ['blog', getPostSlugs()]] as const) {
  for (const slug of slugs) assert.ok(routes.has(`/${prefix}/${slug.replace(/\.md$/, '')}`), `Missing sitemap route: ${prefix}/${slug}`);
}
for (const item of mainNavigation) {
  assert.ok(routes.has(item.href), `Missing navigation route: ${item.href}`);
  for (const child of 'children' in item ? item.children || [] : []) assert.ok(routes.has(child.href), `Missing child route: ${child.href}`);
}
for (const lecture of lectures) for (const link of lecture.relatedYoutube || []) assert.ok(routes.has(link), `Missing related video: ${link}`);
console.log(`Content validation passed: ${urls.length} URLs, ${publications.length} publications, ${projects.length} projects.`);
