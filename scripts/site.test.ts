import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildLectureSlides, slideIndex } from '../src/lib/presentation';
import { searchItems } from '../src/lib/search';
import sitemap from '../src/app/sitemap';
import { getQTSlugs } from '../src/lib/qt';
test('sitemap uses content slugs, not filenames or playlist IDs', () => {
  const urls = new Set(sitemap().map(p => new URL(p.url).pathname));
  assert.ok(urls.has('/lecture/ai/'));
  assert.ok(urls.has('/lecture/ai/present/'));
  assert.ok(urls.has('/youtube/pp/'));
  assert.ok(!urls.has('/lecture/python-for-data-analysis/'));
  for (const slug of getQTSlugs()) assert.ok(urls.has(`/qt/${slug}/`));
});
test('slide navigation rejects malformed hashes and overview includes every topic', () => {
  const topics = Array.from({ length: 9 }, (_, i) => `Topic ${i}`);
  const slides = buildLectureSlides({ titleKo: 'AI', titleEn: 'AI', descriptionKo: 'Overview', topics });
  assert.equal(slides.length, 5);
  topics.forEach(topic => assert.ok(slides.some(slide => slide.points.includes(topic))));
  assert.equal(slideIndex('#2', 5), 1);
  for (const hash of ['#-1', '#NaN', '#99', '#1.5', '']) assert.equal(slideIndex(hash, 5), 0);
});
test('search matches multiple terms across fields and filters content types', () => {
  const items = [{ type: 'lecture', title: 'AI 연구', description: 'Python', href: '/lecture/ai/' }, { type: 'blog', title: 'AI', description: 'Review', href: '/blog/test/' }];
  assert.equal(searchItems(items, 'ai python').length, 1);
  assert.equal(searchItems(items, 'AI', 'blog').length, 1);
  assert.equal(searchItems(items, ' ').length, 0);
});

test('lecture Markdown keeps code separators, renders math/images, and separates notes', async () => {
  const { parseLectureContent, getLectureContent } = await import('../src/lib/lecture-content');
  const source = '---\ntitle: Test\n---\n# First\n\n```text\n---\n```\n\n<!-- notes: presenter only -->\n\n---\n# Second\n\n$$x^2$$\n\n![Diagram](/assets/test.svg)\n\n<script>alert(1)</script>\n';
  const deck = await parseLectureContent(source);
  assert.equal(deck.slides.length, 2);
  assert.ok(deck.slides[0].contentHtml?.includes('---'));
  assert.equal(deck.slides[0].notes, 'presenter only');
  assert.ok(!deck.slides[0].contentHtml?.includes('presenter only'));
  assert.match(deck.slides[1].contentHtml!, /class="katex"/);
  assert.match(deck.slides[1].contentHtml!, /alt="Diagram"/);
  assert.ok(!deck.slides[1].contentHtml?.includes('<script>'));
  assert.equal(await getLectureContent('../blog'), null);
  const pilot = await getLectureContent('ai');
  assert.equal(pilot?.slides.length, 10);
  assert.ok(pilot?.slides.some(slide => slide.contentHtml?.includes('hljs')));
  assert.ok(pilot?.slides.some(slide => slide.contentHtml?.includes('search-tree.svg')));
});
