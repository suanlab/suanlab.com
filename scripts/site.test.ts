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
