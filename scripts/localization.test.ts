import { getAllQTEntries } from '../src/lib/qt';
import test from 'node:test';
import assert from 'node:assert/strict';
import { createElement, type ReactElement } from 'react';
import { englishText } from '../src/data/i18n/translate';
import { localizeNodes, titleSourceForPath } from '../src/components/localized-content';
import { localizedMarkup } from '../src/lib/localized-markup';
import { searchItems } from '../src/lib/search';
import { projects } from '../src/data/projects';
import { researchAreas } from '../src/data/research';
import { courses } from '../src/data/courses';
import { projectEnglish } from '../src/data/i18n/projects';
import { researchEnglish } from '../src/data/i18n/research';
import { courseEnglish } from '../src/data/i18n/courses';

const korean = /[가-힣]/;
test('English display formats retain the canonical source', () => {
  for (const [source, expected] of [
    ['검색 / Search', 'Search'], ['2026년 2학기', 'Fall 2026'],
    ['2026년 10월 25일', 'October 25, 2026'], ['1.5억원', 'KRW 150,000,000'],
    ['창세기 1장 1~5절', 'Genesis 1:1–5'], ['요3:30', 'John 3:30'],
    ['한국 → 미국', 'South Korea → United States'],
  ]) assert.equal(englishText(source, { 한국: 'South Korea', 미국: 'United States' }), expected);
  assert.equal(englishText('unknown-id'), 'unknown-id');
});

test('React localization preserves identity, state boundaries, links and code', () => {
  const click = () => {};
  const source = createElement('button', { key: 'canonical', id: '검색', title: '검색', onClick: click, value: '검색' }, '검색');
  const result = localizeNodes(source, { 검색: 'Search' }) as ReactElement;
  assert.ok(!Array.isArray(result));
  assert.equal(result.type, source.type);
  assert.equal(result.key, source.key);
  assert.equal(result.props.onClick, click);
  assert.equal(result.props.id, '검색');
  assert.equal(result.props.value, '검색');
  assert.equal(result.props.title, 'Search');
  assert.equal(result.props.children, 'Search');
  const link = localizeNodes(createElement('a', { href: '/qt/?book=요한복음' }, '검색'), { 검색: 'Search' }) as ReactElement;
  assert.equal(link.props.href, '/qt/?book=요한복음');
  for (const original of [createElement('span', { lang: 'ko' }, '원문'), createElement('code', {}, '검색')]) {
    assert.equal(localizeNodes(original, { 검색: 'Search' }), original);
  }
});

test('HTML localization preserves IDs, URLs, code, math and source-language regions', () => {
  const html = '<h2 id="원문" title="검색">검색</h2><a href="/검색/">검색</a><pre><code>검색</code></pre><math><mi>검색</mi></math><p lang="ko">검색</p>';
  const result = localizedMarkup(html, text => englishText(text, { 검색: 'Search' }));
  assert.match(result, /id="원문" title="Search">Search/);
  assert.match(result, /href="\/검색\/">Search/);
  assert.match(result, /<code>검색<\/code>/);
  assert.match(result, /<mi>검색<\/mi>/);
  assert.match(result, /lang="ko">검색/);
});

test('Curated research and project descriptions have complete English translations', () => {
  for (const project of projects) for (const text of [project.title, ...project.items]) {
    assert.ok(!korean.test(englishText(text, projectEnglish)), `Project ${project.id}: ${text}`);
  }
  for (const area of researchAreas) for (const text of area.overview.split('\n\n')) {
    assert.ok(!korean.test(englishText(text, researchEnglish)), `Research ${area.slug}: ${text}`);
  }
  for (const course of courses) assert.ok(!korean.test(englishText(course.title, courseEnglish)), course.title);
});

test('Search accepts English and Korean while retaining canonical result destinations', () => {
  const items = [{ type: 'project', title: '워터마킹 연구', description: '데이터 보호', titleEn: 'Watermarking research', descriptionEn: 'Data protection', href: '/project/#project-1' }];
  assert.deepEqual(searchItems(items, 'watermarking protection'), items);
  assert.deepEqual(searchItems(items, '워터마킹'), items);
  assert.equal(items[0].title, '워터마킹 연구');
  assert.equal(searchItems(items, 'watermarking')[0].href, '/project/#project-1');
});

test('Client route metadata keeps its Korean source after an early English title update', () => {
  const previous = { path: '/contact/', original: '문의하기', translated: 'Contact' };
  const early = titleSourceForPath(previous, '/contact/', '개인정보처리방침');
  early.translated = 'Privacy Policy';
  const navigated = titleSourceForPath(early, '/privacy/', 'Privacy Policy');
  assert.equal(navigated.original, '개인정보처리방침');
  assert.equal(navigated.path, '/privacy/');
  const next = titleSourceForPath(navigated, '/terms/', '이용약관');
  assert.equal(next.original, '이용약관');
});

test('Every QT scripture reference has an English display, including legacy source typos', () => {
  for (const entry of getAllQTEntries()) {
    if (entry.bibleReference) assert.ok(!korean.test(englishText(entry.bibleReference)), `${entry.slug}: ${entry.bibleReference}`);
  }
});
