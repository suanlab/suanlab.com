import { test } from 'node:test';
import assert from 'node:assert/strict';
import matter from 'gray-matter';
import { formatAsMarkdown } from './post-format';
import { parseProvenance, publicSourceUrl, type ContentProvenance } from '../../src/lib/content-provenance';

test('source and review metadata survive YAML serialization without trusting generated citations', () => {
  const provenance: ContentProvenance = {
    method: 'ai-assisted', generatedAt: '2026-10-05T00:00:00Z', review: { status: 'pending' },
    source: { kind: 'arxiv', title: 'Real "source" \\ title', authors: ['Author'], identifier: '2312.00752', url: 'https://arxiv.org/abs/2312.00752' },
  };
  const markdown = formatAsMarkdown({ slug: 'test', title: 'Title: "quoted"\nsecond line', date: '2026-10-05', excerpt: 'Summary \\ path', category: 'Paper Review', tags: ['test'], content: 'Generated prose with an unrelated citation.', provenance });
  const parsed = matter(markdown);
  assert.equal(parsed.data.title, 'Title: "quoted"\nsecond line');
  assert.deepEqual(parseProvenance(parsed.data.provenance), provenance);
  assert.equal(parseProvenance(undefined), undefined);
  assert.throws(() => parseProvenance({ ...provenance, review: { status: 'reviewed' } }), /reviewer/);
  assert.throws(() => parseProvenance({ ...provenance, source: { ...provenance.source, url: 'https://arxiv.org/abs/0000.00000' } }), /matching/);
  assert.throws(() => parseProvenance({ ...provenance, source: { kind: 'pdf', title: 'Upload' } }), /hash/);
  assert.doesNotThrow(() => parseProvenance({ ...provenance, review: { status: 'reviewed', reviewer: 'Editor', reviewedAt: '2026-10-06' } }));
});

test('reader notice distinguishes missing history, pending drafts, and named review', async () => {
  const { createElement } = await import('react');
  const { renderToStaticMarkup } = await import('react-dom/server');
  const { LanguageProvider } = await import('../../src/components/language-provider');
  const { default: Notice } = await import('../../src/components/content-provenance-notice');
  const render = (provenance?: ContentProvenance) => renderToStaticMarkup(createElement(LanguageProvider, { children: createElement(Notice, { provenance, paperReview: true }) }));
  assert.match(render(), /검토 기록 없음/);
  const pending: ContentProvenance = { method: 'ai-assisted', generatedAt: '2026-10-05', review: { status: 'pending' }, source: { kind: 'topic', title: 'A learning topic' } };
  assert.match(render(pending), /검토 대기/);
  assert.doesNotMatch(render(pending), /검토 완료/);
  const reviewed: ContentProvenance = { ...pending, review: { status: 'reviewed', reviewer: 'Editor', reviewedAt: '2026-10-06' } };
  assert.match(render(reviewed), /검토 완료/); assert.match(render(reviewed), /Editor/);
});

test('PDF access parameters are not published as source links', () => {
  assert.equal(publicSourceUrl('https://example.com/paper.pdf?token=private'), undefined);
  assert.equal(publicSourceUrl('https://user:password@example.com/paper.pdf'), undefined);
  assert.equal(publicSourceUrl('https://example.com/paper.pdf'), 'https://example.com/paper.pdf');
});
