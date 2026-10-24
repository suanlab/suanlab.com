import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readBlogQuery, updateListQuery } from '../src/lib/list-query';
import { promptHref, readPromptTarget } from '../src/lib/prompt-state';

test('blog deep links retain Unicode filters and reject malformed pages', () => {
  const filters = readBlogQuery(new URLSearchParams('tag=신경망&category=논문리뷰&q=RAG&page=3'));
  assert.deepEqual(filters, { tag: '신경망', category: '논문리뷰', q: 'RAG', page: 3 });
  for (const page of ['', '0', '-1', '1.5', 'Infinity', '2x', '9007199254740992']) {
    assert.equal(readBlogQuery(new URLSearchParams({ page })).page, 1);
  }
});

test('list navigation preserves unrelated parameters and removes cleared filters', () => {
  const original = new URL('https://example.com/qt/?book=요한복음&q=이전&source=search#main-content');
  const next = updateListQuery(original, { q: '하나님의 어린양', book: null, testament: '' });
  assert.equal(next.searchParams.get('q'), '하나님의 어린양');
  assert.equal(next.searchParams.get('source'), 'search');
  assert.equal(next.searchParams.has('book'), false);
  assert.equal(next.searchParams.has('testament'), false);
  assert.equal(next.hash, '#main-content');
  assert.equal(original.searchParams.get('book'), '요한복음');
});

test('prompt search links round-trip specific builder and snippet identities', () => {
  for (const kind of ['builder', 'snippet'] as const) {
    const href = promptHref(kind, 'rag-검색 & 검증');
    assert.deepEqual(readPromptTarget(new URL(href, 'https://example.com').searchParams), { kind, id: 'rag-검색 & 검증' });
  }
  assert.equal(readPromptTarget(new URLSearchParams('builder=&snippet=')), null);
  assert.deepEqual(readPromptTarget(new URLSearchParams('builder=one&snippet=two')), { kind: 'builder', id: 'one' });
});
