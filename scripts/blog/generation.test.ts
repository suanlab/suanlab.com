import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createServer } from 'node:http';
import sharp from 'sharp';
import { chunkText, parsePdfFromBuffer, parsePdfFromUrl } from '../../src/lib/pdf/parser';
import { downloadBytes, HttpError } from '../../src/lib/http';
import { parseGeneratedContent, generateWithGemini } from '../../src/lib/ai/claude';
import { saveImageBuffer } from '../../src/lib/ai/image';
import { saveGeneratedPost, uniqueGenerationSlug } from './save-post';
import type { GeneratedPost } from '../../src/lib/ai/types';

test('PDF chunks remain bounded without punctuation and preserve Unicode text', () => {
  for (const text of ['x'.repeat(1001), '가나다라'.repeat(200), 'A😀B'.repeat(150), 'First sentence.\n\n' + 'word '.repeat(150)]) {
    const chunks = chunkText(text, 100);
    assert.ok(chunks.every(chunk => chunk.length > 0 && chunk.length <= 100));
    assert.equal(chunks.join('').replace(/\s/g, ''), text.replace(/\s/g, ''));
    assert.ok(chunks.every(chunk => !/[\uD800-\uDBFF]$/.test(chunk) && !/^[\uDC00-\uDFFF]/.test(chunk)));
  }
  assert.deepEqual(chunkText('   ', 10), []);
  for (const size of [0, -1, 1, 1.5, NaN, Infinity]) assert.throws(() => chunkText('text', size), /Chunk size/);
});

test('generation rejects empty responses and never invents undefined excerpts', () => {
  for (const response of ['', ' ', '# Title', '```markdown\n# Title\n```']) assert.throws(() => parseGeneratedContent(response), /no article body/);
  const article = parseGeneratedContent('```markdown\n# Title\n\nA useful paragraph.\n```');
  assert.equal(article.excerpt, 'A useful paragraph.');
  assert.equal(article.content, '# Title\n\nA useful paragraph.');
  assert.equal(parseGeneratedContent('# Title\n\n' + 'a'.repeat(201)).excerpt, 'a'.repeat(200) + '...');
});

test('Gemini keeps credentials out of URLs and rejects reasoning-only or truncated responses', async context => {
  const originalKey = process.env.GEMINI_API_KEY;
  process.env.GEMINI_API_KEY = 'test-key-with-no-access';
  let response = { candidates: [{ content: { parts: [{ text: 'Private reasoning', thought: true }, { text: '# Title\n\nPublic article.', thought: false }] }, finishReason: 'STOP' }] };
  context.mock.method(globalThis, 'fetch', async (url: string, options: RequestInit) => {
    assert.equal(new URL(url).search, '');
    assert.equal(new Headers(options.headers).get('x-goog-api-key'), 'test-key-with-no-access');
    return new Response(JSON.stringify(response), { headers: { 'Content-Type': 'application/json' } });
  });
  try {
    assert.equal(await generateWithGemini('Test', { model: 'test-model' }), '# Title\n\nPublic article.');
    response = { candidates: [{ content: { parts: [{ text: 'Private reasoning', thought: true }] }, finishReason: 'STOP' }] };
    await assert.rejects(generateWithGemini('Test', { model: 'test-model' }), /No content/);
    response = { candidates: [{ content: { parts: [{ text: 'Partial article', thought: false }] }, finishReason: 'MAX_TOKENS' }] };
    await assert.rejects(generateWithGemini('Test', { model: 'test-model' }), /truncated/);
  } finally {
    if (originalKey === undefined) delete process.env.GEMINI_API_KEY; else process.env.GEMINI_API_KEY = originalKey;
  }
});

test('HTTP downloads enforce status, declared/streamed size, whole-body timeout and cancellation', async () => {
  const server = createServer((req, res) => {
    if (req.url === '/missing') { res.writeHead(404); res.end('private download URL'); }
    else if (req.url === '/declared') { res.writeHead(200, { 'Content-Length': '1000' }); res.end('large'); }
    else if (req.url === '/streamed') { res.writeHead(200); res.write('12345678'); res.end('12345678'); }
    else if (req.url === '/slow') { res.writeHead(200); res.write('prefix'); }
    else res.end('not a PDF');
  });
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  assert.ok(address && typeof address === 'object');
  const base = `http://127.0.0.1:${address.port}`;
  try {
    assert.equal((await downloadBytes(base)).toString(), 'not a PDF');
    await assert.rejects(downloadBytes(`${base}/missing`), error => error instanceof HttpError && error.status === 404 && !error.message.includes('private'));
    for (const route of ['/declared', '/streamed']) await assert.rejects(downloadBytes(base + route, { maxBytes: 12 }), /size limit/);
    await assert.rejects(downloadBytes(`${base}/slow`, { timeoutMs: 40 }), /timed out/);
    await assert.rejects(downloadBytes(base, { signal: AbortSignal.abort(new Error('Cancelled')) }), /Cancelled/);
    await assert.rejects(parsePdfFromUrl(base), /not a PDF/);
    await assert.rejects(parsePdfFromBuffer(Buffer.from('<html>Error</html>')), /not a PDF/);
  } finally {
    server.closeAllConnections();
    await new Promise<void>(resolve => server.close(() => resolve()));
  }
});

test('concurrent publication saves refuse collisions and traversal without overwriting content', async () => {
  const cwd = await fs.mkdtemp(path.join(os.tmpdir(), 'suanlab-save-'));
  const post: GeneratedPost = { slug: '20261024-post', title: 'Title', date: '2026-10-24', excerpt: 'Summary', category: 'General', tags: [], content: 'First article' };
  try {
    const outcomes = await Promise.allSettled([saveGeneratedPost(post, undefined, cwd), saveGeneratedPost({ ...post, content: 'Second article' }, undefined, cwd)]);
    assert.equal(outcomes.filter(result => result.status === 'fulfilled').length, 1);
    assert.equal(outcomes.filter(result => result.status === 'rejected').length, 1);
    const saved = await fs.readFile(path.join(cwd, 'content/blog/20261024-post.md'), 'utf8');
    for (const slug of ['../escape', '/tmp/escape', 'bad/name', '']) await assert.rejects(saveGeneratedPost({ ...post, slug }, undefined, cwd), /slug/);
    await assert.rejects(saveGeneratedPost({ ...post, content: 'Replacement' }, undefined, cwd), { code: 'EEXIST' });
    assert.equal(await fs.readFile(path.join(cwd, 'content/blog/20261024-post.md'), 'utf8'), saved);
    const slugs = Array.from({ length: 100 }, () => uniqueGenerationSlug('20261024-paper-'));
    assert.equal(new Set(slugs).size, slugs.length);
    assert.ok(slugs.every(slug => /^20261024-paper-[a-f0-9]{8}$/.test(slug)));
  } finally { await fs.rm(cwd, { recursive: true, force: true }); }
});

test('image bytes match their advertised format and existing assets cannot be replaced', async () => {
  const cwd = await fs.mkdtemp(path.join(os.tmpdir(), 'suanlab-image-'));
  try {
    const png = await sharp({ create: { width: 4, height: 4, channels: 3, background: '#abcdef' } }).png().toBuffer();
    const url = await saveImageBuffer(png, 'example', 'image/jpeg', cwd);
    assert.equal(url, '/assets/images/blog/example.jpg');
    assert.equal((await sharp(path.join(cwd, 'public', url)).metadata()).format, 'jpeg');
    await assert.rejects(saveImageBuffer(png, 'example', 'image/jpeg', cwd), { code: 'EEXIST' });
    await assert.rejects(saveImageBuffer(png, '../escape', 'image/jpeg', cwd), /filename/);
  } finally { await fs.rm(cwd, { recursive: true, force: true }); }
});
