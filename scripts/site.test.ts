import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildLectureSlides, slideIndex } from '../src/lib/presentation';
import { searchItems } from '../src/lib/search';
import sitemap from '../src/app/sitemap';
import { getQTSlugs } from '../src/lib/qt';
import { conferenceCalendar, deadlineInstant, deadlineKind, submissionStatus, upcomingDeadlines } from '../src/lib/conference-deadlines';
import { conferences } from '../src/data/conferences';
import { defaultPromptValues, normalizePromptValues, encodePromptShare, decodePromptShare, detectPromptVariables, substitutePromptVariables } from '../src/lib/prompt-state';
import { promptBuilders, promptSnippets } from '../src/data/prompts';
import { promptWorkflows } from '../src/data/prompts/workflows';
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

test('calendar date validation rejects rollover and incomplete dates', async () => {
  const { isCalendarDate } = await import('../src/lib/content-date');
  assert.equal(isCalendarDate('2024-02-29'), true);
  for (const value of ['2026-02-29', '2026-02-30', '2026-13-01', '2026-1-2', undefined]) assert.equal(isCalendarDate(value), false);
});

test('date-only legacy QT files retain their date, title, and stable slug', async () => {
  const { getQTBySlug } = await import('../src/lib/qt');
  const entry = getQTBySlug('2015-02-01');
  assert.equal(entry?.date, '2015-02-01');
  assert.equal(entry?.slug, '2015-02-01');
  assert.match(entry?.title || '', /잠언/);
  assert.notEqual(entry?.title, '---');
});

test('conference clocks distinguish AoE, Pacific daylight saving and unpublished times', () => {
  const schedule = { deadlines: [], timezone: 'AoE' };
  assert.equal(deadlineInstant({ type: 'Full Paper', date: '2026-10-06' }, schedule)?.toISOString(), '2026-10-07T11:59:59.000Z');
  const pacific = { deadlines: [], timezone: 'America/Los_Angeles', deadlineTime: '17:00:00' };
  assert.equal(deadlineInstant({ type: 'Full Paper', date: '2026-10-07' }, pacific)?.toISOString(), '2026-10-08T00:00:00.000Z');
  assert.equal(deadlineInstant({ type: 'Full Paper', date: '2026-11-11' }, pacific)?.toISOString(), '2026-11-12T01:00:00.000Z');
  assert.equal(deadlineInstant({ type: 'Conference', date: '2026-10-07' }, pacific), null);
  assert.equal(deadlineInstant({ type: 'Full Paper', date: '2026-10-07' }, { deadlines: [] }), null);
  assert.equal(deadlineInstant({ type: 'Tutorial Submission', date: '2026-10-07', timezone: null }, pacific), null);
  for (const date of [null, '2026-02-30']) assert.equal(deadlineInstant({ type: 'Full Paper', date }, schedule), null);
  assert.equal(deadlineInstant({ type: 'Full Paper', date: '2026-10-07', status: 'tentative' }, schedule), null);
});

test('submission status never counts upcoming reviews or conferences as an open paper deadline', () => {
  const now = new Date('2026-10-06T12:00:00Z');
  const conf = { timezone: 'AoE', deadlines: [{ type: 'Full Paper', date: '2026-09-01' }, { type: 'Notification', date: '2026-11-01' }, { type: 'Conference', date: '2027-01-01' }] };
  assert.equal(submissionStatus(conf, now), 'closed');
  assert.equal(upcomingDeadlines(conf, now, 'submission').length, 0);
  assert.equal(upcomingDeadlines(conf, now, 'schedule').length, 2);
  assert.equal(submissionStatus({ ...conf, deadlines: [...conf.deadlines, { type: 'Full Paper (Cycle 2)', date: null }] }, now), 'tba');
  assert.equal(submissionStatus({ timezone: 'AoE', deadlines: [{ type: 'Full Paper', date: '2026-10-06' }] }, now), 'upcoming');
  assert.equal(deadlineKind({ type: 'Invited Final Paper', date: '2027-04-16', kind: 'milestone' }), 'milestone');
});

test('calendar export excludes tentative dates and preserves UTF-8, escaping and all-day ranges', () => {
  const now = new Date('2026-10-06T00:00:00Z');
  const conf = { id: 'test-2027', name: '학회'.repeat(25), year: 2027, url: 'https://example.org/', location: 'Seoul, Korea; Hall\nA', deadlines: [
    { type: 'Full Paper', date: '2026-10-10' },
    { type: 'TBA Paper', date: null },
    { type: 'Tentative Paper', date: '2026-10-12', status: 'tentative' as const },
    { type: 'Conference', date: '2027-01-01', endDate: '2027-01-03' },
  ] };
  const ics = conferenceCalendar([conf], now, 'schedule');
  const unfolded = ics.replace(/\r\n /g, '');
  assert.equal((ics.match(/BEGIN:VEVENT/g) ?? []).length, 2);
  assert.match(unfolded, /DTSTART;VALUE=DATE:20261010/);
  assert.match(unfolded, /DTEND;VALUE=DATE:20270104/);
  assert.ok(unfolded.includes('LOCATION:Seoul\\, Korea\\; Hall\\nA'));
  assert.ok(!unfolded.includes('Tentative Paper'));
  assert.ok(!unfolded.includes('\uFFFD'));
  for (const line of ics.split('\r\n')) assert.ok(Buffer.byteLength(line, 'utf8') <= 75);
  const timed = conferenceCalendar([{ ...conf, timezone: 'AoE' }], now, 'submission');
  assert.match(timed, /DTSTART:20261011T115959Z/);
  assert.equal((timed.match(/BEGIN:VEVENT/g) ?? []).length, 1);
});

test('published conference records have valid dates and explicit uncertainty for previously invented placeholders', async () => {
  const { isCalendarDate } = await import('../src/lib/content-date');
  assert.equal(new Set(conferences.map(c => c.id)).size, conferences.length);
  for (const conf of conferences) {
    if (conf.verifiedAt) { assert.ok(isCalendarDate(conf.verifiedAt)); assert.ok(conf.sourceUrl?.startsWith('https://')); }
    for (const d of conf.deadlines) {
      assert.ok(d.date === null || isCalendarDate(d.date), `${conf.id}: ${d.type}`);
      if (d.endDate) { assert.ok(isCalendarDate(d.endDate)); assert.ok(d.date && d.endDate >= d.date); }
    }
  }
  for (const id of ['icml-2027', 'ecmlpkdd-2027']) assert.equal(conferences.find(c => c.id === id)?.deadlines.find(d => d.type === 'Conference')?.date, null);
  assert.equal(conferences.find(c => c.id === 'kdd-2027')?.deadlines.find(d => d.type === 'Full Paper (Cycle 2)')?.date, null);
  const vldb = conferences.find(c => c.id === 'vldb-2027')!;
  const papers = vldb.deadlines.filter(d => deadlineKind(d) === 'submission');
  assert.equal(papers.length, 6);
  assert.ok(papers.every(d => d.date && d.date <= '2027-03-01'));
});

test('prompt share links normalize untrusted fields and support Korean placeholders', () => {
  const fields = [{ id: 'topic', type: 'textarea' }, { id: 'lang', type: 'select', options: [{ value: 'en' }] }, { id: 'tools', type: 'multiselect', options: [{ value: 'RAG' }] }];
  assert.deepEqual(defaultPromptValues(fields, 'en'), { topic: '', lang: 'en', tools: [] });
  assert.equal(defaultPromptValues([{ id: 'role', type: 'select', options: [{ value: 'researcher' }] }], 'ko').role, 'researcher');
  assert.deepEqual(normalizePromptValues(fields, { topic: ['wrong'], lang: 'invalid', tools: ['RAG', 'RAG', 3, 'invalid'], unknown: 'ignored' }), { tools: ['RAG'] });
  assert.deepEqual(normalizePromptValues(fields, { tools: 'RAG' }), {});
  const values = { topic: '초지능 연구', tools: ['RAG'], lang: 'en' };
  const decoded = decodePromptShare(`#${encodePromptShare('rag-system', values)}`)!;
  assert.equal(decoded.b, 'rag-system');
  assert.deepEqual(normalizePromptValues(fields, decoded.v), values);
  for (const hash of ['#bad', '#%', btoa('{"b":"rag-system","v":[]}')]) assert.equal(decodePromptShare(hash), null);
  assert.deepEqual(detectPromptVariables('{{연구_주제}} {{context}} {{연구_주제}}'), ['연구_주제', 'context']);
  assert.equal(substitutePromptVariables('{{연구_주제}} {{context}}', { 연구_주제: 'RAG', context: '' }), 'RAG {{context}}');
});

test('research workflows resolve builders and new prompts produce both language outputs', () => {
  const builderIds = new Set(promptBuilders.map(b => b.id));
  assert.equal(builderIds.size, promptBuilders.length);
  assert.equal(new Set(promptSnippets.map(s => s.id)).size, promptSnippets.length);
  for (const wf of promptWorkflows) for (const step of wf.steps) assert.ok(builderIds.has(step.builderId));
  for (const id of ['rag-system', 'agent-system', 'llm-evaluation', 'multimodal-analysis', 'text-to-sql', 'structured-extraction', 'llm-operations', 'prompt-evaluation']) {
    const b = promptBuilders.find(b => b.id === id)!;
    const input = Object.fromEntries(b.fields.map(f => [f.id, f.type === 'multiselect' ? [] : 'test context']));
    assert.match(b.generate({ ...input, lang: 'ko' }), /근거와 검증/);
    assert.match(b.generate({ ...input, lang: 'en' }), /Evidence and verification/);
  }
  const writing = promptBuilders.find(b => b.id === 'writing')!.generate({ venue: 'neurips' });
  assert.ok(!writing.includes('neurips_2024'));
});
