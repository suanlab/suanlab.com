import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import { getLearningContent } from '../src/lib/learning-content';

test('thermometer supplement preserves both original manuscripts byte for byte', () => {
  const originals = {
    ko: 'cbd39803e0920a32e57a44e18331c4e0901034db8569609fb757d75723e6713a',
    en: '58c2b34a66bcbcc3e6ef4c94b171bc7fbd0144e1e2e485462f56e7e775bac21b',
  };
  for (const lang of ['ko', 'en'] as const) {
    const path = `content/learning/physics-math-for-ai/thermodynamics.${lang}.md`;
    const source = readFileSync(path, 'utf8');
    const start = source.search(/^## (Research extension:|연구 확장:)/m);
    assert.ok(start >= 0);
    const end = source.indexOf('\n## ', start + 4) + 1;
    assert.ok(end > start);
    const restored = source.slice(0, start) + source.slice(end);
    assert.equal(createHash('sha256').update(restored).digest('hex'), originals[lang]);
  }
});

test('paired equations, executable table, accessible diagram and reader rendering agree', async () => {
  const pairs = [];
  for (const lang of ['ko', 'en']) {
    const source = readFileSync(`content/learning/physics-math-for-ai/thermodynamics.${lang}.md`, 'utf8');
    if (lang === 'en') assert.doesNotMatch(source, /[가-힣]/);
    assert.doesNotMatch(source, /[\u0000-\u0008\u000b-\u001f]/);
    const start = source.search(/^## (Research extension:|연구 확장:)/m);
    const end = source.indexOf('\n## ', start + 4) + 1;
    const block = source.slice(start, end);
    const ast = unified().use(remarkParse).use(remarkGfm).use(remarkMath).parse(block);
    assert.doesNotMatch(JSON.stringify(ast), /"type":"emphasis"/);
    const math = ast.children.flatMap(n => n.type === 'math' ? [n.value] : []);
    const codes = ast.children.flatMap(n => n.type === 'code' && n.lang === 'python' ? [n.value] : []);
    assert.equal(math.length, 10);
    assert.equal(codes.length, 1);
    const tables = ast.children.filter(n => n.type === 'table');
    assert.equal(tables.length, 1);
    const rows = tables[0].children.slice(1).map(row => row.children.map(cell => cell.children.map(n => 'value' in n ? n.value : '').join('')));
    const output = execFileSync('python3', ['-c', codes[0]], { encoding: 'utf8', timeout: 20000 }).trim().split('\n').map(row => row.split(/\s+/));
    assert.equal(rows.length, 4);
    assert.deepEqual(output, rows);
    assert.equal(ast.children.filter(n => n.type === 'heading' && n.depth === 4).length, 3);
    const images = ast.children.flatMap(n => n.type === 'paragraph' ? n.children.filter(c => c.type === 'image') : []);
    assert.equal(images.length, 1);
    assert.ok(images[0].alt && images[0].alt.length > 40);
    assert.equal(images[0].url, `/assets/learning/thermometry-information-scale.${lang}.svg`);
    pairs.push({ math, codes, output });
  }
  assert.deepEqual(pairs[0], pairs[1]);
  const reader = await getLearningContent('thermodynamics');
  for (const lang of ['ko', 'en'] as const) {
    assert.ok(!reader[lang].includes('katex-error'));
    assert.ok(reader[lang].includes(`thermometry-information-scale.${lang}.svg`));
    assert.equal(reader.outline[lang].filter(h => /Research extension:|연구 확장:/.test(h.title)).length, 1);
  }
});

test('independent state enumeration, likelihood boundaries and score differentiation', () => {
  const report = JSON.parse(execFileSync('python3', ['scripts/learning/verify-thermometry-research.py'], { encoding: 'utf8', timeout: 20000 }));
  assert.equal(report.status, 'passed');
  assert.equal(report.namedStateCases, 27);
  assert.equal(report.finiteCountLaws, 81);
  assert.equal(report.boundaryCountCases, 4);
  assert.ok(Math.abs(report.relativeInformationMaximumX - 2.3993572805154677) < 1e-13);
});
