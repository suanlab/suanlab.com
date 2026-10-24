import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createContentStore } from '../src/lib/content-store';
import { registerQTSlugs } from '../src/lib/qt-slugs';

test('content cache reuses parsed records and detects development edits, additions and deletions', () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'suanlab-content-'));
  const environment: Record<string, string | undefined> = process.env;
  const originalMode = environment.NODE_ENV;
  environment.NODE_ENV = 'development';
  try {
    fs.writeFileSync(path.join(directory, 'a.md'), 'first');
    let parses = 0;
    const store = createContentStore(directory, (filename, source) => { parses++; return { filename, source }; });
    assert.equal(store.all()[0].source, 'first');
    store.all().reverse(); store.get('a.md');
    assert.equal(parses, 1);
    fs.writeFileSync(path.join(directory, 'a.md'), 'second version');
    assert.equal(store.get('a.md')?.source, 'second version');
    fs.writeFileSync(path.join(directory, 'b.md'), 'new file');
    assert.deepEqual(store.all().map(entry => entry.filename), ['a.md', 'b.md']);
    fs.unlinkSync(path.join(directory, 'a.md'));
    assert.deepEqual(store.filenames(), ['b.md']);
    assert.equal(store.get('a.md'), null);
    assert.equal(store.get('../escape.md'), null);
    assert.equal(parses, 3);
  } finally {
    if (originalMode === undefined) delete environment.NODE_ENV; else environment.NODE_ENV = originalMode;
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

test('new and retired QT files never reassign existing duplicate-date URLs', () => {
  const original = { '2026-10-24 Z.md': '2026-10-24', '2026-10-24 retired.md': '2026-10-24-2' };
  const first = registerQTSlugs(['2026-10-24 A.md', '2026-10-24 Z.md'], original);
  assert.equal(first['2026-10-24 Z.md'], '2026-10-24');
  assert.equal(first['2026-10-24 A.md'], '2026-10-24-3');
  const second = registerQTSlugs(['2026-10-24 0.md', '2026-10-24 A.md'], first);
  assert.equal(second['2026-10-24 A.md'], '2026-10-24-3');
  assert.equal(second['2026-10-24 0.md'], '2026-10-24-4');
  assert.deepEqual(original, { '2026-10-24 Z.md': '2026-10-24', '2026-10-24 retired.md': '2026-10-24-2' });
});
