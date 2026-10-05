import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { JobQueue, requestKey } from './job-queue';
test('serializes requests, persists deduplication, and recovers interrupted jobs', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'suanlab-queue-'));
  try {
    const file = path.join(dir, 'jobs.json'); const queue = new JobQueue(file); const order: number[] = [];
    const first = queue.run('first', async () => { order.push(1); await new Promise(r => setTimeout(r, 15)); order.push(2); });
    const second = queue.run('second', async () => { order.push(3); });
    await assert.rejects(queue.run('first', async () => {}));
    await Promise.all([first, second]); assert.deepEqual(order, [1, 2, 3]);
    await assert.rejects(new JobQueue(file).run('first', async () => {}));
    await assert.rejects(queue.run('bad', async () => { throw new Error('test failure'); }));
    await queue.run('after failure', async () => { order.push(4); });
    assert.equal(queue.jobs[requestKey('bad')].status, 'failed');
    queue.update(requestKey('bad'), { status: 'running' });
    assert.equal(new JobQueue(file).jobs[requestKey('bad')].status, 'interrupted');
    assert.equal(requestKey('https://arxiv.org/pdf/2312.00752v2.pdf'), requestKey('2312.00752'));
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});

test('publication retry is serialized, durable, and never repeats generation', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'suanlab-retry-'));
  try {
    const file = path.join(dir, 'jobs.json'); const queue = new JobQueue(file);
    let generated = 0; const id = requestKey('paper');
    await assert.rejects(queue.run('paper', async id => {
      generated++; queue.update(id, { files: ['content/blog/saved.md'], stage: 'pushing' });
      queue.update(id, { stage: 'publish-failed' });
    }));
    const order: string[] = [];
    const active = queue.run('another', async () => { order.push('start'); await new Promise(r => setTimeout(r, 10)); order.push('end'); });
    const retry = queue.retryPublication(id, files => { order.push('publish'); assert.deepEqual(files, ['content/blog/saved.md']); return 'master -> master'; });
    await assert.rejects(queue.retryPublication(id, () => 'master -> master'), /already/);
    await Promise.all([active, retry]);
    assert.deepEqual(order, ['start', 'end', 'publish']); assert.equal(generated, 1);
    const restored = new JobQueue(file);
    assert.equal(restored.jobs[id].status, 'completed');
    assert.equal(restored.jobs[id].attempts?.length, 2);
    assert.equal(restored.jobs[id].attempts?.[0].failureStage, 'pushing');
    assert.equal(restored.jobs[id].attempts?.[1].outcome, 'completed');
    assert.equal(restored.metrics().retries, 1); assert.equal(restored.metrics().pushed, 1);
    await assert.rejects(restored.retryPublication(id, () => 'master -> master'), /already been pushed/);
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});

test('successful publication does not erase partial generation failure', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'suanlab-partial-'));
  try {
    const queue = new JobQueue(path.join(dir, 'jobs.json')); const id = requestKey('batch');
    await assert.rejects(queue.run('batch', async id => {
      queue.update(id, { files: ['saved.md'], stage: 'saved' });
      queue.update(id, { status: 'failed', error: 'One paper failed to generate' });
      queue.update(id, { stage: 'pushing' }); queue.update(id, { stage: 'publish-failed' });
    }));
    await queue.retryPublication(id, () => 'master -> master');
    assert.equal(queue.jobs[id].stage, 'pushed'); assert.equal(queue.jobs[id].status, 'failed');
    assert.match(queue.jobs[id].error!, /One paper/);
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});
