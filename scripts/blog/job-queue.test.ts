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
