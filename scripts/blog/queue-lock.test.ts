import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn, spawnSync } from 'node:child_process';
import { once } from 'node:events';

test('kernel queue lock rejects a second writer and releases after termination', { skip: process.platform !== 'linux' }, async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'suanlab-lock-'));
  const lock = path.join(dir, 'queue.lock');
  const args = ['--no-fork', '--nonblock', '--conflict-exit-code', '75', lock];
  const owner = spawn('flock', [...args, process.execPath, '-e', 'console.log("ready");setInterval(()=>{},1000)']);
  try {
    await once(owner.stdout!, 'data');
    assert.equal(spawnSync('flock', [...args, 'true']).status, 75);
    const exited = once(owner, 'exit'); owner.kill('SIGKILL'); await exited;
    assert.equal(spawnSync('flock', [...args, 'true']).status, 0);
  } finally { owner.kill(); fs.rmSync(dir, { recursive: true, force: true }); }
});
