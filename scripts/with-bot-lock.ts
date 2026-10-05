import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';

const [action, ...args] = process.argv.slice(2);
const entries: Record<string, string> = { bot: 'scripts/slack-bot.ts', retry: 'scripts/blog/retry-publish.ts' };
if (!entries[action]) throw new Error('Use bot or retry JOB_ID.');
fs.mkdirSync('.runtime', { recursive: true });
// Linux flock is released by the kernel after exit or a crash. Both entry points
// share the lock so offline retries cannot race with the daemon's ledger writes.
const child = spawn('flock', ['--no-fork', '--nonblock', '--conflict-exit-code', '75', '.runtime/slack-queue.lock', process.execPath,
  path.resolve('node_modules/tsx/dist/cli.mjs'), entries[action], ...args], {
  stdio: 'inherit', env: { ...process.env, SUANLAB_QUEUE_LOCKED: '1' },
});
for (const signal of ['SIGINT', 'SIGTERM'] as const) process.on(signal, () => child.kill(signal));
child.on('error', error => { console.error(error.message); process.exitCode = 1; });
child.on('exit', code => {
  if (code === 75) console.error('The bot owns the queue. Use /suanblog retry JOB_ID, or stop the bot before an offline retry.');
  process.exitCode = code ?? 1;
});
