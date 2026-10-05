import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { publishPosts } from './publish';

test('publishes only generated assets and treats shell syntax as a literal commit message', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'suanlab-publish-'));
  const cwd = path.join(root, 'repo');
  const remote = path.join(root, 'remote.git');
  const git = (...args: string[]) => execFileSync('git', args, { cwd, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim();
  try {
    fs.mkdirSync(cwd);
    execFileSync('git', ['init', '--bare', remote], { stdio: 'ignore' });
    git('init', '-b', 'master');
    git('config', 'user.email', 'test@example.com');
    git('config', 'user.name', 'Publish test');
    fs.writeFileSync(path.join(cwd, 'site.txt'), 'original');
    git('add', '.'); git('commit', '-m', 'Initial');
    git('remote', 'add', 'origin', remote);
    fs.writeFileSync(path.join(cwd, 'site.txt'), 'design in progress');
    fs.mkdirSync(path.join(cwd, 'content/blog'), { recursive: true });
    fs.mkdirSync(path.join(cwd, 'public/assets/images/blog'), { recursive: true });
    const post = 'content/blog/test.md';
    const thumbnail = 'public/assets/images/blog/test.jpg';
    fs.writeFileSync(path.join(cwd, post), '---\ntitle: Test\ndate: 2026-10-05\ntags: [test]\nthumbnail: /assets/images/blog/test.jpg\n---\nTest');
    fs.writeFileSync(path.join(cwd, thumbnail), 'image');
    const message = 'Add blog: $(touch INJECTED) "quoted"';
    assert.equal(publishPosts([post], message, cwd), 'master -> master');
    assert.equal(git('log', '-1', '--format=%s'), message);
    assert.equal(git('show', 'HEAD:site.txt'), 'original');
    assert.equal(fs.existsSync(path.join(cwd, 'INJECTED')), false);
    assert.deepEqual(git('diff-tree', '--no-commit-id', '--name-only', '-r', 'HEAD').split('\n').sort(), [post, thumbnail].sort());
    assert.equal(git('rev-parse', 'HEAD'), git('rev-parse', 'origin/master'));
    process.env.SUANLAB_PUBLISH_MODE = 'review';
    assert.equal(publishPosts([post], 'Review only', cwd), 'Saved for review');
    delete process.env.SUANLAB_PUBLISH_MODE;
    git('add', 'site.txt');
    assert.match(publishPosts([post], 'Do not include staged edits', cwd), /failed/);
    assert.equal(git('diff', '--cached', '--name-only'), 'site.txt');
    git('reset', '--', 'site.txt');
    git('checkout', '-b', 'design');
    assert.match(publishPosts([post], 'Wrong branch', cwd), /failed/);
  } finally {
    delete process.env.SUANLAB_PUBLISH_MODE;
    fs.rmSync(root, { recursive: true, force: true });
  }
});
