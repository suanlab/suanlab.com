import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { validatePost } from './validate-post';

test('publication rejects missing assets and invalid metadata before staging', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'post-validation-'));
  const header = '---\ntitle: Example\nexcerpt: Summary\ncategory: Tutorial\ndate: 2026-10-05\ntags: [AI]\n---\n';
  try {
    const asset = 'public/assets/images/blog/figure.svg';
    fs.mkdirSync(path.join(root, path.dirname(asset)), { recursive: true });
    assert.throws(() => validatePost(header + '![](/assets/images/blog/figure.svg)', root), /Missing local asset/);
    fs.writeFileSync(path.join(root, asset), '<svg />');
    assert.deepEqual(validatePost(header + '![](/assets/images/blog/figure.svg)', root).assets, [asset]);
    assert.throws(() => validatePost(header.replace('excerpt: Summary', 'excerpt: ""') + 'Body', root), /excerpt/);
    assert.throws(() => validatePost(header + '[link](javascript:alert)', root), /scheme/);
    assert.throws(() => validatePost(header, root), /empty/);
    fs.symlinkSync('/etc/hosts', path.join(root, 'public/assets/images/blog/outside'));
    assert.throws(() => validatePost(header + '![](/assets/images/blog/outside)', root), /inside public/);
    assert.doesNotThrow(() => validatePost(header + '```md\n![](/assets/missing.png)\n```', root));
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});
