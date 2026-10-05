import { execFileSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import { validatePost } from './validate-post';

/** Publish only the generated posts and their local thumbnails, never site edits. */
export function publishPosts(postFiles: string[], message: string, cwd = process.cwd(), onStage: (stage: string) => void = () => {}): string {
  const git = (...args: string[]) => execFileSync('git', args, {
    cwd, encoding: 'utf-8', timeout: 300000,
    env: { ...process.env, GIT_TERMINAL_PROMPT: '0' },
  }).trim();

  try {
    onStage('validating');
    if (git('branch', '--show-current') !== 'master') {
      throw new Error('Automatic publication requires the master branch. Generated files were saved locally.');
    }
    if (git('diff', '--cached', '--name-only')) {
      throw new Error('Staged changes exist. Commit or unstage them before publishing. Generated files were saved locally.');
    }
    const files = new Set<string>();
    for (const file of postFiles) {
      const relative = path.relative(cwd, path.resolve(cwd, file));
      if (!relative.startsWith('content/blog/') || !relative.endsWith('.md')) {
        throw new Error('Only blog markdown files may be published.');
      }
      const realFile = fs.realpathSync(path.join(cwd, relative));
      if (!realFile.startsWith(fs.realpathSync(path.join(cwd, 'content/blog')) + path.sep)) throw new Error('Post must remain inside content/blog.');
      files.add(relative);
      const { data, assets } = validatePost(fs.readFileSync(path.join(cwd, relative), 'utf8'), cwd);
      for (const asset of assets) {
        if (asset.startsWith('public/assets/images/blog/')) files.add(asset);
        else git('ls-files', '--error-unmatch', '--', asset);
      }
      if (typeof data.thumbnail === 'string' && data.thumbnail.startsWith('/assets/images/blog/')) {
        const thumbnail = path.normalize(`public${data.thumbnail}`);
        if (!thumbnail.startsWith('public/assets/images/blog/')) throw new Error('Invalid thumbnail path.');
        if (!fs.existsSync(path.join(cwd, thumbnail))) throw new Error('Generated thumbnail is missing.');
        if (!fs.realpathSync(path.join(cwd, thumbnail)).startsWith(fs.realpathSync(path.join(cwd, 'public/assets/images/blog')) + path.sep)) throw new Error('Thumbnail must remain inside blog assets.');
        files.add(thumbnail);
      }
    }
    if (!files.size) throw new Error('No generated posts to publish.');
    const paths = [...files];
    if (process.env.SUANLAB_PUBLISH_MODE === 'review') { onStage('review'); return 'Saved for review'; }
    onStage('syncing');
    git('fetch', 'origin', 'master');
    const [ahead, behind] = git('rev-list', '--left-right', '--count', 'HEAD...origin/master').split(/\s+/).map(Number);
    if (behind) throw new Error('Local master is behind origin/master. Synchronize before retrying saved files.');
    if (ahead) {
      const pending = git('log', '--format=', '--name-only', 'origin/master..HEAD').split('\n').filter(Boolean);
      if (pending.some(file => !files.has(file))) throw new Error('Unpushed commits include unrelated files. Publish them separately before retrying.');
    }
    onStage('staging');
    git('add', '--', ...paths);
    if (git('diff', '--cached', '--name-only', '--', ...paths)) {
      onStage('committing');
      git('commit', '--only', '-m', message, '--', ...paths);
    }
    onStage('pushing');
    git('push', 'origin', 'master');
    return 'master -> master';
  } catch (error) {
    // Do not report a successful deployment after a failed commit or push.
    console.error('[PUBLISH]', error instanceof Error ? error.message : 'Git publication failed');
    return 'Publication failed; generated files remain local.';
  }
}
