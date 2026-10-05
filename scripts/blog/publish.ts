import { execFileSync } from 'child_process';
import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';

/** Publish only the generated posts and their local thumbnails, never site edits. */
export function publishPosts(postFiles: string[], message: string, cwd = process.cwd()): string {
  const git = (...args: string[]) => execFileSync('git', args, {
    cwd, encoding: 'utf-8', timeout: 300000,
    env: { ...process.env, GIT_TERMINAL_PROMPT: '0' },
  }).trim();

  try {
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
      const { data } = matter(fs.readFileSync(path.join(cwd, relative), 'utf8'));
      if (!data.title || !data.date || Number.isNaN(Date.parse(data.date)) || !Array.isArray(data.tags)) throw new Error('Generated post metadata is incomplete or invalid.');
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
    if (process.env.SUANLAB_PUBLISH_MODE === 'review') return 'Saved for review';
    git('add', '--', ...paths);
    if (git('diff', '--cached', '--name-only', '--', ...paths)) {
      git('commit', '--only', '-m', message, '--', ...paths);
    }
    git('push', 'origin', 'master');
    return 'master -> master';
  } catch (error) {
    // Do not report a successful deployment after a failed commit or push.
    console.error('[PUBLISH]', error instanceof Error ? error.message : 'Git publication failed');
    return 'Publication failed; generated files remain local.';
  }
}
