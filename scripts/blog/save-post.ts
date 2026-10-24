import fs from 'node:fs/promises';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { formatAsMarkdown } from './post-format';
import type { GeneratedPost } from '../../src/lib/ai/types';

/** Each generation gets its own image names before any files are written. */
export function uniqueGenerationSlug(base: string): string {
  return `${base.replace(/-+$/, '')}-${randomUUID().slice(0, 8)}`;
}

export async function saveGeneratedPost(post: GeneratedPost, customSlug?: string, cwd = process.cwd()): Promise<string> {
  const slug = customSlug || post.slug;
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(slug)) throw new Error('Post slug must contain lowercase letters, digits and single hyphens.');
  const directory = path.join(cwd, 'content/blog');
  await fs.mkdir(directory, { recursive: true });
  const filename = path.join(directory, `${slug}.md`);
  // An explicit output name must never silently replace a published article.
  await fs.writeFile(filename, formatAsMarkdown({ ...post, slug }), { encoding: 'utf8', flag: 'wx' });
  return filename;
}
