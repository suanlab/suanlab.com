import fs from 'node:fs';
import path from 'node:path';
import matter from 'gray-matter';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import { visit } from 'unist-util-visit';

/** Validate metadata and local assets before staging any generated content. */
export function validatePost(source: string, cwd = process.cwd()) {
  const { data, content } = matter(source);
  for (const key of ['title', 'excerpt', 'category']) {
    if (typeof data[key] !== 'string' || !data[key].trim()) throw new Error(`Post ${key} is required.`);
  }
  if (!data.date || Number.isNaN(Date.parse(data.date))) throw new Error('Post date is invalid.');
  if (!Array.isArray(data.tags) || data.tags.some(tag => typeof tag !== 'string' || !tag.trim())) throw new Error('Post tags must be non-empty strings.');
  if (!content.trim()) throw new Error('Post body is empty.');
  const assets = new Set<string>();
  function checkUrl(value: string) {
    const url = new URL(value, 'https://suanlab.com');
    if (!['https:', 'http:', 'mailto:', 'tel:'].includes(url.protocol)) throw new Error('Unsupported content URL scheme.');
    if (url.origin !== 'https://suanlab.com' || !url.pathname.startsWith('/assets/')) return;
    const publicDir = path.resolve(cwd, 'public');
    const filename = path.resolve(publicDir, `.${decodeURIComponent(url.pathname)}`);
    if (!filename.startsWith(publicDir + path.sep) || !fs.existsSync(filename) || !fs.statSync(filename).isFile()) throw new Error(`Missing local asset: ${url.pathname}`);
    if (!fs.realpathSync(filename).startsWith(fs.realpathSync(publicDir) + path.sep)) throw new Error('Asset must remain inside public.');
    assets.add(path.relative(cwd, filename));
  }
  if (data.thumbnail) checkUrl(data.thumbnail);
  visit(unified().use(remarkParse).parse(content), node => {
    if (node.type === 'link' || node.type === 'image' || node.type === 'definition') checkUrl(node.url);
  });
  return { data, assets: [...assets] };
}
