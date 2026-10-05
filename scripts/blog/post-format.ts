import matter from 'gray-matter';
import type { GeneratedPost } from '../../src/lib/ai/types';
import { parseProvenance } from '../../src/lib/content-provenance';

export function formatAsMarkdown(post: GeneratedPost): string {
  const { slug: _slug, content, ...metadata } = post;
  parseProvenance(metadata.provenance);
  // YAML serialization preserves quotes, backslashes, multiline titles, and nested provenance.
  return matter.stringify(content, Object.fromEntries(Object.entries({ ...metadata, tags: [...new Set(metadata.tags)] }).filter(([, value]) => value !== undefined)));
}
