import { generateWithAI, parseGeneratedContent } from '../../src/lib/ai/claude';
import { buildTopicPrompt } from '../../src/lib/ai/prompts';
import { generateAndSaveThumbnail } from '../../src/lib/ai/image';
import type { TopicGeneratorOptions, GeneratedPost } from '../../src/lib/ai/types';
import fs from 'fs/promises';
import path from 'path';

const CONTENT_DIR = path.join(process.cwd(), 'content/blog');

/**
 * Generate a blog post from a topic
 */
export async function generateFromTopic(
  options: TopicGeneratorOptions & { generateImage?: boolean }
): Promise<GeneratedPost> {
  const prompt = buildTopicPrompt({
    topic: options.topic,
    category: options.category,
    tags: options.tags,
  });

  const rawContent = await generateWithAI(prompt, {
    maxTokens: 4096,
    temperature: 0.7,
  });

  const parsed = parseGeneratedContent(rawContent);
  const date = new Date().toISOString().split('T')[0];
  const slug = generateSlug(parsed.title);

  // Generate thumbnail image if requested
  let thumbnail = '/assets/images/blog/default.jpg';
  if (options.generateImage) {
    thumbnail = await generateAndSaveThumbnail(options.topic, slug, 'technical');
  }

  return {
    slug,
    title: parsed.title,
    date,
    excerpt: parsed.excerpt,
    category: options.category || 'General',
    tags: options.tags?.length ? options.tags : parsed.suggestedTags,
    content: rawContent,
    thumbnail,
  };
}

/**
 * Format post as markdown with frontmatter
 */
export function formatAsMarkdown(post: GeneratedPost): string {
  const frontmatter = `---
title: "${escapeQuotes(post.title)}"
date: "${post.date}"
excerpt: "${escapeQuotes(post.excerpt)}"
category: "${post.category}"
tags: ${JSON.stringify(post.tags)}
thumbnail: "${post.thumbnail}"
---

`;

  return frontmatter + post.content;
}

/**
 * Save post to content directory
 */
export async function savePost(
  post: GeneratedPost,
  customSlug?: string
): Promise<string> {
  const slug = customSlug || post.slug;
  const markdown = formatAsMarkdown({ ...post, slug });
  const filepath = path.join(CONTENT_DIR, `${slug}.md`);

  // Ensure directory exists
  await fs.mkdir(CONTENT_DIR, { recursive: true });

  await fs.writeFile(filepath, markdown, 'utf-8');
  return filepath;
}

/**
 * Generate URL-friendly slug from title
 */
function generateSlug(title: string): string {
  const date = new Date().toISOString().split('T')[0].replace(/-/g, '');
  const cleanTitle = title
    .toLowerCase()
    // Remove Korean characters for URL safety, but keep meaning
    .replace(/[가-힣]+/g, (match) => {
      // Simple romanization mapping for common terms
      const mapping: Record<string, string> = {
        '딥러닝': 'deep-learning',
        '머신러닝': 'machine-learning',
        '인공지능': 'ai',
        '자연어처리': 'nlp',
        '컴퓨터비전': 'computer-vision',
        '데이터': 'data',
        '분석': 'analysis',
        '모델': 'model',
        '학습': 'learning',
        '신경망': 'neural-network',
        '트랜스포머': 'transformer',
        '논문': 'paper',
        '리뷰': 'review',
      };
      return mapping[match] || '';
    })
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '')
    .slice(0, 50);

  return `${date}-${cleanTitle || 'post'}`;
}

/**
 * Escape quotes in string for YAML frontmatter
 */
function escapeQuotes(str: string): string {
  return str.replace(/"/g, '\\"');
}
