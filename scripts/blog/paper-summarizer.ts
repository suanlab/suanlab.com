import { generateWithAI, parseGeneratedContent } from '../../src/lib/ai/claude';
import {
  buildPaperPrompt,
  buildChunkPrompt,
  buildSynthesisPrompt,
} from '../../src/lib/ai/prompts';
import { generateAndSaveThumbnail } from '../../src/lib/ai/image';
import type { PaperSummarizerOptions, GeneratedPost, PaperMetadata } from '../../src/lib/ai/types';
import {
  parsePdfFromBuffer,
  parsePdfFromFile,
  parsePdfFromUrl,
  chunkText,
  extractMetadataFromText,
} from '../../src/lib/pdf/parser';
import {
  extractArxivId,
  getArxivMetadata,
  fetchArxivPdf,
  formatMetadata,
} from '../../src/lib/pdf/arxiv';
import fs from 'fs/promises';
import path from 'path';

const CONTENT_DIR = path.join(process.cwd(), 'content/blog');

/**
 * Generate a blog post summarizing a paper
 */
export async function generateFromPaper(
  options: PaperSummarizerOptions & { generateImage?: boolean }
): Promise<GeneratedPost> {
  let pdfText: string;
  let metadata: PaperMetadata;

  // Get PDF content and metadata based on source
  if (options.arxivId) {
    const arxivId = extractArxivId(options.arxivId);
    console.log(`Fetching arXiv metadata for ${arxivId}...`);
    metadata = await getArxivMetadata(arxivId);

    console.log(`Downloading PDF...`);
    const pdfBuffer = await fetchArxivPdf(arxivId);

    console.log(`Parsing PDF (${metadata.title})...`);
    const parsed = await parsePdfFromBuffer(pdfBuffer);
    pdfText = parsed.text;
  } else if (options.pdfUrl) {
    console.log(`Fetching PDF from URL...`);
    const parsed = await parsePdfFromUrl(options.pdfUrl);
    pdfText = parsed.text;

    // Extract metadata from PDF content
    const extracted = extractMetadataFromText(pdfText);
    metadata = {
      id: 'url-pdf',
      title: parsed.info.title || extracted.title || 'Untitled Paper',
      authors: extracted.authors || [parsed.info.author || 'Unknown'],
      abstract: extracted.abstract || '',
      published: parsed.info.creationDate || new Date().toISOString(),
      categories: [],
      pdfUrl: options.pdfUrl,
    };
  } else if (options.pdfBuffer) {
    console.log(`Parsing PDF buffer...`);
    const parsed = await parsePdfFromBuffer(options.pdfBuffer);
    pdfText = parsed.text;

    const extracted = extractMetadataFromText(pdfText);
    metadata = {
      id: 'uploaded-pdf',
      title: parsed.info.title || extracted.title || 'Untitled Paper',
      authors: extracted.authors || [parsed.info.author || 'Unknown'],
      abstract: extracted.abstract || '',
      published: parsed.info.creationDate || new Date().toISOString(),
      categories: [],
      pdfUrl: '',
    };
  } else if (options.localPath) {
    console.log(`Parsing local PDF: ${options.localPath}`);
    const parsed = await parsePdfFromFile(options.localPath);
    pdfText = parsed.text;

    const extracted = extractMetadataFromText(pdfText);
    metadata = {
      id: 'local-pdf',
      title: parsed.info.title || extracted.title || 'Untitled Paper',
      authors: extracted.authors || [parsed.info.author || 'Unknown'],
      abstract: extracted.abstract || '',
      published: parsed.info.creationDate || new Date().toISOString(),
      categories: [],
      pdfUrl: '',
    };
  } else {
    throw new Error('No paper source provided');
  }

  // Generate summary
  const rawContent = await summarizePaper(pdfText, metadata, options);
  const parsed = parseGeneratedContent(rawContent);
  const date = new Date().toISOString().split('T')[0];

  // Generate slug from paper title
  const slug = generatePaperSlug(metadata.title, metadata.id);

  // Generate thumbnail image if requested
  let thumbnail = '/assets/images/blog/paper-review.jpg';
  if (options.generateImage) {
    thumbnail = await generateAndSaveThumbnail(metadata.title, slug, 'technical');
  }

  return {
    slug,
    title: `[논문 리뷰] ${metadata.title}`,
    date,
    excerpt: metadata.abstract.slice(0, 200) + '...',
    category: 'Paper Review',
    tags: ['Paper Review', ...metadata.categories.slice(0, 3)],
    content: rawContent,
    thumbnail,
  };
}

/**
 * Summarize paper content using Claude
 */
async function summarizePaper(
  pdfText: string,
  metadata: PaperMetadata,
  options: PaperSummarizerOptions
): Promise<string> {
  const maxCharsPerChunk = 40000; // ~10k tokens per chunk (reduced for stability)
  const chunks = chunkText(pdfText, maxCharsPerChunk);

  console.log(`Paper split into ${chunks.length} chunks`);

  if (chunks.length === 1) {
    // Small paper - direct summarization
    console.log('Generating detailed summary...');
    const prompt = buildPaperPrompt({
      title: metadata.title,
      authors: metadata.authors.join(', '),
      date: metadata.published.split('T')[0],
      arxivId: metadata.id,
      content: chunks[0],
    });

    return generateWithAI(prompt, {
      maxTokens: 16000,
      temperature: 0.5,
    });
  }

  // Large paper - chunk and synthesize
  console.log('Summarizing chunks...');
  const chunkSummaries: string[] = [];

  for (let i = 0; i < chunks.length; i++) {
    console.log(`  Processing chunk ${i + 1}/${chunks.length}...`);
    const summary = await generateWithAI(buildChunkPrompt(chunks[i]), {
      maxTokens: 4000,
      temperature: 0.3,
    });
    chunkSummaries.push(summary);
  }

  // Synthesize all summaries
  console.log('Synthesizing detailed final summary...');
  const metadataStr = formatMetadata(metadata);
  const synthesisPrompt = buildSynthesisPrompt(metadataStr, chunkSummaries);

  return generateWithAI(synthesisPrompt, {
    maxTokens: 16000,
    temperature: 0.5,
  });
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

  await fs.mkdir(CONTENT_DIR, { recursive: true });
  await fs.writeFile(filepath, markdown, 'utf-8');

  return filepath;
}

/**
 * Generate slug for paper review post
 */
function generatePaperSlug(title: string, paperId: string): string {
  const date = new Date().toISOString().split('T')[0].replace(/-/g, '');

  // Clean paper ID for URL
  const cleanId = paperId.replace(/[^a-zA-Z0-9]/g, '-').slice(0, 20);

  // Clean title
  const cleanTitle = title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '')
    .slice(0, 30);

  return `${date}-paper-${cleanId}-${cleanTitle}`;
}

/**
 * Escape quotes for YAML frontmatter
 */
function escapeQuotes(str: string): string {
  return str.replace(/"/g, '\\"');
}
