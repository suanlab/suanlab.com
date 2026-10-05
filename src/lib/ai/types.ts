import type { ContentProvenance } from '../content-provenance';
export type GenerationStage = 'fetching-source' | 'parsing-source' | 'generating-text' | 'extracting-figure' | 'generating-image';

export interface TopicGeneratorOptions {
  onProgress?: (stage: GenerationStage) => void;
  topic: string;
  category?: string;
  tags?: string[];
  style?: 'tutorial' | 'explanation' | 'news' | 'review';
  language?: 'ko' | 'en';
}

export interface PaperSummarizerOptions {
  onProgress?: (stage: GenerationStage) => void;
  arxivId?: string;
  pdfUrl?: string;
  pdfBuffer?: Buffer;
  localPath?: string;
  summaryStyle?: 'detailed' | 'brief' | 'technical';
}

export interface PaperMetadata {
  id: string;
  title: string;
  authors: string[];
  abstract: string;
  published: string;
  categories: string[];
  pdfUrl: string;
}

export interface GeneratedPost {
  provenance?: ContentProvenance;
  slug: string;
  title: string;
  subtitle?: string;
  date: string;
  excerpt: string;
  category: string;
  tags: string[];
  content: string;
  thumbnail?: string;
}

export interface BlogFrontmatter {
  title: string;
  date: string;
  excerpt: string;
  category: string;
  tags: string[];
  thumbnail?: string;
}
