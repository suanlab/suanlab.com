export interface TopicGeneratorOptions {
  topic: string;
  category?: string;
  tags?: string[];
  style?: 'tutorial' | 'explanation' | 'news' | 'review';
  language?: 'ko' | 'en';
}

export interface PaperSummarizerOptions {
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
  slug: string;
  title: string;
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
