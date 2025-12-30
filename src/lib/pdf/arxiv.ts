import type { PaperMetadata } from '../ai/types';

/**
 * Extract arXiv ID from URL or direct input
 */
export function extractArxivId(input: string): string {
  // Clean the input
  input = input.trim();

  // URL patterns
  const urlPatterns = [
    /arxiv\.org\/abs\/(\d+\.\d+)/,
    /arxiv\.org\/pdf\/(\d+\.\d+)/,
    /arxiv\.org\/abs\/([a-z-]+\/\d+)/,
    /arxiv\.org\/pdf\/([a-z-]+\/\d+)/,
  ];

  for (const pattern of urlPatterns) {
    const match = input.match(pattern);
    if (match) return match[1];
  }

  // Direct ID input (new format: YYMM.NNNNN)
  const newIdMatch = input.match(/^(\d{4}\.\d{4,5})(v\d+)?$/);
  if (newIdMatch) return newIdMatch[1];

  // Old format: category/YYMMNNN
  const oldIdMatch = input.match(/^([a-z-]+\/\d{7})(v\d+)?$/);
  if (oldIdMatch) return oldIdMatch[1];

  throw new Error(`Invalid arXiv ID or URL: ${input}`);
}

/**
 * Fetch metadata from arXiv API
 */
export async function getArxivMetadata(arxivId: string): Promise<PaperMetadata> {
  const apiUrl = `https://export.arxiv.org/api/query?id_list=${arxivId}`;

  const response = await fetch(apiUrl);
  if (!response.ok) {
    throw new Error(`Failed to fetch arXiv metadata: ${response.statusText}`);
  }

  const xml = await response.text();
  return parseArxivXml(xml, arxivId);
}

/**
 * Parse arXiv API XML response
 */
function parseArxivXml(xml: string, arxivId: string): PaperMetadata {
  // Extract the <entry> section first (to avoid getting feed-level metadata)
  const entryMatch = xml.match(/<entry>([\s\S]*?)<\/entry>/);
  const entryXml = entryMatch ? entryMatch[1] : xml;

  // Helper to extract tag content from entry
  const getTag = (tag: string): string => {
    const pattern = new RegExp(`<${tag}[^>]*>([\\s\\S]*?)</${tag}>`);
    const match = entryXml.match(pattern);
    return match ? match[1].trim() : '';
  };

  // Extract authors (inside <author><name>...</name></author>)
  const authorMatches = [...entryXml.matchAll(/<author[^>]*>[\s\S]*?<name>([^<]+)<\/name>[\s\S]*?<\/author>/g)];
  const authors = authorMatches.map((m) => m[1].trim());

  // Extract categories from entry
  const categoryMatches = [...entryXml.matchAll(/term="([^"]+)"/g)];
  const categories = categoryMatches.map((m) => m[1]).filter((c) => !c.includes('http'));

  // Clean title (remove extra whitespace)
  const rawTitle = getTag('title');
  const title = rawTitle.replace(/\s+/g, ' ').trim();

  // Clean abstract
  const rawAbstract = getTag('summary');
  const abstract = rawAbstract.replace(/\s+/g, ' ').trim();

  return {
    id: arxivId,
    title,
    authors,
    abstract,
    published: getTag('published'),
    categories,
    pdfUrl: `https://arxiv.org/pdf/${arxivId}.pdf`,
  };
}

/**
 * Fetch PDF from arXiv
 */
export async function fetchArxivPdf(arxivId: string): Promise<Buffer> {
  const pdfUrl = `https://arxiv.org/pdf/${arxivId}.pdf`;

  const response = await fetch(pdfUrl, {
    headers: {
      'User-Agent': 'SuanLab-BlogGenerator/1.0',
    },
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch arXiv PDF: ${response.statusText}`);
  }

  return Buffer.from(await response.arrayBuffer());
}

/**
 * Format metadata for display
 */
export function formatMetadata(metadata: PaperMetadata): string {
  return `
**Title**: ${metadata.title}
**Authors**: ${metadata.authors.join(', ')}
**Published**: ${metadata.published.split('T')[0]}
**arXiv ID**: ${metadata.id}
**Categories**: ${metadata.categories.join(', ')}

**Abstract**:
${metadata.abstract}
`.trim();
}
