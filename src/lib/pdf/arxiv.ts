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
 * Helper: wait with exponential backoff
 */
async function waitWithBackoff(attempt: number, baseMs: number, retryAfter?: string): Promise<void> {
  if (retryAfter) {
    const seconds = parseInt(retryAfter, 10);
    if (!isNaN(seconds) && seconds > 0) {
      console.log(`  Retry-After header: waiting ${seconds}s...`);
      await new Promise(r => setTimeout(r, seconds * 1000));
      return;
    }
  }
  const waitMs = baseMs * Math.pow(2, attempt - 1); // 30s, 60s, 120s, ...
  console.log(`  Backing off ${waitMs / 1000}s before retry...`);
  await new Promise(r => setTimeout(r, waitMs));
}

/**
 * Fetch metadata from arXiv API
 */
export async function getArxivMetadata(arxivId: string, retries = 5): Promise<PaperMetadata> {
  const apiUrl = `https://export.arxiv.org/api/query?id_list=${arxivId}`;

  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const response = await fetch(apiUrl, {
        headers: {
          'User-Agent': 'SuanLab-BlogGenerator/1.0',
        },
      });

      // Handle HTTP 429 (Too Many Requests) explicitly
      if (response.status === 429) {
        const retryAfter = response.headers.get('Retry-After') ?? undefined;
        if (attempt < retries) {
          console.log(`Rate limited (HTTP 429), attempt ${attempt}/${retries}`);
          await waitWithBackoff(attempt, 30000, retryAfter);
          continue;
        }
        throw new Error('arXiv API rate limit exceeded (HTTP 429). Please try again later.');
      }

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const xml = await response.text();

      // Check for rate limit in response body (arXiv sometimes returns 200 with rate limit message)
      if (xml.includes('Rate exceeded') || xml.trim().length < 100) {
        if (attempt < retries) {
          console.log(`Rate limited (body), waiting before retry ${attempt + 1}/${retries}...`);
          await waitWithBackoff(attempt, 30000);
          continue;
        }
        throw new Error('arXiv API rate limit exceeded. Please try again later.');
      }

      return parseArxivXml(xml, arxivId);
    } catch (error) {
      if (attempt === retries) {
        throw new Error(`Failed to fetch arXiv metadata after ${retries} attempts: ${error instanceof Error ? error.message : 'Unknown Error'}`);
      }
      // For non-429 errors, use shorter backoff
      const waitMs = 10000 * attempt; // 10s, 20s, 30s, ...
      console.log(`Attempt ${attempt} failed: ${error instanceof Error ? error.message : 'Unknown'}. Retrying in ${waitMs / 1000}s...`);
      await new Promise(r => setTimeout(r, waitMs));
    }
  }

  throw new Error('Failed to fetch arXiv metadata');
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
 * Fetch PDF from arXiv (with retry for rate limiting)
 */
export async function fetchArxivPdf(arxivId: string, retries = 3): Promise<Buffer> {
  const pdfUrl = `https://arxiv.org/pdf/${arxivId}.pdf`;

  for (let attempt = 1; attempt <= retries; attempt++) {
    const response = await fetch(pdfUrl, {
      headers: {
        'User-Agent': 'SuanLab-BlogGenerator/1.0',
      },
    });

    if (response.status === 429) {
      if (attempt < retries) {
        const retryAfter = response.headers.get('Retry-After') ?? undefined;
        console.log(`PDF fetch rate limited (429), attempt ${attempt}/${retries}`);
        await waitWithBackoff(attempt, 30000, retryAfter);
        continue;
      }
      throw new Error('arXiv PDF rate limit exceeded (HTTP 429). Please try again later.');
    }

    if (!response.ok) {
      if (attempt < retries) {
        console.log(`PDF fetch failed (${response.status}), retrying in ${10 * attempt}s...`);
        await new Promise(r => setTimeout(r, 10000 * attempt));
        continue;
      }
      throw new Error(`Failed to fetch arXiv PDF: ${response.status} ${response.statusText}`);
    }

    return Buffer.from(await response.arrayBuffer());
  }

  throw new Error('Failed to fetch arXiv PDF');
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
