import pdfParse from 'pdf-parse';
import fs from 'fs/promises';

export interface ParsedPdf {
  text: string;
  numPages: number;
  info: {
    title?: string;
    author?: string;
    creationDate?: string;
  };
}

export async function parsePdfFromFile(filePath: string): Promise<ParsedPdf> {
  const buffer = await fs.readFile(filePath);
  return parsePdfFromBuffer(buffer);
}

export async function parsePdfFromBuffer(buffer: Buffer): Promise<ParsedPdf> {
  const data = await pdfParse(buffer);

  return {
    text: cleanText(data.text),
    numPages: data.numpages,
    info: {
      title: data.info?.Title,
      author: data.info?.Author,
      creationDate: data.info?.CreationDate,
    },
  };
}

export async function parsePdfFromUrl(url: string): Promise<ParsedPdf> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch PDF: ${response.statusText}`);
  }

  const buffer = Buffer.from(await response.arrayBuffer());
  return parsePdfFromBuffer(buffer);
}

function cleanText(text: string): string {
  return (
    text
      // Remove multiple spaces
      .replace(/[ \t]+/g, ' ')
      // Normalize line breaks
      .replace(/\r\n/g, '\n')
      // Remove page numbers and headers/footers (common patterns)
      .replace(/^\d+\s*$/gm, '')
      // Remove excessive newlines
      .replace(/\n{3,}/g, '\n\n')
      .trim()
  );
}

/**
 * Split text into chunks for processing with LLM
 * @param text - The full text to chunk
 * @param maxChars - Maximum characters per chunk (default ~100k chars = ~25k tokens)
 */
export function chunkText(text: string, maxChars: number = 100000): string[] {
  const chunks: string[] = [];
  const paragraphs = text.split(/\n\n+/);

  let currentChunk = '';

  for (const para of paragraphs) {
    if (currentChunk.length + para.length + 2 > maxChars) {
      if (currentChunk) {
        chunks.push(currentChunk.trim());
      }
      // If single paragraph is too long, split by sentences
      if (para.length > maxChars) {
        const sentences = para.split(/(?<=[.!?])\s+/);
        let sentenceChunk = '';
        for (const sentence of sentences) {
          if (sentenceChunk.length + sentence.length > maxChars) {
            if (sentenceChunk) {
              chunks.push(sentenceChunk.trim());
            }
            sentenceChunk = sentence;
          } else {
            sentenceChunk += ' ' + sentence;
          }
        }
        currentChunk = sentenceChunk;
      } else {
        currentChunk = para;
      }
    } else {
      currentChunk += '\n\n' + para;
    }
  }

  if (currentChunk.trim()) {
    chunks.push(currentChunk.trim());
  }

  return chunks;
}

/**
 * Extract metadata from PDF text (fallback when metadata is not available)
 */
export function extractMetadataFromText(text: string): {
  title?: string;
  authors?: string[];
  abstract?: string;
} {
  const lines = text.split('\n').filter((l) => l.trim());

  // Title is usually the first non-empty line
  const title = lines[0]?.trim();

  // Try to find abstract
  const abstractMatch = text.match(
    /abstract[:\s]*\n?([\s\S]*?)(?=\n\s*(?:1\.?\s*introduction|keywords|1\s+introduction))/i
  );
  const abstract = abstractMatch?.[1]?.trim();

  // Try to find authors (usually between title and abstract)
  const authorsMatch = text
    .slice(0, text.toLowerCase().indexOf('abstract'))
    .match(/([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)*)/);
  const authors = authorsMatch?.[1]?.split(/\s*,\s*/).filter(Boolean);

  return {
    title,
    authors,
    abstract,
  };
}
