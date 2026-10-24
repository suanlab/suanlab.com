import { downloadBytes } from '../http';
import pdfParse from 'pdf-parse';
import fs from 'fs/promises';

export const MAX_PDF_BYTES = 50 * 1024 * 1024;

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
  if ((await fs.stat(filePath)).size > MAX_PDF_BYTES) throw new Error('PDF exceeds the size limit.');
  const buffer = await fs.readFile(filePath);
  return parsePdfFromBuffer(buffer);
}

export async function parsePdfFromBuffer(buffer: Buffer): Promise<ParsedPdf> {
  if (buffer.length > MAX_PDF_BYTES) throw new Error('PDF exceeds the size limit.');
  if (!buffer.subarray(0, 1024).includes(Buffer.from('%PDF-'))) throw new Error('Source is not a PDF.');
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
  return parsePdfFromBuffer(await downloadBytes(url, { maxBytes: MAX_PDF_BYTES }));
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
  if (!Number.isSafeInteger(maxChars) || maxChars < 2) throw new Error('Chunk size must be an integer of at least two characters.');
  const chunks: string[] = [];
  let remaining = text.trim();
  while (remaining.length > maxChars) {
    const window = remaining.slice(0, maxChars + 1);
    const paragraph = window.lastIndexOf('\n\n', maxChars - 1);
    const whitespace = window.search(/\s+\S*$/);
    let end = paragraph > maxChars / 2 ? paragraph : whitespace > maxChars / 2 ? whitespace : maxChars;
    // Do not split UTF-16 surrogate pairs when a hard boundary is necessary.
    if (/[\uD800-\uDBFF]/.test(remaining[end - 1])) end--;
    chunks.push(remaining.slice(0, end).trim());
    remaining = remaining.slice(end).trim();
  }
  if (remaining) chunks.push(remaining);
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
