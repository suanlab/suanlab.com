import fs from 'fs';
import path from 'path';

const QT_DIR = path.join(process.cwd(), 'content/qt');

// 성경 책 순서 (구약 39권 + 신약 27권)
export const BIBLE_BOOKS = {
  // 구약 (Old Testament)
  '창세기': { order: 1, abbr: '창', testament: 'old' },
  '출애굽기': { order: 2, abbr: '출', testament: 'old' },
  '레위기': { order: 3, abbr: '레', testament: 'old' },
  '민수기': { order: 4, abbr: '민', testament: 'old' },
  '신명기': { order: 5, abbr: '신', testament: 'old' },
  '여호수아': { order: 6, abbr: '수', testament: 'old' },
  '사사기': { order: 7, abbr: '삿', testament: 'old' },
  '룻기': { order: 8, abbr: '룻', testament: 'old' },
  '사무엘상': { order: 9, abbr: '삼상', testament: 'old' },
  '사무엘하': { order: 10, abbr: '삼하', testament: 'old' },
  '열왕기상': { order: 11, abbr: '왕상', testament: 'old' },
  '열왕기하': { order: 12, abbr: '왕하', testament: 'old' },
  '역대상': { order: 13, abbr: '대상', testament: 'old' },
  '역대하': { order: 14, abbr: '대하', testament: 'old' },
  '에스라': { order: 15, abbr: '스', testament: 'old' },
  '느헤미야': { order: 16, abbr: '느', testament: 'old' },
  '에스더': { order: 17, abbr: '에', testament: 'old' },
  '욥기': { order: 18, abbr: '욥', testament: 'old' },
  '시편': { order: 19, abbr: '시', testament: 'old' },
  '잠언': { order: 20, abbr: '잠', testament: 'old' },
  '전도서': { order: 21, abbr: '전', testament: 'old' },
  '아가': { order: 22, abbr: '아', testament: 'old' },
  '이사야': { order: 23, abbr: '사', testament: 'old' },
  '예레미야': { order: 24, abbr: '렘', testament: 'old' },
  '예레미야애가': { order: 25, abbr: '애', testament: 'old' },
  '에스겔': { order: 26, abbr: '겔', testament: 'old' },
  '다니엘': { order: 27, abbr: '단', testament: 'old' },
  '호세아': { order: 28, abbr: '호', testament: 'old' },
  '요엘': { order: 29, abbr: '욜', testament: 'old' },
  '아모스': { order: 30, abbr: '암', testament: 'old' },
  '오바댜': { order: 31, abbr: '옵', testament: 'old' },
  '요나': { order: 32, abbr: '욘', testament: 'old' },
  '미가': { order: 33, abbr: '미', testament: 'old' },
  '나훔': { order: 34, abbr: '나', testament: 'old' },
  '하박국': { order: 35, abbr: '합', testament: 'old' },
  '스바냐': { order: 36, abbr: '습', testament: 'old' },
  '학개': { order: 37, abbr: '학', testament: 'old' },
  '스가랴': { order: 38, abbr: '슥', testament: 'old' },
  '말라기': { order: 39, abbr: '말', testament: 'old' },
  // 신약 (New Testament)
  '마태복음': { order: 40, abbr: '마', testament: 'new' },
  '마가복음': { order: 41, abbr: '막', testament: 'new' },
  '누가복음': { order: 42, abbr: '눅', testament: 'new' },
  '요한복음': { order: 43, abbr: '요', testament: 'new' },
  '사도행전': { order: 44, abbr: '행', testament: 'new' },
  '로마서': { order: 45, abbr: '롬', testament: 'new' },
  '고린도전서': { order: 46, abbr: '고전', testament: 'new' },
  '고린도후서': { order: 47, abbr: '고후', testament: 'new' },
  '갈라디아서': { order: 48, abbr: '갈', testament: 'new' },
  '에베소서': { order: 49, abbr: '엡', testament: 'new' },
  '빌립보서': { order: 50, abbr: '빌', testament: 'new' },
  '골로새서': { order: 51, abbr: '골', testament: 'new' },
  '데살로니가전서': { order: 52, abbr: '살전', testament: 'new' },
  '데살로니가후서': { order: 53, abbr: '살후', testament: 'new' },
  '디모데전서': { order: 54, abbr: '딤전', testament: 'new' },
  '디모데후서': { order: 55, abbr: '딤후', testament: 'new' },
  '디도서': { order: 56, abbr: '딛', testament: 'new' },
  '빌레몬서': { order: 57, abbr: '몬', testament: 'new' },
  '히브리서': { order: 58, abbr: '히', testament: 'new' },
  '야고보서': { order: 59, abbr: '약', testament: 'new' },
  '베드로전서': { order: 60, abbr: '벧전', testament: 'new' },
  '베드로후서': { order: 61, abbr: '벧후', testament: 'new' },
  '요한일서': { order: 62, abbr: '요일', testament: 'new' },
  '요한이서': { order: 63, abbr: '요이', testament: 'new' },
  '요한삼서': { order: 64, abbr: '요삼', testament: 'new' },
  '유다서': { order: 65, abbr: '유', testament: 'new' },
  '요한계시록': { order: 66, abbr: '계', testament: 'new' },
} as const;

export type BibleBook = keyof typeof BIBLE_BOOKS;

export interface QTEntry {
  slug: string;
  date: string;
  title: string;
  bibleBook?: BibleBook;
  chapter?: number;
  verses?: string;
  bibleReference?: string;
  content: string;
  reflection: string;
}

export interface QTByBook {
  book: BibleBook;
  bookInfo: typeof BIBLE_BOOKS[BibleBook];
  entries: QTEntry[];
}

// Cache for filename to slug mapping
let slugCache: Map<string, string> | null = null;

/**
 * Build slug cache for all files (handles duplicates with indices)
 */
function buildSlugCache(): Map<string, string> {
  if (slugCache) return slugCache;

  const files = fs.readdirSync(QT_DIR).filter(f => f.endsWith('.md'));
  const dateCount: Map<string, number> = new Map();
  const cache: Map<string, string> = new Map();

  // Sort files to ensure consistent ordering
  files.sort();

  for (const filename of files) {
    const match = filename.match(/^(\d{4}-\d{2}-\d{2})/);
    if (match) {
      const date = match[1];
      const count = dateCount.get(date) || 0;

      // Generate slug: date only for first entry, date-N for duplicates
      const slug = count === 0 ? date : `${date}-${count + 1}`;
      cache.set(filename, slug);
      dateCount.set(date, count + 1);
    } else {
      // Fallback for files without date prefix
      const slug = filename.replace(/\.md$/, '').replace(/[^a-zA-Z0-9-]/g, '-').toLowerCase();
      cache.set(filename, slug);
    }
  }

  slugCache = cache;
  return cache;
}

/**
 * Get slug for a filename
 */
function getSlugForFilename(filename: string): string {
  const cache = buildSlugCache();
  return cache.get(filename) || filename.replace(/\.md$/, '');
}

/**
 * Get filename for a slug
 */
function getFilenameForSlug(slug: string): string | null {
  const cache = buildSlugCache();
  for (const [filename, cachedSlug] of cache) {
    if (cachedSlug === slug) {
      return filename;
    }
  }
  return null;
}

/**
 * Parse QT file content
 */
function parseQTContent(filename: string, content: string): QTEntry {
  const lines = content.split('\n');

  // Extract date and title from filename
  const filenameMatch = filename.match(/^(\d{4}-\d{2}-\d{2})\s+(.+)\.md$/);
  const date = filenameMatch?.[1] || '';
  const title = filenameMatch?.[2] || lines[0]?.trim() || 'Untitled';

  // Generate URL-safe ASCII slug
  const slug = getSlugForFilename(filename);

  // Find Bible reference (e.g., "(창세기 1장 1~5절)")
  let bibleBook: BibleBook | undefined;
  let chapter: number | undefined;
  let verses: string | undefined;
  let bibleReference: string | undefined;

  const refPattern = /\(([가-힣]+)\s*(\d+)장\s*([\d~\-,절]+)?\)/;
  for (const line of lines) {
    const match = line.match(refPattern);
    if (match) {
      const bookName = match[1];
      bibleReference = match[0].replace(/[()]/g, '');
      chapter = parseInt(match[2]);
      verses = match[3]?.replace(/절/g, '');

      // Find matching Bible book
      for (const [name] of Object.entries(BIBLE_BOOKS)) {
        if (bookName === name || bookName.startsWith(name.slice(0, 2))) {
          bibleBook = name as BibleBook;
          break;
        }
      }
      break;
    }
  }

  // Extract verse lines and reflection
  const verseLines: string[] = [];
  const reflectionLines: string[] = [];
  let inVerses = false;
  let pastReference = false;

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i];

    if (refPattern.test(line)) {
      pastReference = true;
      continue;
    }

    if (!pastReference) {
      // Check if it's a numbered verse line
      if (/^\d+\.?\s/.test(line.trim())) {
        inVerses = true;
        verseLines.push(line);
      } else if (inVerses && line.trim()) {
        verseLines.push(line);
      }
    } else {
      // After reference, it's reflection
      if (line.trim() && line.trim() !== 'Suan Lee') {
        reflectionLines.push(line);
      }
    }
  }

  return {
    slug,
    date,
    title,
    bibleBook,
    chapter,
    verses,
    bibleReference,
    content: verseLines.join('\n'),
    reflection: reflectionLines.join('\n'),
  };
}

/**
 * Get all QT entries
 */
export function getAllQTEntries(): QTEntry[] {
  const files = fs.readdirSync(QT_DIR).filter(f => f.endsWith('.md'));

  const entries = files.map(filename => {
    const content = fs.readFileSync(path.join(QT_DIR, filename), 'utf-8');
    return parseQTContent(filename, content);
  });

  // Sort by date (newest first)
  return entries.sort((a, b) => b.date.localeCompare(a.date));
}

/**
 * Get QT entries organized by Bible book
 */
export function getQTByBibleBook(): QTByBook[] {
  const entries = getAllQTEntries();
  const byBook: Map<BibleBook, QTEntry[]> = new Map();

  for (const entry of entries) {
    if (entry.bibleBook) {
      const existing = byBook.get(entry.bibleBook) || [];
      existing.push(entry);
      byBook.set(entry.bibleBook, existing);
    }
  }

  // Sort entries within each book by chapter and date
  for (const [, bookEntries] of byBook) {
    bookEntries.sort((a, b) => {
      if (a.chapter && b.chapter && a.chapter !== b.chapter) {
        return a.chapter - b.chapter;
      }
      return a.date.localeCompare(b.date);
    });
  }

  // Convert to array sorted by Bible book order
  const result: QTByBook[] = [];
  for (const [book, info] of Object.entries(BIBLE_BOOKS)) {
    const bookEntries = byBook.get(book as BibleBook);
    if (bookEntries && bookEntries.length > 0) {
      result.push({
        book: book as BibleBook,
        bookInfo: info,
        entries: bookEntries,
      });
    }
  }

  return result.sort((a, b) => a.bookInfo.order - b.bookInfo.order);
}

/**
 * Get a single QT entry by slug
 */
export function getQTBySlug(slug: string): QTEntry | null {
  const filename = getFilenameForSlug(slug);
  if (!filename) return null;

  const content = fs.readFileSync(path.join(QT_DIR, filename), 'utf-8');
  return parseQTContent(filename, content);
}

/**
 * Get QT slugs for static generation
 */
export function getQTSlugs(): string[] {
  const cache = buildSlugCache();
  return Array.from(cache.values());
}

/**
 * Get QT statistics
 */
export function getQTStats() {
  const entries = getAllQTEntries();
  const byBook = getQTByBibleBook();

  const oldTestament = byBook.filter(b => b.bookInfo.testament === 'old');
  const newTestament = byBook.filter(b => b.bookInfo.testament === 'new');

  const years = new Set(entries.map(e => e.date.slice(0, 4)));

  return {
    total: entries.length,
    booksCount: byBook.length,
    oldTestamentBooks: oldTestament.length,
    newTestamentBooks: newTestament.length,
    oldTestamentEntries: oldTestament.reduce((sum, b) => sum + b.entries.length, 0),
    newTestamentEntries: newTestament.reduce((sum, b) => sum + b.entries.length, 0),
    years: Array.from(years).sort(),
    dateRange: {
      start: entries[entries.length - 1]?.date,
      end: entries[0]?.date,
    },
  };
}
