import { bibleBookEnglish, bibleAbbreviations } from './bible';
import { commonEnglish } from './common';
export type EnglishDictionary = Readonly<Record<string, string>>;
const months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];

/** Translate display text only. IDs, URLs, filters and source records stay canonical. */
export function englishText(text: string, dictionary: EnglishDictionary = {}): string {
  const normalized = text.trim().replace(/\s+/g, ' ');
  const translated = dictionary[normalized] ?? commonEnglish[normalized] ?? bibleBookEnglish[normalized];
  if (translated !== undefined) return text.replace(text.trim(), translated);
  if (!/[가-힣]/.test(normalized)) return text;
  const bilingual = normalized.match(/^.*[가-힣].*?\s[\/·]\s([^가-힣]+)$/);
  if (bilingual) return bilingual[1].trim();
  const reviews = normalized.match(/^(.*) - (\d+)건 리뷰$/);
  if (reviews) return `${reviews[1]} - ${reviews[2]} reviews`;
  for (const separator of [' · ', ' → ']) {
    if (!normalized.includes(separator)) continue;
    const parts = normalized.split(separator);
    const result = parts.map(part => englishText(part, dictionary));
    if (result.some((part, index) => part !== parts[index])) return result.join(separator);
  }
  if (/^\[논문 리뷰\]\s*/.test(normalized)) return normalized.replace(/^\[논문 리뷰\]/, '[Paper Review]');
  const books = { ...bibleBookEnglish, ...bibleAbbreviations };
  const bible = Object.keys(books).find(book => normalized.startsWith(book) && /^\s*\d/.test(normalized.slice(book.length)));
  if (bible) return `${books[bible]} ${normalized.slice(bible.length).trim().replace(/장\s*/g, ":").replace(/[절편]/g, "").replace(/~/g, "–")}`;
  const semester = normalized.match(/^(\d{4})년\s*([12])학기$/);
  if (semester) return `${semester[2] === '1' ? 'Spring' : 'Fall'} ${semester[1]}`;
  const date = normalized.match(/^(\d{4})년(?:\s*(\d{1,2})월)?(?:\s*(\d{1,2})일)?$/);
  if (date) return date[2] ? `${months[Number(date[2]) - 1]}${date[3] ? ` ${Number(date[3])},` : ''} ${date[1]}` : date[1];
  const currency = normalized.match(/^([\d,.]+)\s*(억원|만원|원)$/);
  if (currency) return `KRW ${(Number(currency[1].replace(/,/g, '')) * (currency[2] === '억원' ? 100000000 : currency[2] === '만원' ? 10000 : 1)).toLocaleString('en-US')}`;
  return text;
}
