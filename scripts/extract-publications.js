const fs = require('fs');
const path = require('path');

const html = fs.readFileSync(path.join(__dirname, '../../WWW/publication/index.html'), 'utf-8');

const publications = [];

// Split by toggle end markers and process each
const parts = html.split(/<\/div><!-- \/toggle -->/);

let id = 1;

for (const part of parts) {
  // Find the last toggle start in this part
  const startIdx = part.lastIndexOf('<div class="toggle mix');
  if (startIdx === -1) continue;

  const block = part.substring(startIdx);

  // Extract type
  const typeMatch = block.match(/<div class="toggle mix ([a-z]+)/);
  if (!typeMatch) continue;
  const type = typeMatch[1];

  // Extract label
  const labelMatch = block.match(/<label>([\s\S]*?)<\/label>/);
  if (!labelMatch) continue;
  const labelHtml = labelMatch[1].trim();

  // Extract toggle-content
  const contentMatch = block.match(/<div class="toggle-content">([\s\S]*)/);
  const contentHtml = contentMatch ? contentMatch[1].trim() : '';

  // Extract badge
  const badgeMatch = labelHtml.match(/<span[^>]*>\[([^\]]+)\]<\/span>/);
  const badge = badgeMatch ? '[' + badgeMatch[1] + ']' : undefined;

  // Extract award
  const awardMatch = labelHtml.match(/<span[^>]*>\(([^)]+상[^)]*)\)<\/span>/);
  const award = awardMatch ? '(' + awardMatch[1] + ')' : undefined;

  // Extract impact
  const impactMatch = labelHtml.match(/\(IF: [^)]+\)/);
  const impact = impactMatch ? impactMatch[0] : undefined;

  // Extract title - handle both ASCII quotes and Unicode curly quotes
  let title = '';
  // Try ASCII quotes first: <b>"..."</b>
  let titleMatch = labelHtml.match(/<b>"([^"]+)"<\/b>/);
  if (!titleMatch) {
    // Try Unicode curly quotes: <b>\u201C...\u201D</b>
    titleMatch = labelHtml.match(/<b>\u201C([^\u201D]+)\u201D<\/b>/);
  }
  if (titleMatch) {
    title = titleMatch[1].replace(/,\s*$/, '').trim();
  }

  if (!title) continue;

  // Extract authors (before the title <b>" or <b>\u201C)
  let authors = '';
  let idx = labelHtml.indexOf('<b>"');
  if (idx === -1) {
    idx = labelHtml.indexOf('<b>\u201C');
  }
  if (idx !== -1) {
    authors = labelHtml.substring(0, idx)
      .replace(/<span[^>]*>[^<]*<\/span>\s*/g, '')
      .replace(/<\/?b>/g, '')
      .replace(/,\s*$/, '')
      .trim();
  }

  // Extract venue
  const venueMatch = contentHtml.match(/<i class="fa fa-institution"><\/i>\s*<span[^>]*>([^<]+)<\/span>/);
  const venue = venueMatch ? venueMatch[1].trim() : '';

  // Extract date
  const dateMatch = contentHtml.match(/<i class="fa fa-calendar"><\/i>\s*<span[^>]*>([^<]+)<\/span>/);
  const date = dateMatch ? dateMatch[1].trim() : '';

  // Extract abstract
  const abstractMatch = contentHtml.match(/<p class="lead">Abstract<\/p>\s*<p[^>]*>([\s\S]*?)<\/p>/);
  let abstract = abstractMatch ? abstractMatch[1].trim().replace(/\s+/g, ' ') : undefined;
  if (abstract === '') abstract = undefined;

  // Extract keywords
  const keywordsMatch = contentHtml.match(/<p class="lead">Keywords<\/p>\s*<p[^>]*>([\s\S]*?)<\/p>/);
  let keywords = keywordsMatch ? keywordsMatch[1].trim().replace(/\s+/g, ' ') : undefined;
  if (keywords === '') keywords = undefined;

  // Extract URL
  const urlMatch = contentHtml.match(/<a href="([^"]+)" target="_blank">/);
  const url = urlMatch ? urlMatch[1] : undefined;

  const pub = {
    id: id++,
    type: type,
    title: title,
    authors: authors,
    venue: venue,
    date: date,
  };

  if (badge) pub.badge = badge;
  else if (award) pub.badge = award;
  if (impact) pub.impact = impact;
  if (abstract) pub.abstract = abstract;
  if (keywords) pub.keywords = keywords;
  if (url) pub.url = url;

  publications.push(pub);
}

// Count by type
const counts = {};
publications.forEach(p => {
  counts[p.type] = (counts[p.type] || 0) + 1;
});

// Generate TypeScript
const output = `export type PublicationType = 'journal' | 'conference' | 'djournal' | 'dconference' | 'book' | 'patent' | 'report' | 'column';

export interface Publication {
  id: number;
  type: PublicationType;
  title: string;
  authors: string;
  venue: string;
  date: string;
  abstract?: string;
  keywords?: string;
  url?: string;
  badge?: string;
  impact?: string;
}

export const publicationTypes: { key: PublicationType | 'all'; label: string; count: number }[] = [
  { key: 'all', label: 'All', count: ${publications.length} },
  { key: 'journal', label: 'International Journal', count: ${counts.journal || 0} },
  { key: 'conference', label: 'International Conference', count: ${counts.conference || 0} },
  { key: 'djournal', label: 'Domestic Journal', count: ${counts.djournal || 0} },
  { key: 'dconference', label: 'Domestic Conference', count: ${counts.dconference || 0} },
  { key: 'book', label: 'Book', count: ${counts.book || 0} },
  { key: 'patent', label: 'Patent', count: ${counts.patent || 0} },
  { key: 'report', label: 'Report', count: ${counts.report || 0} },
  { key: 'column', label: 'Column', count: ${counts.column || 0} },
];

export const publications: Publication[] = ${JSON.stringify(publications, null, 2)};

export const getPublicationsByType = (type: PublicationType | 'all'): Publication[] => {
  if (type === 'all') return publications;
  return publications.filter((p) => p.type === type);
};
`;

fs.writeFileSync(path.join(__dirname, '../src/data/publications/index.ts'), output);
console.log('Extracted ' + publications.length + ' publications');
console.log('Counts by type:', counts);
