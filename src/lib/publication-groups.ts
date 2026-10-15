export interface ConferenceVenue {
  id: string;
  acronym: string;
  name: string;
  homepageUrl?: string;
  aliases?: string[];
}

interface ConferencePaper {
  type: string;
  venue: string;
  date: string;
  badge?: string;
}

export interface ConferencePublicationGroup<T extends ConferencePaper> {
  id: string;
  conference: ConferenceVenue;
  year: number;
  month: number;
  papers: { publication: T; track: 'main' | 'findings' }[];
}

const months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'];
const normalize = (value: string) => value.replace(/[\u2010-\u2015]/g, '-');
const escapePattern = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

function publicationMonth(date: string): number {
  const english = months.findIndex(month => new RegExp(`\\b${month.slice(0, 3)}(?:${month.slice(3)})?\\b`, 'i').test(date));
  if (english !== -1) return english + 1;
  const numeric = date.match(/\b\d{4}[.-](\d{1,2})(?:[.-]|$)/) ?? date.match(/(\d{1,2})월/);
  const month = Number(numeric?.[1]);
  return month >= 1 && month <= 12 ? month : 0;
}

/** Group only explicitly selected conference venues; preserve Findings attribution.
 * Newest year/month first, with editorial venue order breaking date ties.
 */
export function groupConferencePublications<T extends ConferencePaper>(
  publications: readonly T[],
  venues: readonly ConferenceVenue[],
  { includeFindings = true }: { includeFindings?: boolean } = {},
): ConferencePublicationGroup<T>[] {
  const groups = new Map<string, ConferencePublicationGroup<T>>();
  for (const publication of publications) {
    if (publication.type !== 'conference') continue;
    const source = normalize(`${publication.badge ?? ''} ${publication.venue}`);
    if (/workshop|student research|companion|demonstration|\bdemo\b|\btutorial\b/i.test(source)) continue;
    const track = /\bfindings\b/i.test(source) ? 'findings' : 'main';
    if (track === 'findings' && !includeFindings) continue;
    const conference = venues.find(venue => [venue.acronym, venue.name, ...(venue.aliases ?? [])].some(alias =>
      new RegExp(`(^|[^a-z0-9])${escapePattern(normalize(alias))}(?=$|[^a-z0-9])`, 'i').test(source),
    ));
    if (!conference) continue;
    const year = Number(publication.badge?.match(/\b(?:19|20)\d{2}\b/)?.[0] ?? publication.venue.match(/\b(?:19|20)\d{2}\b/)?.[0] ?? publication.date.match(/\b(?:19|20)\d{2}\b/)?.[0]);
    if (!year) continue;
    const id = `${conference.id}-${year}`;
    let group = groups.get(id);
    if (!group) {
      group = { id, conference, year, month: 0, papers: [] };
      groups.set(id, group);
    }
    group.month = Math.max(group.month, publicationMonth(publication.date));
    group.papers.push({ publication, track });
  }
  const order = new Map(venues.map((venue, index) => [venue.id, index]));
  return [...groups.values()].sort((a, b) => b.year - a.year || b.month - a.month || order.get(a.conference.id)! - order.get(b.conference.id)!);
}
