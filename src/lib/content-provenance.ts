export interface ContentProvenance {
  method: 'ai-assisted';
  generatedAt: string;
  review: { status: 'pending' | 'reviewed'; reviewer?: string; reviewedAt?: string };
  source: {
    kind: 'arxiv' | 'pdf' | 'topic';
    title: string;
    authors?: string[];
    url?: string;
    identifier?: string;
    sha256?: string;
  };
}

/** PDF download links can contain temporary credentials; retain a hash instead. */
export function publicSourceUrl(value: string): string | undefined {
  if (!value) return undefined;
  const url = new URL(value);
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search) return undefined;
  return url.href;
}

/** Absent provenance means unknown history, never an implicit human review. */
export function parseProvenance(value: unknown): ContentProvenance | undefined {
  if (value === undefined) return undefined;
  const p = value as ContentProvenance;
  if (!p || p.method !== 'ai-assisted' || typeof p.generatedAt !== 'string' || Number.isNaN(Date.parse(p.generatedAt))) throw new Error('Invalid generation provenance');
  if (!p.review || !['pending', 'reviewed'].includes(p.review.status)) throw new Error('Invalid review status');
  if (p.review.status === 'reviewed' && (typeof p.review.reviewer !== 'string' || !p.review.reviewer.trim() || typeof p.review.reviewedAt !== 'string' || Number.isNaN(Date.parse(p.review.reviewedAt)))) throw new Error('Reviewed content requires a reviewer and review date');
  if (!p.source || !['arxiv', 'pdf', 'topic'].includes(p.source.kind) || typeof p.source.title !== 'string' || !p.source.title.trim()) throw new Error('Invalid source provenance');
  if (p.source.authors && (!Array.isArray(p.source.authors) || p.source.authors.some(a => typeof a !== 'string'))) throw new Error('Invalid source authors');
  if (p.source.url) {
    const url = new URL(p.source.url);
    if (!['https:', 'http:'].includes(url.protocol) || url.username || url.password) throw new Error('Source URL must be a public HTTP(S) URL without credentials');
  }
  if (p.source.kind === 'arxiv' && (!p.source.identifier || !/^(?:\d{4}\.\d{4,5}|[a-z-]+\/\d{7})(v\d+)?$/i.test(p.source.identifier) || p.source.url !== `https://arxiv.org/abs/${p.source.identifier}`)) throw new Error('arXiv source requires a matching identifier and canonical URL');
  if (p.source.kind === 'pdf' && !p.source.url && !/^[a-f0-9]{64}$/.test(p.source.sha256 || '')) throw new Error('Local PDF source requires a content hash');
  return p;
}
