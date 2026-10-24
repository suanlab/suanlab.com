export interface BlogQuery { q: string; category: string | null; tag: string | null; page: number }
export const emptyBlogQuery: BlogQuery = { q: '', category: null, tag: null, page: 1 };

export function readBlogQuery(params: URLSearchParams): BlogQuery {
  const page = params.get('page') ?? '';
  return { q: params.get('q') ?? '', category: params.get('category') || null, tag: params.get('tag') || null,
    page: /^\d+$/.test(page) && Number.isSafeInteger(Number(page)) && Number(page) > 0 ? Number(page) : 1 };
}

/** Keep unrelated query parameters and URL fragments when changing list filters. */
export function updateListQuery(url: URL, values: Record<string, string | null>): URL {
  const next = new URL(url);
  for (const [key, value] of Object.entries(values)) {
    if (value) next.searchParams.set(key, value); else next.searchParams.delete(key);
  }
  return next;
}
