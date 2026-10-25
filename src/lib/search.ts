export interface SearchItem { type: string; title: string; description: string; href: string; titleEn?: string; descriptionEn?: string; }
export function searchItems(items: SearchItem[], query: string, type = 'all') {
  const terms = query.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
  if (query.trim().length < 2) return [];
  return items.filter((item) => (type === 'all' || item.type === type) && terms.every((term) => `${item.title} ${item.description} ${item.titleEn || ''} ${item.descriptionEn || ''}`.toLocaleLowerCase().includes(term)));
}
