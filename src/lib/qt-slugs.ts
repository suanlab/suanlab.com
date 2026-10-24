/** Retain retired mappings so newly added entries cannot take an old URL. */
export function registerQTSlugs(filenames: string[], existing: Record<string, string>): Record<string, string> {
  const result = { ...existing };
  const used = new Set(Object.values(existing));
  if (used.size !== Object.keys(existing).length) throw new Error('Duplicate registered QT slugs.');
  for (const name of [...filenames].sort()) {
    if (result[name]) continue;
    const date = name.match(/^\d{4}-\d{2}-\d{2}/)?.[0];
    if (!date) throw new Error(`QT filename must start with a date: ${name}`);
    let slug = date, index = 2;
    while (used.has(slug)) slug = `${date}-${index++}`;
    result[name] = slug;
    used.add(slug);
  }
  return result;
}
