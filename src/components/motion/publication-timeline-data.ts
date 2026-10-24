export function publicationYearCounts(publications: readonly { sortYear: number }[]) {
  const counts = new Map<number, number>();
  for (const { sortYear } of publications) {
    if (!Number.isInteger(sortYear) || sortYear < 1900 || sortYear > 9999) continue;
    counts.set(sortYear, (counts.get(sortYear) ?? 0) + 1);
  }
  return [...counts].sort(([a], [b]) => a - b).map(([year, count]) => ({ year, count }));
}
